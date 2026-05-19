"""Build a (multi-)player pair-score training dataset.

Output: a PairHead-ready cache where every stored snapshot carries
  * single-frame inputs (planet/comet/fleet features, masks, routing,
    ships_log, eta_norm, is_comet). ``CachedPairDataset`` stacks the
    T=6 history window lazily at training time so the cache stays small,
  * ``pair_labels (P, P) bool`` — current-turn-only matrix where
    ``[s, t] = True`` iff the expert launched a fleet from planet
    slot ``s`` to planet slot ``t`` on this turn (multi-positive when
    the expert ran a coalition launch),
  * ``pair_ships (P, P) int32`` — total ships sent ``s → t`` this turn
    (summed across multi-fleet launches). Zero elsewhere. Downstream
    ``pair_frac_head`` training derives per-source fractions from this.
  * ``pair_valid (P, P) bool`` =
    ``planet_mask[s] & planet_mask[t] & (s != t)``.

The action featurizer only stores the *first* expert launch per turn,
so coalition launches are reconstructed here by walking the raw
replay JSONs: any fleet that exists in ``obs[t+1].fleets`` but not in
``obs[t].fleets`` and is owned by the learner seat counts as a launch,
with source from ``F_FROM``, ships from ``F_SHIPS``, and target
resolved via the same ``_resolve_target`` helper used by the v2 fleet
featurizer.

Slot indexing matches the planet CSV's row order (== ``obs.planets``
order, == raw ``planet_id``) so the pair matrix lines up with
``EntitySnapshotDataset``'s ``pid_to_idx`` 1:1.

Multi-player mixing: ``--player`` accepts a comma-separated list. The
build concatenates per-player replays, snapshots, and keys, derives a
deterministic mix slug from sorted-player names, and (when
``--keep-non-acted`` is set) applies a per-player acted-ratio floor by
randomly subsampling non-acted snapshots.

Cache layout::

    data/datasets/_pair_cache/<slug>_T6/
        <slug>_T6_p64_f1024_acted.pt        # default: acted only
        <slug>_T6_p64_f1024_all.pt           # opt-in via --keep-non-acted

Each cache file is a single .pt with::

    {
        "config":    {player (or players list), history_offsets,
                      max_planets, max_fleets, keep_non_acted,
                      acted_min_ratio, seed, stem_to_player,
                      per_player_counts},
        "snapshots": list[dict[str, torch.Tensor]],
        "keys":      list[tuple[str, int]],
        "acted_indices": list[int],
    }

Run::

    python scripts/build_pair_dataset_orbital_occle.py
    python scripts/build_pair_dataset_orbital_occle.py --keep-non-acted
    python scripts/build_pair_dataset_orbital_occle.py \\
        --player bowwowforeach,Ebi \\
        --keep-non-acted \\
        --acted-min-ratio 0.30
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from agents.physics_utils import F_FROM, F_ID, F_OWNER, F_SHIPS  # noqa: E402
from agents.transformer_v2.featurizer.fleet_featurizer import (  # noqa: E402
    _resolve_target,
)
from agents.transformer_v2.paths import (  # noqa: E402
    DATASETS_ROOT,
    ENTITY_DATASET_DIR,
    FLEET_DATASET_DIR,
    PLANET_DATASET_DIR,
    REPLAYS_DIR,
)
from agents.transformer_v2.pretrain.entity_encoder import (  # noqa: E402
    EntitySnapshotDataset,
    HISTORY_OFFSETS_T6,
    load_comet_lookup,
    merge_per_stem_comet_csvs,
    COMET_PER_STEM_DIR,
)
from agents.transformer_v2.pretrain.pair_score import (  # noqa: E402
    player_replay_stems,
)


PLAYER: str = "Orbital Occle"
#: Root directory for per-player pair caches. Each player's cache lives
#: in a subdirectory named after the player slug + "_T6".
PAIR_CACHE_ROOT: Path = DATASETS_ROOT / "_pair_cache"

#: Default RNG seed for the non-acted subsample. Override via ``--seed``.
DEFAULT_SEED: int = 1729


def _player_slug(player: str | list[str]) -> str:
    """Filesystem-safe slug.

    Single player: drops spaces; ASCII names pass through unchanged so
    old single-player paths keep resolving (e.g. ``"Orbital Occle"`` →
    ``"OrbitalOccle"``).

    Mixed players: sorted alphabetically (case-insensitive) and joined
    with ``_`` so the slug is deterministic regardless of input order.
    """
    if isinstance(player, str):
        return player.replace(" ", "")
    names = sorted({p.strip() for p in player if p.strip()}, key=str.lower)
    if len(names) == 1:
        return names[0].replace(" ", "")
    return "_".join(n.replace(" ", "") for n in names)


def _cache_dir_for(player: str | list[str]) -> Path:
    """Per-player (or mixed-player) cache subdir under
    :data:`PAIR_CACHE_ROOT`."""
    return PAIR_CACHE_ROOT / f"{_player_slug(player)}_T6"


# Back-compat: default OO cache path used by the original notebook.
PAIR_CACHE_DIR: Path = _cache_dir_for(PLAYER)


# ---------- Replay → (src, tgt) pair edges per turn ----------
def _parse_seat_from_stem(stem: str) -> int | None:
    """Stem like ``75365996_2_0`` → learner seat ``0``. ``None`` for
    locally-played stems that don't carry a seat in the filename.
    """
    if stem.startswith("local"):
        return None
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def _extract_pair_edges(
    replay_path: Path,
    *,
    learner_slot: int,
) -> tuple[str, dict[int, list[tuple[int, int, int]]]]:
    """Walk one replay and return ``episode_id`` plus a
    ``{turn: [(src_pid, tgt_pid, ships), ...]}`` map for every learner
    launch.

    Coalition launches naturally appear as multiple entries under the
    same turn. ``ships`` is the raw fleet's ``F_SHIPS`` count at the
    moment of launch (i.e. read from the new fleet that just appeared
    in ``obs[t+1].fleets``). The pair-frac head sums these per ``(s,
    t)`` cell and divides downstream — we deliberately do not normalize
    here so a single multi-fleet launch retains its launch-time ship
    counts intact.
    """
    with gzip.open(replay_path, "rt") as fh:
        replay = json.load(fh)
    episode_id = replay.get("id") or replay_path.stem.split(".")[0]
    steps = replay.get("steps") or []

    edges: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for t in range(len(steps) - 1):
        step = steps[t]
        next_step = steps[t + 1]
        if not step or not next_step:
            continue
        seat = (
            step[learner_slot] if len(step) > learner_slot else step[0]
        )
        next_seat = (
            next_step[learner_slot]
            if len(next_step) > learner_slot
            else next_step[0]
        )
        obs = seat.get("observation") if isinstance(seat, dict) else None
        next_obs = (
            next_seat.get("observation") if isinstance(next_seat, dict) else None
        )
        if not obs or not next_obs:
            continue

        prev_ids = {int(f[F_ID]) for f in (obs.get("fleets") or [])}
        next_planets = next_obs.get("planets") or []
        for f in next_obs.get("fleets") or []:
            fid = int(f[F_ID])
            if fid in prev_ids:
                continue
            # Learner-only filter: pair labels mean "the expert
            # launched this", never "anyone launched this".
            if int(f[F_OWNER]) != learner_slot:
                continue
            src_pid = int(f[F_FROM])
            tgt_pid, _eta, _planet = _resolve_target(f, next_planets)
            if tgt_pid is None:
                # Off-board trajectory; can't supervise as a pair.
                continue
            ships = int(f[F_SHIPS])
            edges[t].append((src_pid, int(tgt_pid), ships))
    return episode_id, dict(edges)


def build_replay_edges(
    player: str,
    replay_dir: Path,
) -> dict[str, dict]:
    """Per-stem ``{stem: {"episode_id": str, "edges": {turn: [(src_pid,
    tgt_pid, ships), ...]}}}`` for one player.

    Keyed by stem (== filename without ``.json.gz``) so the dataset
    class can join by stem against planet/fleet/entity CSV names.
    """
    pdir = replay_dir / player
    if not pdir.is_dir():
        raise FileNotFoundError(f"no replay dir for {player!r} at {pdir}")
    replays = sorted(pdir.glob("*.json.gz"))
    if not replays:
        raise FileNotFoundError(
            f"no .json.gz replays under {pdir} for player={player!r}"
        )
    out: dict[str, dict] = {}
    t_start = time.time()
    for i, rp in enumerate(replays):
        stem = rp.name[: -len(".json.gz")]
        seat = _parse_seat_from_stem(stem)
        if seat is None:
            continue
        try:
            episode_id, edges = _extract_pair_edges(rp, learner_slot=seat)
        except Exception as e:
            print(f"    [replay] {stem}: error parsing — {e}; skipping", flush=True)
            continue
        # The edge map is keyed by stem internally so we can recover it
        # in the dataset by joining against planet/fleet/entity CSV
        # stems. ``episode_id`` is the actual UUID inside the replay,
        # which is what ``EntitySnapshotDataset.keys[i]`` carries — we
        # also stash an episode_id ↔ stem alias dict so the lookup
        # works either way.
        out[stem] = {"episode_id": episode_id, "edges": edges}
        if (i + 1) % 25 == 0 or (i + 1) == len(replays):
            dt = time.time() - t_start
            print(
                f"    [replay] {i + 1}/{len(replays)} stems "
                f"({dt:.1f}s, {(i + 1) / max(dt, 1e-3):.1f}/s)",
                flush=True,
            )
    return out


def build_replay_edges_multi(
    players: list[str],
    replay_dir: Path,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Build replay edges across multiple players.

    Returns ``(replay_edges, stem_to_player)``. ``replay_edges`` is the
    merged per-stem map (stems are unique because they include the
    Kaggle episode id + seat, but we still guard against accidental
    collisions). ``stem_to_player`` maps every stem to its source
    player so downstream code can run per-player accounting and apply
    the per-player acted-ratio floor.
    """
    merged: dict[str, dict] = {}
    stem_to_player: dict[str, str] = {}
    for player in players:
        print(f"[build] walking replays for player={player!r} ...", flush=True)
        per_player = build_replay_edges(player, replay_dir)
        for stem, payload in per_player.items():
            if stem in merged:
                # Should not happen — replay stems are episode-id based.
                # Keep the first seen mapping and warn.
                print(
                    f"    [stems] WARN duplicate stem {stem!r} across "
                    f"players {stem_to_player[stem]!r} and {player!r}; "
                    f"keeping the first.",
                    flush=True,
                )
                continue
            merged[stem] = payload
            stem_to_player[stem] = player
    return merged, stem_to_player


# ---------- Dataset class ----------
class OOPairDataset(EntitySnapshotDataset):
    """:class:`EntitySnapshotDataset` + per-snapshot ``pair_labels`` /
    ``pair_valid`` ``(P, P)`` tensors.

    The pair labels are built from the per-stem edge map produced by
    :func:`build_replay_edges`. Multi-launch turns yield multiple
    ``True`` cells in the same matrix. ``pair_valid`` is the broadcast
    ``planet_mask[s] & planet_mask[t] & (s != t)`` matrix — the
    expected mask for any per-pair loss (CE over flattened P² grid,
    BCE, etc.).

    Slot indexing rule: planet CSVs are emitted in ``obs.planets``
    order, so ``pid_to_idx[planet_id] == planet_id`` for every row the
    base class processes. When a learner launch's source/target falls
    outside the first ``max_planets`` slots (rare), the pair is
    dropped — supervising on a slot the encoder can't see would just
    be noise.
    """

    def __init__(
        self,
        planet_csv_paths: list,
        fleet_csv_paths: list,
        entity_csv_paths: list,
        *,
        max_planets: int = 64,
        max_fleets: int = 1024,
        learner_slot: int = 0,
        num_players: int = 4,
        comet_lookup: dict | None = None,
        history_offsets: tuple[int, ...] = HISTORY_OFFSETS_T6,
        # ``{stem: {"episode_id": str, "edges": {turn: [(src_pid, tgt_pid), ...]}}}``
        replay_edges: dict[str, dict] | None = None,
        keep_non_acted: bool = False,
        num_load_workers: int | None = None,
    ):
        if replay_edges is None:
            raise ValueError("OOPairDataset requires replay_edges")
        self._replay_edges = replay_edges
        # Built lazily inside ``_filter_stems`` — episode_id → edges
        # map, so ``_build_snapshot`` can join by the same key the
        # base class uses for its (episode_id, turn) snapshot key.
        self._edges_by_episode_id: dict[str, dict[int, list[tuple[int, int]]]] = {}
        for stem, payload in replay_edges.items():
            self._edges_by_episode_id[payload["episode_id"]] = payload["edges"]
        self.keep_non_acted = bool(keep_non_acted)

        super().__init__(
            planet_csv_paths=planet_csv_paths,
            fleet_csv_paths=fleet_csv_paths,
            entity_csv_paths=entity_csv_paths,
            max_planets=max_planets,
            max_fleets=max_fleets,
            learner_slot=learner_slot,
            num_players=num_players,
            comet_lookup=comet_lookup,
            history_offsets=history_offsets,
            num_load_workers=num_load_workers,
        )

        # Keep every snapshot in ``self.snapshots`` even when the
        # training target is acted-only. T=6 history lookup needs
        # non-acted context frames; dropping them corrupts most
        # ``t-1..t-5`` slots into zero frames. ``acted_indices`` is the
        # row filter downstream training should use when
        # ``keep_non_acted=False``.
        self.acted_indices = self._compute_acted_indices()
        if not self.keep_non_acted:
            dropped = len(self.snapshots) - len(self.acted_indices)
            print(
                f"    [acted-index] training rows {len(self.acted_indices)}/"
                f"{len(self.snapshots)}; retaining {dropped} non-acted "
                f"context frames for T-history",
                flush=True,
            )

    def _filter_stems(self, stems: list[str]) -> list[str]:
        """Only keep stems for which we managed to extract replay edges
        (drops the rare malformed-replay case)."""
        keep = [s for s in stems if s in self._replay_edges]
        dropped = len(stems) - len(keep)
        if dropped:
            print(
                f"    [stems] dropping {dropped}/{len(stems)} stems without "
                f"replay edges; {len(keep)} kept",
                flush=True,
            )
        return keep

    def _build_snapshot(
        self,
        key: tuple[str, int],
        planet_rows: list[dict[str, str]],
        fleet_rows: list[dict[str, str]],
        entity_rows: list[dict[str, str]],
    ) -> dict[str, torch.Tensor]:
        snapshot = super()._build_snapshot(
            key, planet_rows, fleet_rows, entity_rows,
        )
        P = self.max_planets
        episode_id, turn = key

        # Slot indexing mirrors the base class's ``pid_to_idx`` build
        # so the pair matrix lines up with the encoder's planet axis.
        pid_to_idx: dict[int, int] = {}
        for i, row in enumerate(planet_rows[:P]):
            pid_to_idx[int(row["planet_id"])] = i

        pair_labels = torch.zeros(P, P, dtype=torch.bool)
        pair_ships = torch.zeros(P, P, dtype=torch.int32)
        edges_for_episode = self._edges_by_episode_id.get(episode_id) or {}
        for edge in edges_for_episode.get(turn, []):
            # ``edge`` is (src_pid, tgt_pid, ships); legacy edge maps
            # without ships fall through with a 2-tuple, treated as
            # ships=0 so callers built before this commit still load.
            if len(edge) == 3:
                src_pid, tgt_pid, ships = edge
            else:
                src_pid, tgt_pid = edge
                ships = 0
            s = pid_to_idx.get(src_pid)
            t = pid_to_idx.get(tgt_pid)
            if s is None or t is None:
                continue
            if 0 <= s < P and 0 <= t < P:
                pair_labels[s, t] = True
                # Sum across multi-fleet coalition launches that share
                # the same (src, tgt) cell. Raw ship counts, not
                # fractions — the head computes per-source frac itself.
                pair_ships[s, t] = pair_ships[s, t] + int(ships)

        # ``pair_valid`` mirrors planet_mask × planet_mask, with the
        # self-pair diagonal masked off (the expert never launches
        # from a planet to itself).
        planet_mask = snapshot["planet_mask"]
        pair_valid = planet_mask.unsqueeze(1) & planet_mask.unsqueeze(0)
        pair_valid.fill_diagonal_(False)

        snapshot["pair_labels"] = pair_labels
        snapshot["pair_valid"] = pair_valid
        snapshot["pair_ships"] = pair_ships
        return snapshot

    def _compute_acted_indices(self) -> list[int]:
        """Rows whose current-turn pair set is non-empty.

        These are the supervised rows for acted-only behavior cloning.
        They are stored as indices rather than used to delete snapshots
        so history stacking can still retrieve non-acted context turns.
        """
        return [
            i for i, snap in enumerate(self.snapshots)
            if bool(snap["pair_labels"].any())
        ]


# ---------- Cache I/O ----------
def cache_path(
    *,
    player: str | list[str],
    max_planets: int,
    max_fleets: int,
    keep_non_acted: bool,
    cache_dir: Path | None = None,
) -> Path:
    """Per-player (or mixed-player) cache path.

    ``cache_dir`` defaults to ``_cache_dir_for(player)`` so the file
    lands under ``data/datasets/_pair_cache/<slug>_T6/``. ``player``
    may be a string (single-player) or a list of player names (mixed
    build); the slug is derived deterministically from sorted names.
    """
    tag = "all" if keep_non_acted else "acted"
    safe_player = _player_slug(player)
    name = f"{safe_player}_T6_p{max_planets}_f{max_fleets}_{tag}.pt"
    if cache_dir is None:
        cache_dir = _cache_dir_for(player)
    return cache_dir / name


def save_cache(dataset: OOPairDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``player`` is recorded from the dataset itself (set at
    # ``build_dataset`` time) so a non-OO build doesn't get mislabeled
    # by the legacy module-level ``PLAYER`` constant. For mixed builds
    # we also stash the full list of players, the stem→player map, and
    # the per-player snapshot counts so downstream consumers can run
    # per-player accounting from the cache alone.
    players = getattr(dataset, "players", None)
    config: dict = {
        "player": getattr(dataset, "player", PLAYER),
        "history_offsets": list(dataset.history_offsets or ()),
        "max_planets": dataset.max_planets,
        "max_fleets": dataset.max_fleets,
        "keep_non_acted": dataset.keep_non_acted,
    }
    if players is not None and len(players) > 1:
        config["players"] = list(players)
    extra = getattr(dataset, "build_meta", None)
    if extra:
        config.update(extra)
    payload = {
        "config": config,
        "snapshots": dataset.snapshots,
        "keys": dataset.keys,
        "acted_indices": getattr(dataset, "acted_indices", None),
    }
    torch.save(payload, path)


def load_cache(path: Path) -> dict:
    """Convenience loader for downstream training code.

    Returns the raw payload dict; the training loop can reconstruct
    an ``OOPairDataset``-like in-memory holder however it likes
    (typically ``CachedPairDataset(payload)`` wrapper).
    """
    return torch.load(path, weights_only=False)


class CachedPairDataset(torch.utils.data.Dataset):
    """Drop-in dataset for the on-disk pair cache.

    Mirrors :class:`OOPairDataset.__getitem__` semantics. The cache
    stores **single-frame** snapshots; when ``history_offsets`` is
    present, this wrapper lazily stacks the requested T frames at
    ``__getitem__`` time. ``pair_labels`` / ``pair_valid`` stay
    current-turn-only.
    """

    def __init__(self, payload_or_path):
        if isinstance(payload_or_path, (str, Path)):
            payload = load_cache(Path(payload_or_path))
        else:
            payload = payload_or_path
        self.config = payload["config"]
        self.snapshots = payload["snapshots"]
        self.keys = payload["keys"]
        self.history_offsets = tuple(self.config.get("history_offsets") or ())
        self.cache_has_acted_indices = "acted_indices" in payload
        acted = payload.get("acted_indices")
        if acted is None:
            # Backward-compatible fallback for old acted-only caches.
            acted = [
                i for i, snap in enumerate(self.snapshots)
                if bool(snap["pair_labels"].any())
            ]
        self.acted_indices = list(acted)

        # Lazy ``(episode_id, turn) → idx`` for history stacking at
        # __getitem__ time (same trick the base class uses).
        self._key_to_idx = {k: i for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.snapshots)

    # _STACK_KEYS lifted from EntitySnapshotDataset so a cached
    # snapshot rebuild matches what the live dataset would emit.
    _STACK_KEYS: tuple[str, ...] = EntitySnapshotDataset._STACK_KEYS

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cur = self.snapshots[idx]
        if not self.history_offsets:
            return cur
        ep, t = self.keys[idx]
        history: list[dict | None] = []
        for off in self.history_offsets:
            prev_idx = self._key_to_idx.get((ep, t - off))
            history.append(
                self.snapshots[prev_idx] if prev_idx is not None else None
            )
        out: dict[str, torch.Tensor] = {}
        for key, val in cur.items():
            if key in self._STACK_KEYS:
                stacked: list[torch.Tensor] = []
                for snap in history:
                    if snap is None:
                        if val.dtype == torch.long:
                            zeros = (
                                torch.full_like(val, -1)
                                if "_idx" in key else torch.zeros_like(val)
                            )
                        else:
                            zeros = torch.zeros_like(val)
                        stacked.append(zeros)
                    else:
                        stacked.append(snap[key])
                out[key] = torch.stack(stacked, dim=0)
            else:
                out[key] = val
        return out


# ---------- Build entry point ----------
def _apply_acted_ratio_floor(
    dataset: OOPairDataset,
    *,
    snapshot_to_player: list[str],
    acted_min_ratio: float,
    seed: int,
) -> dict[str, dict[str, int]]:
    """Per-player non-acted subsampling to enforce
    ``acted/(acted+non_acted) >= acted_min_ratio``.

    Drops snapshots (and their keys) in place. Snapshots are categorized
    by whether ``pair_labels.any()`` is true. For each player, if the
    acted-ratio is already at or above the threshold, all snapshots are
    kept; otherwise non-acted rows are uniformly subsampled (RNG seeded
    by ``seed`` + the player name) down to the smallest count that
    satisfies the floor.

    Returns a ``{player: {"acted": int, "non_acted_before": int,
    "non_acted_after": int}}`` stats dict for the build log.

    .. note::
       Dropping non-acted snapshots reduces the T-history context for
       neighbouring rows. This is the explicit trade-off for the
       ``--keep-non-acted`` mode: non-acted rows are training rows, and
       too many of them swamps the loss, so we cap their share.
    """
    # Partition snapshot indices by player.
    by_player: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"acted": [], "non_acted": []}
    )
    for i, snap in enumerate(dataset.snapshots):
        bucket = "acted" if bool(snap["pair_labels"].any()) else "non_acted"
        by_player[snapshot_to_player[i]][bucket].append(i)

    drop_indices: set[int] = set()
    stats: dict[str, dict[str, int]] = {}
    for player, buckets in sorted(by_player.items()):
        acted = buckets["acted"]
        non_acted = buckets["non_acted"]
        n_a = len(acted)
        n_na_before = len(non_acted)
        total = n_a + n_na_before
        if total == 0:
            stats[player] = {
                "acted": 0,
                "non_acted_before": 0,
                "non_acted_after": 0,
            }
            continue
        cur_ratio = n_a / max(total, 1)
        # Solve n_a / (n_a + k) >= r → k <= n_a/r - n_a  (r > 0).
        if acted_min_ratio <= 0.0 or n_a == 0:
            keep_na = n_na_before
        else:
            max_na = int(n_a / acted_min_ratio - n_a)
            keep_na = min(n_na_before, max(0, max_na))
        if keep_na < n_na_before:
            # Per-player seed string keeps the subsample deterministic
            # AND independent across players (so adding a new player
            # to a mixed build doesn't reshuffle the others).
            rng = random.Random(f"{seed}:{player}")
            kept = set(rng.sample(non_acted, keep_na))
            for idx in non_acted:
                if idx not in kept:
                    drop_indices.add(idx)
        new_ratio = n_a / max(n_a + keep_na, 1)
        stats[player] = {
            "acted": n_a,
            "non_acted_before": n_na_before,
            "non_acted_after": keep_na,
        }
        print(
            f"    [ratio] player={player!r} acted={n_a} "
            f"non_acted: {n_na_before} → {keep_na} "
            f"(ratio {cur_ratio:.3f} → {new_ratio:.3f}, "
            f"floor={acted_min_ratio:.2f})",
            flush=True,
        )

    if not drop_indices:
        return stats

    # Drop in place. Snapshots, keys, and snapshot_to_player stay in
    # sync; the dataset's ``acted_indices`` is rebuilt below by
    # ``_compute_acted_indices``.
    keep_mask = [i not in drop_indices for i in range(len(dataset.snapshots))]
    dataset.snapshots = [
        s for s, k in zip(dataset.snapshots, keep_mask) if k
    ]
    dataset.keys = [k for k, m in zip(dataset.keys, keep_mask) if m]
    # ``snapshot_to_player`` is a list passed in by the caller; mutate
    # in place so caller can rebuild per-player counts on the post-drop
    # dataset.
    snapshot_to_player[:] = [
        p for p, m in zip(snapshot_to_player, keep_mask) if m
    ]
    dataset.acted_indices = dataset._compute_acted_indices()
    return stats


def build_dataset(
    *,
    player: str | list[str] = PLAYER,
    max_planets: int = 64,
    max_fleets: int = 1024,
    keep_non_acted: bool = False,
    acted_min_ratio: float = 0.30,
    seed: int = DEFAULT_SEED,
    replay_dir: Path = REPLAYS_DIR,
    planet_dir: Path = PLANET_DATASET_DIR,
    fleet_dir: Path = FLEET_DATASET_DIR,
    entity_dir: Path = ENTITY_DATASET_DIR,
    num_load_workers: int | None = None,
) -> OOPairDataset:
    """Build a pair-score dataset for one or more players.

    ``player`` accepts either a single name (back-compat with the
    original Orbital Occle call site) or a list of names. Multi-player
    builds concatenate replays, snapshots, and keys, and apply the
    ``acted_min_ratio`` floor per-player before merging.

    ``acted_min_ratio`` is only consulted when ``keep_non_acted`` is
    True; the acted-only path is already at 100% acted.
    """
    players: list[str] = [player] if isinstance(player, str) else list(player)
    if not players:
        raise ValueError("build_dataset requires at least one player")
    is_mixed = len(players) > 1
    print(f"[build] players={players!r}", flush=True)

    # Collect per-player stems + the merged replay-edges map.
    stems_by_player: dict[str, list[str]] = {}
    for p in players:
        stems_by_player[p] = sorted(player_replay_stems(replay_dir, p))
        print(
            f"[build] {len(stems_by_player[p])} replay stems for {p!r}",
            flush=True,
        )

    print("[build] walking replays to extract per-turn pair edges ...", flush=True)
    if is_mixed:
        replay_edges, stem_to_player = build_replay_edges_multi(
            players, replay_dir,
        )
    else:
        replay_edges = build_replay_edges(players[0], replay_dir)
        stem_to_player = {s: players[0] for s in replay_edges}
    n_pairs = sum(
        sum(len(v) for v in payload["edges"].values())
        for payload in replay_edges.values()
    )
    print(
        f"[build] extracted {n_pairs} learner-launch pair edges across "
        f"{len(replay_edges)} stems",
        flush=True,
    )

    # Build the merged stems list (cross-player, sorted by stem so the
    # CSV-order stays deterministic).
    all_stems = sorted(replay_edges.keys())

    # Filter CSV paths down to the merged stem set.
    def _csvs(dir_path: Path, prefix: str) -> list:
        out: list = []
        for s in all_stems:
            p = dir_path / f"{prefix}{s}.csv"
            if p.exists():
                out.append(p)
        return out
    planet_csvs = _csvs(planet_dir, "planet_")
    fleet_csvs = _csvs(fleet_dir, "fleet_")
    entity_csvs = _csvs(entity_dir, "entity_")
    print(
        f"[build] planet={len(planet_csvs)} fleet={len(fleet_csvs)} "
        f"entity={len(entity_csvs)} CSVs on disk",
        flush=True,
    )

    # Comet features (best-effort): per-stem overlay on top of global
    # lookup. Same call the entity pretrain uses; missing comet rows
    # silently fall back to scalar-only inside the base class.
    print("[build] loading comet lookup ...", flush=True)
    comet_lookup = load_comet_lookup()
    added = merge_per_stem_comet_csvs(
        comet_lookup, COMET_PER_STEM_DIR, [p.stem.removeprefix("planet_") for p in planet_csvs],
    )
    print(
        f"[build] comet lookup: {len(comet_lookup)} entries "
        f"({added} added from per-stem CSVs)",
        flush=True,
    )

    print("[build] materializing snapshots ...", flush=True)
    t0 = time.time()
    dataset = OOPairDataset(
        planet_csv_paths=planet_csvs,
        fleet_csv_paths=fleet_csvs,
        entity_csv_paths=entity_csvs,
        max_planets=max_planets,
        max_fleets=max_fleets,
        learner_slot=0,                       # CSVs are already learner-relative
        num_players=4,
        comet_lookup=comet_lookup,
        history_offsets=HISTORY_OFFSETS_T6,
        replay_edges=replay_edges,
        keep_non_acted=keep_non_acted,
        num_load_workers=num_load_workers,
    )
    build_s = time.time() - t0
    print(
        f"[build] {len(dataset)} snapshots in {build_s:.1f}s "
        f"(keep_non_acted={keep_non_acted})",
        flush=True,
    )

    # Map every snapshot to its source player via stem→player. The
    # dataset keys are ``(episode_id, turn)``; episode_id == stem in
    # practice (the replay JSON's ``id`` field matches the filename
    # stem), but we still build a lookup via the replay-edges payload
    # so we don't rely on that invariant.
    episode_to_player: dict[str, str] = {}
    for stem, payload in replay_edges.items():
        episode_to_player[payload["episode_id"]] = stem_to_player[stem]
    snapshot_to_player: list[str] = [
        episode_to_player.get(ep, players[0]) for ep, _t in dataset.keys
    ]

    # Per-player counts BEFORE any ratio control.
    pre_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"acted": 0, "non_acted": 0, "total": 0}
    )
    for i, snap in enumerate(dataset.snapshots):
        p = snapshot_to_player[i]
        pre_counts[p]["total"] += 1
        if bool(snap["pair_labels"].any()):
            pre_counts[p]["acted"] += 1
        else:
            pre_counts[p]["non_acted"] += 1
    for p, c in sorted(pre_counts.items()):
        ratio = c["acted"] / max(c["total"], 1)
        print(
            f"    [pre-ratio] player={p!r} total={c['total']} "
            f"acted={c['acted']} non_acted={c['non_acted']} "
            f"acted_ratio={ratio:.3f}",
            flush=True,
        )

    # Apply per-player acted-ratio floor (only meaningful when we are
    # actually retaining non-acted rows as training rows).
    ratio_stats: dict[str, dict[str, int]] = {}
    if keep_non_acted and acted_min_ratio > 0.0:
        print(
            f"[build] enforcing per-player acted-ratio floor "
            f"={acted_min_ratio:.2f} (seed={seed}) ...",
            flush=True,
        )
        ratio_stats = _apply_acted_ratio_floor(
            dataset,
            snapshot_to_player=snapshot_to_player,
            acted_min_ratio=acted_min_ratio,
            seed=seed,
        )

    # Per-player counts AFTER ratio control — also recomputed for the
    # acted-only path so the smoke test prints a meaningful breakdown.
    post_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"acted": 0, "non_acted": 0, "total": 0}
    )
    for i, snap in enumerate(dataset.snapshots):
        p = snapshot_to_player[i]
        post_counts[p]["total"] += 1
        if bool(snap["pair_labels"].any()):
            post_counts[p]["acted"] += 1
        else:
            post_counts[p]["non_acted"] += 1

    # Stamp metadata on the dataset so ``save_cache`` writes a
    # multi-player-aware config.
    dataset.player = players[0] if not is_mixed else _player_slug(players)
    dataset.players = players
    dataset.snapshot_to_player = snapshot_to_player
    dataset.build_meta = {
        "acted_min_ratio": float(acted_min_ratio),
        "seed": int(seed),
        "per_player_counts_pre": {p: dict(c) for p, c in pre_counts.items()},
        "per_player_counts": {p: dict(c) for p, c in post_counts.items()},
        "ratio_stats": ratio_stats,
        "snapshot_to_player": list(snapshot_to_player),
    }
    return dataset


# ---------- Smoke test ----------
def _smoke_test(dataset: OOPairDataset) -> None:
    """Print per-player counts, acted-ratio, and one acted snapshot's
    shape + ``pair_ships`` alignment with ``pair_labels``."""
    if len(dataset) == 0:
        print("[smoke] empty dataset — nothing to test", flush=True)
        return

    # Per-player snapshot breakdown (uses the build-time annotation).
    per_player = getattr(
        dataset, "build_meta", {}
    ).get("per_player_counts") or {}
    if per_player:
        print("\n[smoke] per-player snapshot counts:", flush=True)
        for p, c in sorted(per_player.items()):
            r = c["acted"] / max(c["total"], 1)
            print(
                f"    {p!r}: total={c['total']} acted={c['acted']} "
                f"non_acted={c['non_acted']} acted_ratio={r:.3f}",
                flush=True,
            )

    # Find a snapshot with non-empty pair_labels for a meaningful
    # printout (the acted-only filter usually puts these first).
    sample_idx = 0
    for i in range(min(len(dataset), 200)):
        if bool(dataset.snapshots[i]["pair_labels"].any()):
            sample_idx = i
            break

    print(
        f"\n[smoke] sample idx={sample_idx} key={dataset.keys[sample_idx]}",
        flush=True,
    )
    item = dataset[sample_idx]
    for k in sorted(item):
        if isinstance(item[k], torch.Tensor):
            print(
                f"    {k}: shape={tuple(item[k].shape)} dtype={item[k].dtype}",
                flush=True,
            )

    pl = item["pair_labels"]
    pv = item["pair_valid"]
    ps = item.get("pair_ships")
    print(
        f"\n[smoke] pair_labels.sum()={int(pl.sum().item())} "
        f"pair_valid.sum()={int(pv.sum().item())} "
        f"(P={pl.shape[0]})",
        flush=True,
    )
    if pl.sum() > 0:
        nz = torch.nonzero(pl, as_tuple=False)
        print(f"[smoke] active (src, tgt) cells: {nz.tolist()}", flush=True)
    if ps is not None:
        total_ships = int(ps.sum().item())
        max_ships = int(ps.max().item())
        # Confirm pair_ships > 0 ↔ pair_labels == True.
        labels_match_ships = bool(((ps > 0) == pl).all().item())
        print(
            f"[smoke] pair_ships: total={total_ships} max_per_cell={max_ships} "
            f"dtype={ps.dtype}",
            flush=True,
        )
        print(
            f"[smoke] pair_ships > 0 aligns with pair_labels: "
            f"{labels_match_ships}",
            flush=True,
        )
        if not labels_match_ships:
            # Locate disagreements for debugging.
            mismatch = (ps > 0) != pl
            count = int(mismatch.sum().item())
            print(
                f"[smoke] WARN {count} cells disagree between "
                f"pair_ships and pair_labels",
                flush=True,
            )

    # Corpus-level pair-label stats.
    total_snapshots = len(dataset)
    total_positives = sum(int(s["pair_labels"].sum().item()) for s in dataset.snapshots)
    snapshots_with_any = sum(
        1 for s in dataset.snapshots if bool(s["pair_labels"].any())
    )
    multi_launch = sum(
        1 for s in dataset.snapshots if int(s["pair_labels"].sum().item()) > 1
    )
    overall_acted_ratio = snapshots_with_any / max(total_snapshots, 1)
    print(
        f"\n[smoke] corpus: snapshots={total_snapshots} "
        f"total_positives={total_positives} "
        f"acted_snapshots={snapshots_with_any} "
        f"acted_ratio={overall_acted_ratio:.3f} "
        f"multi_launch_snapshots={multi_launch}",
        flush=True,
    )


# ---------- CLI ----------
def _parse_player_arg(raw: str) -> list[str]:
    """Comma-separated list. A single name is returned as a list of
    one — ``build_dataset`` handles both single and mixed."""
    names = [p.strip() for p in raw.split(",") if p.strip()]
    if not names:
        raise argparse.ArgumentTypeError(
            f"--player must list at least one player; got {raw!r}"
        )
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a T=6 pair-score dataset. Accepts a single player "
            "or a comma-separated list to mix multiple players' "
            "replays into one cache."
        ),
    )
    parser.add_argument(
        "--player", default=PLAYER,
        help="Player name, or comma-separated list of names "
             "(e.g. 'bowwowforeach,Ebi').",
    )
    parser.add_argument("--max-planets", type=int, default=64)
    parser.add_argument("--max-fleets", type=int, default=1024)
    parser.add_argument(
        "--keep-non-acted", action="store_true",
        help="Use non-acted snapshots as supervised all-negative rows. "
             "Without this flag, all snapshots are kept but only "
             "acted_indices are trained on (non-acted rows stay as "
             "T-history context).",
    )
    parser.add_argument(
        "--acted-min-ratio", type=float, default=0.30,
        help="When --keep-non-acted is set, randomly subsample "
             "non-acted snapshots per-player so the acted-ratio is at "
             "least this. Set to 0 to disable. (default: 0.30)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"RNG seed for the non-acted subsample "
             f"(default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--num-load-workers", type=int, default=None,
        help="If >1, parse CSVs in a ProcessPool (default: serial).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Override the auto-derived cache path.",
    )
    args = parser.parse_args()

    players = _parse_player_arg(args.player)
    # Back-compat: a single name resolves to the legacy single-player
    # ``build_dataset(player=str)`` signature transparently.
    player_arg: str | list[str] = players[0] if len(players) == 1 else players

    dataset = build_dataset(
        player=player_arg,
        max_planets=args.max_planets,
        max_fleets=args.max_fleets,
        keep_non_acted=args.keep_non_acted,
        acted_min_ratio=args.acted_min_ratio,
        seed=args.seed,
        num_load_workers=args.num_load_workers,
    )

    out = args.out or cache_path(
        player=player_arg,
        max_planets=args.max_planets,
        max_fleets=args.max_fleets,
        keep_non_acted=args.keep_non_acted,
    )
    print(f"\n[save] writing cache → {out}", flush=True)
    t0 = time.time()
    save_cache(dataset, out)
    size_mb = out.stat().st_size / 1e6
    print(f"[save] wrote {size_mb:.1f} MB in {time.time() - t0:.1f}s", flush=True)

    _smoke_test(dataset)


if __name__ == "__main__":
    main()
