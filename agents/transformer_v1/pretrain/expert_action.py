"""Action-snapshot dataset glue.

Decoder code (ActionDecoder / ContextualActionDecoder / PairActionDecoder /
GlobalStateDecoder, plus the BC training & PPO inference utilities) was
removed during the encoder-freeze + single-pair-head smoke test redesign.
This module now exists only to expose :class:`ActionSnapshotDataset` and a
couple of constants reused by the new pair-score trainer (see
``agents/transformer_v1/pretrain/pair_score.py``).

If you need the previous decoder/training/inference code, recover it from
git history (commit ``f4056d3`` and earlier).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..featurizer import ACTION_IGNORE_INDEX
from .cross_entity import CrossEntitySnapshotDataset

# Lower-bound logit on frac samples (logit(0.05) ≈ -2.94). Kept here as a
# stable reference value for future frac-head bring-up; the current pair-
# score head does not consume it.
FRAC_Z_MIN: float = -2.94


# ---------- Dataset ----------
class ActionSnapshotDataset(CrossEntitySnapshotDataset):
    """Same tensors as :class:`CrossEntitySnapshotDataset` (planet+fleet+
    entity+cross), plus per-snapshot expert-action labels keyed by
    ``(episode_id, turn)``.

    Labels:
      * ``source_planet_idx`` (Long) — int 0..P-1 or
        ``ACTION_IGNORE_INDEX`` (=-100) when the expert didn't act
      * ``target_planet_idx`` (Long) — same convention
      * ``expert_acted`` (Float) — 0/1
      * ``frac_label`` (Float) — ``ships_sent / source_ships`` (BEFORE
        launch), in (0, 1]; NaN when no acted / source unknown
      * ``src_valid`` (Bool, ``(max_planets,)``) — owned & launchable
      * ``tgt_valid`` (Bool, ``(max_planets,)``) — planet exists

    Mask side cache (``data/datasets/action/_masks/<stem>.npz``) is
    loaded alongside action CSVs in ``__init__`` and sliced per
    snapshot. A missing mask file falls back to all-False (defensive —
    the loss code masks loss contributions accordingly).
    """

    def __init__(
        self,
        planet_csv_paths: list[Path],
        fleet_csv_paths: list[Path],
        entity_csv_paths: list[Path],
        cross_entity_csv_paths: list[Path],
        action_csv_paths: list[Path],
        *,
        max_planets: int = 64,
        max_fleets: int = 256,
        learner_slot: int = 0,
        num_players: int = 4,
        n_history: int = 3,
        num_load_workers: int | None = None,
    ):
        # Pre-load action rows + mask side cache BEFORE super().__init__,
        # so the inherited ``_build_snapshot`` calls can pick up our
        # extension. Action CSVs are tiny (~10 KB each, ~10 MB total)
        # so the all-at-once load doesn't pressure memory; only
        # cross_entity needs streaming.
        from ..featurizer.action_featurizer import ACTION_MASK_CACHE_DIR

        self._action_by_key: dict[tuple[str, int], dict[str, str]] = {}
        # episode_id → (turn → row index) so slice(turn) is O(1).
        self._mask_index: dict[str, dict[int, int]] = {}
        # episode_id → (src_valid, tgt_valid) np.ndarray each (T, P_in_file).
        self._mask_arrays: dict[str, tuple[Any, Any]] = {}
        for p in action_csv_paths:
            with p.open() as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                continue
            ep_id = rows[0]["episode_id"]
            for row in rows:
                key = (row["episode_id"], int(row["turn"]))
                self._action_by_key[key] = row

            # Mask file path: action_<stem>.csv → _masks/<stem>.npz.
            stem = p.stem.removeprefix("action_")
            mask_path = ACTION_MASK_CACHE_DIR / f"{stem}.npz"
            if not mask_path.exists():
                continue
            try:
                arr = np.load(mask_path)
                turns = arr["turns"]
                self._mask_index[ep_id] = {int(t): i for i, t in enumerate(turns)}
                self._mask_arrays[ep_id] = (arr["src_valid"], arr["tgt_valid"])
            except Exception as e:  # noqa: BLE001
                # Defensive — bad mask file shouldn't kill the whole load.
                print(f"[ActionSnapshotDataset] warn: failed to load {mask_path}: {e}",
                      flush=True)
        super().__init__(
            planet_csv_paths=planet_csv_paths,
            fleet_csv_paths=fleet_csv_paths,
            entity_csv_paths=entity_csv_paths,
            cross_entity_csv_paths=cross_entity_csv_paths,
            max_planets=max_planets,
            max_fleets=max_fleets,
            learner_slot=learner_slot,
            num_players=num_players,
            n_history=n_history,
            num_load_workers=num_load_workers,
        )
        self._action_by_key.clear()
        self._mask_index.clear()
        self._mask_arrays.clear()

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
        action_row = self._action_by_key.get(key)
        if action_row is not None:
            source_idx = int(action_row["source_planet_idx"])
            target_idx = int(action_row["target_planet_idx"])
            expert_acted = float(action_row["expert_acted"])
            raw_frac = action_row.get("frac_label", "")
            try:
                frac_label = float(raw_frac) if raw_frac else float("nan")
            except (TypeError, ValueError):
                frac_label = float("nan")
        else:
            source_idx = ACTION_IGNORE_INDEX
            target_idx = ACTION_IGNORE_INDEX
            expert_acted = 0.0
            frac_label = float("nan")
        snapshot["source_planet_idx"] = torch.tensor(source_idx, dtype=torch.long)
        snapshot["target_planet_idx"] = torch.tensor(target_idx, dtype=torch.long)
        snapshot["expert_acted"] = torch.tensor(expert_acted, dtype=torch.float32)
        snapshot["frac_label"] = torch.tensor(frac_label, dtype=torch.float32)

        # Per-turn validity masks. Shape (max_planets,) bool. The npz
        # may have fewer entities than self.max_planets if the featurizer
        # ran with a smaller cap — pad with False.
        ep_id, turn = key
        src_valid = torch.zeros(self.max_planets, dtype=torch.bool)
        tgt_valid = torch.zeros(self.max_planets, dtype=torch.bool)
        idx_map = self._mask_index.get(ep_id)
        if idx_map is not None and turn in idx_map:
            row_i = idx_map[turn]
            src_arr, tgt_arr = self._mask_arrays[ep_id]
            P_file = src_arr.shape[1]
            P_use = min(P_file, self.max_planets)
            src_valid[:P_use] = torch.from_numpy(src_arr[row_i, :P_use])
            tgt_valid[:P_use] = torch.from_numpy(tgt_arr[row_i, :P_use])
        # Force-include the expert's chosen source whenever it acted.
        # The featurizer's "launchable" rule (compute_surplus >= min_launch)
        # is a heuristic and disagrees with the expert ~25% of the time —
        # mostly when the expert launches with a smaller surplus than our
        # min_launch threshold. Excluding the expert's own source here
        # would push CE→inf and discard the supervision signal entirely.
        if expert_acted > 0.5 and 0 <= source_idx < self.max_planets:
            src_valid[source_idx] = True
        snapshot["src_valid"] = src_valid
        snapshot["tgt_valid"] = tgt_valid
        return snapshot

    @classmethod
    def from_cache(cls, cache_path: str | Path) -> "ActionSnapshotDataset":
        """Construct a dataset by ``torch.load``-ing a previously-saved
        snapshot bundle. Skips ``__init__``'s CSV parsing entirely.
        """
        ckpt = torch.load(cache_path, map_location="cpu", weights_only=False)
        instance = cls.__new__(cls)
        instance.max_planets = ckpt["max_planets"]
        instance.max_fleets = ckpt["max_fleets"]
        instance.learner_slot = ckpt["learner_slot"]
        instance.num_players = ckpt["num_players"]
        instance.n_history = ckpt["n_history"]
        instance.snapshots = ckpt["snapshots"]
        instance.keys = ckpt["keys"]
        instance._key_to_idx = {k: i for i, k in enumerate(instance.keys)}
        instance._action_by_key = {}
        instance._cross_entity_paths = {}
        instance._cross_entity_by_key = {}
        return instance

    def save_cache(self, cache_path: str | Path) -> None:
        """Serialize this dataset's materialized snapshots to disk.

        Reload via :meth:`from_cache` to skip CSV parsing on subsequent
        runs. The cache is invalidated by changes to the source CSVs —
        the caller is responsible for picking a path that encodes the
        relevant config (max_planets, n_history, episode set, etc.).
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "snapshots": self.snapshots,
                "keys": self.keys,
                "max_planets": self.max_planets,
                "max_fleets": self.max_fleets,
                "learner_slot": self.learner_slot,
                "num_players": self.num_players,
                "n_history": self.n_history,
                "format_version": 1,
            },
            cache_path,
        )
