"""Expert-action featurizer for imitation pretraining.

One row per ``(episode_id, turn)``. Unlike the per-(planet, player)
entity CSV, this dataset is *snapshot-level*: it answers "what did the
expert do on this turn" with two categorical labels — which planet the
expert launched **from** and which planet the expert aimed **at**.

The labels are derived from the same t→t+1 transition scan used by
:func:`entity_featurizer._action_planet_flags`, but here we pick the
**first new expert fleet** by ``F_ID`` (Kaggle env's ``next_fleet_id``
is monotonic, so smallest-id = first-created in this turn). Multi-launch
turns therefore collapse to one row per turn — keeping the categorical
softmax targets unambiguous.

Schema:

    episode_id, turn, num_planets,
    source_planet_idx,    # int 0..P-1; -100 if no expert action
    target_planet_idx,    # int 0..P-1; -100 if no expert action
    expert_acted,         # 0/1
    winner_seat           # broadcast from _compute_episode_summary

The ``-100`` sentinel pairs with ``nn.CrossEntropyLoss(ignore_index=-100)``
so masking is automatic — no separate mask tensor needed.

``planet_idx`` is the position of the planet in ``obs.planets`` (capped
at ``max_entities``), matching what ``CrossEntityAttention`` sees. If
the expert acted on a planet outside the first ``max_entities``, both
indices fall back to ``-100`` for that turn (effectively skip).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...physics_utils import (
    F_FROM,
    F_ID,
    F_OWNER,
    P_ID,
)
from ..paths import ACTION_DATASET_DIR

# Side cache for per-turn launchable / target-valid masks. Sits next to
# the action CSVs so loaders can match a CSV by stem to its
# ``_masks/<stem>.npz``. Re-imported by
# :class:`agents.transformer_v1.pretrain.expert_action.ActionSnapshotDataset`.
ACTION_MASK_CACHE_DIR: Path = ACTION_DATASET_DIR / "_masks"

from .entity_featurizer import (
    _compute_episode_summary,
    _infer_learner_slot_from_episode_path,
    _newly_launched_fleet_ids,
)
from .fleet_featurizer import _resolve_target


ACTION_LABEL_COLS: tuple[str, ...] = (
    "source_planet_idx",
    "target_planet_idx",
    "expert_acted",
    "winner_seat",
)

# nn.CrossEntropyLoss ignore_index sentinel.
ACTION_IGNORE_INDEX: int = -100


def _first_expert_action(
    prev_obs: Any,
    next_obs: Any,
    *,
    learner_slot: int,
) -> tuple[int | None, int | None]:
    """Return ``(source_pid, target_pid)`` of the *first* new expert fleet.

    "First" = smallest ``F_ID`` (the env's ``next_fleet_id`` is monotonic,
    so this is deterministic and matches the order the expert agent
    actually queued the fleets in). Returns:

      * ``(None, None)`` — no expert fleet launched this turn (or no
        next observation to inspect).
      * ``(src_pid, None)`` — expert launched, but
        :func:`_resolve_target` couldn't determine the first-hit planet
        (rare; off-board trajectory). Caller should still mark
        ``expert_acted=1`` and supervise the source head.
      * ``(src_pid, tgt_pid)`` — expert launched and target is known.

    Source is taken straight from ``F_FROM`` and is always present when
    a new expert fleet exists, so we don't drop the source supervision
    just because the target isn't resolvable.
    """
    new_ids = _newly_launched_fleet_ids(prev_obs, next_obs)
    if not new_ids or not next_obs:
        return None, None
    next_planets = next_obs.get("planets") or []
    candidates: list[Any] = []
    for f in next_obs.get("fleets") or []:
        if int(f[F_ID]) not in new_ids:
            continue
        # Expert-only filter: imitation labels need to mean "the
        # learner launched this", not "anyone launched this".
        if int(f[F_OWNER]) != learner_slot:
            continue
        candidates.append(f)
    if not candidates:
        return None, None
    candidates.sort(key=lambda f: int(f[F_ID]))
    first = candidates[0]
    src_pid = int(first[F_FROM])
    tgt_pid, _eta, _planet = _resolve_target(first, next_planets)
    return src_pid, (int(tgt_pid) if tgt_pid is not None else None)


def save_episode_action_csv(
    episode_path: str | Path,
    out_path: str | Path | None = None,
    *,
    learner_slot: int | None = None,
    num_players: int | None = None,
    max_entities: int = 64,
) -> Path:
    """Featurize every turn in one episode into expert-action labels.

    Joins one-to-one with the cross-entity CSV by ``(episode_id, turn)``.

    Output defaults to ``data/datasets/action/``.
    """
    import csv
    import gzip
    import json

    episode_path = Path(episode_path)
    if out_path is None:
        out_path = (
            ACTION_DATASET_DIR
            / f"action_{episode_path.stem.split('.')[0]}.csv"
        )
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(episode_path, "rb") as fh:
        replay = json.load(fh)
    steps = replay.get("steps") or []
    if num_players is None:
        num_players = len(steps[0]) if steps else 4
    if learner_slot is None:
        learner_slot = _infer_learner_slot_from_episode_path(episode_path)
    episode_id = replay.get("id") or episode_path.stem

    episode_summary = _compute_episode_summary(
        steps, learner_slot=learner_slot, num_players=num_players,
    )
    winner_seat = int(episode_summary["winner_seat"])

    header = [
        "episode_id",
        "turn",
        "num_planets",
        *ACTION_LABEL_COLS,
    ]

    rows_written = 0
    with out_path.open("w", newline="") as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(header)
        for t, step in enumerate(steps):
            if not step:
                continue
            seat = step[learner_slot] if len(step) > learner_slot else step[0]
            obs = seat.get("observation") if isinstance(seat, dict) else None
            if not obs:
                continue
            raw_planets = obs.get("planets") or []
            visible_planets = raw_planets[:max_entities]
            num_planets = len(visible_planets)

            # Build planet_id → planet_idx map matching what
            # CrossEntityAttention sees (= position in obs.planets,
            # capped at max_entities).
            pid_to_idx = {
                int(p[P_ID]): i for i, p in enumerate(visible_planets)
            }

            # Pull next_obs from the same seat as ``obs`` so any
            # seat-locality in the env doesn't leak across the
            # transition.
            next_obs = None
            if t + 1 < len(steps) and steps[t + 1]:
                next_step = steps[t + 1]
                next_seat = (
                    next_step[learner_slot]
                    if len(next_step) > learner_slot
                    else next_step[0]
                )
                next_obs = (
                    next_seat.get("observation")
                    if isinstance(next_seat, dict) else None
                )

            src_pid, tgt_pid = _first_expert_action(
                obs, next_obs, learner_slot=learner_slot,
            )
            if src_pid is None:
                source_idx = ACTION_IGNORE_INDEX
                target_idx = ACTION_IGNORE_INDEX
                expert_acted = 0
            else:
                # Expert acted — supervise expert_acted=1 regardless of
                # whether the categorical heads can be supervised. Per-
                # head -100 sentinels mask source/target independently
                # so a missing target doesn't drop a valid source.
                expert_acted = 1
                src_idx = pid_to_idx.get(src_pid)
                source_idx = (
                    int(src_idx) if src_idx is not None
                    else ACTION_IGNORE_INDEX
                )
                tgt_idx = pid_to_idx.get(tgt_pid) if tgt_pid is not None else None
                target_idx = (
                    int(tgt_idx) if tgt_idx is not None
                    else ACTION_IGNORE_INDEX
                )

            writer.writerow([
                episode_id,
                t,
                num_planets,
                source_idx,
                target_idx,
                expert_acted,
                winner_seat,
            ])
            rows_written += 1

    print(f"[action_featurizer] wrote {rows_written} rows → {out_path}")
    return out_path
