"""Expert-action imitation pretraining.

Resumes from a frozen ``cross_entity_best.pt`` and trains a small
:class:`ActionDecoder` on top of the cross-entity attention. The
decoder predicts:

  * ``source_planet_idx`` — categorical (P-way softmax) over per-planet
    contextual tokens; "which planet did the expert launch FROM"
  * ``target_planet_idx`` — categorical (P-way softmax); "which planet
    was the expert's first-launch target"
  * ``expert_acted`` — binary, "did the expert launch anything this turn"

Source / target labels use ``ignore_index = -100`` for turns where no
expert action happened, so the categorical losses are masked
automatically without a separate validity tensor.

Two modes:

  * ``--train-mode frozen`` — train only the new decoder; everything
    else is held in ``no_grad``. Use this to bring up the head from
    scratch on top of an existing cross-entity ckpt. (Stage 0.)
  * ``--train-mode gradual-unfreeze`` — resume from an action ckpt
    (``--resume-action-pt``) and progressively thaw cross-attention →
    entity encoder → planet/fleet top-halves → full stack.

Run from the repo root:

    # Stage 0: frozen decoder bring-up (5 epochs)
    python -m agents.transformer_v1.pretrain.expert_action \\
        --resume-cross-pt data/runs/cross_entity/<run>/cross_entity_best.pt \\
        --train-mode frozen --epochs 5

    # Stages 1-4: gradual unfreeze (5 epochs each)
    python -m agents.transformer_v1.pretrain.expert_action \\
        --resume-action-pt data/runs/action/<run>/action_best.pt \\
        --train-mode gradual-unfreeze --stage-epochs 5,5,5,5

Outputs (under ``data/runs/action/<timestamp>/``):
  * ``action_best.pt`` / ``action_last.pt``
  * ``log.json`` — per-epoch train + val per-head metrics
  * ``test_summary.json`` — per-head test metrics from the best ckpt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal
from torch.utils.data import DataLoader

from ..aggregator import CrossEntityAttention
from ..encoder.entity_encoder import PlanetEntityEncoder
from ..encoder.fleet_encoder import FleetEncoder
from ..encoder.planet_encoder import PlanetEncoder
from ..featurizer import (
    ACTION_IGNORE_INDEX,
    CROSS_ENTITY_VALUE_HORIZONS,
    ENTITY_N_OWNER_CLASSES,
)
from ..featurizer.entity_featurizer import GAME_PHASE_N_CLASSES
from ..paths import (
    ACTION_DATASET_DIR,
    ACTION_RUNS_DIR,
    CROSS_ENTITY_DATASET_DIR,
    CROSS_ENTITY_RUNS_DIR,
    ENTITY_DATASET_DIR,
    ENTITY_RUNS_DIR,
    FLEET_DATASET_DIR,
    FLEET_RUNS_DIR,
    PLANET_DATASET_DIR,
    PLANET_RUNS_DIR,
)
from .cross_entity import (
    CrossEntitySnapshotDataset,
    _entity_tokens_per_step,
    _freeze_all,
    _latest_run_dir,
    _load_frozen_encoders,
    _resolve_attr,
)

# ---------------------------------------------------------------------------
# Truncation constant for lower-truncated Normal frac sampling
# ---------------------------------------------------------------------------
# frac_z is sampled in logit-space.  The env clamps small-frac actions to
# min_launch ships, which effectively creates a lower bound on the logit.
# We use a conservative constant floor:  logit(0.05) ≈ -2.94, corresponding
# to ~5% of source ships.  This avoids per-state plumbing (option b).
# Single source of truth: imported by runner.py and ppo.py.
FRAC_Z_MIN: float = -2.94  # logit(0.05) — frac samples truncated to >= 5%


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
        # Pre-load action rows BEFORE super().__init__, so the inherited
        # ``_build_snapshot`` calls can pick up our extension. Action
        # CSVs are tiny (~10 KB each, ~10 MB total) so the all-at-once
        # load doesn't pressure memory; only cross_entity needs streaming.
        self._action_by_key: dict[tuple[str, int], dict[str, str]] = {}
        for p in action_csv_paths:
            with p.open() as fh:
                for row in csv.DictReader(fh):
                    key = (row["episode_id"], int(row["turn"]))
                    self._action_by_key[key] = row
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
        else:
            source_idx = ACTION_IGNORE_INDEX
            target_idx = ACTION_IGNORE_INDEX
            expert_acted = 0.0
        snapshot["source_planet_idx"] = torch.tensor(source_idx, dtype=torch.long)
        snapshot["target_planet_idx"] = torch.tensor(target_idx, dtype=torch.long)
        snapshot["expert_acted"] = torch.tensor(expert_acted, dtype=torch.float32)
        return snapshot

    # ---- Disk cache: serialize the materialized snapshots once so
    # subsequent runs skip the CSV parse + tensor build entirely. On a
    # 51 GB-RAM Colab the cache fits comfortably in memory; on a 12 GB
    # T4 it still saves the parse step (the snapshots themselves are
    # ~5 GB resident regardless of cache).

    @classmethod
    def from_cache(cls, cache_path: str | Path) -> "ActionSnapshotDataset":
        """Construct a dataset by ``torch.load``-ing a previously-saved
        snapshot bundle. Skips ``__init__``'s CSV parsing entirely.
        """
        ckpt = torch.load(cache_path, map_location="cpu", weights_only=False)
        # Bypass __init__ — populate fields directly.
        instance = cls.__new__(cls)
        instance.max_planets = ckpt["max_planets"]
        instance.max_fleets = ckpt["max_fleets"]
        instance.learner_slot = ckpt["learner_slot"]
        instance.num_players = ckpt["num_players"]
        instance.n_history = ckpt["n_history"]
        instance.snapshots = ckpt["snapshots"]
        instance.keys = ckpt["keys"]
        instance._key_to_idx = {k: i for i, k in enumerate(instance.keys)}
        # These attrs are only used during construction; populate empty
        # to keep instance-state schema stable.
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


# ---------- Model ----------
class ActionDecoder(nn.Module):
    """Three small linear heads on top of CrossEntityAttention outputs.

    Per-planet heads (operate on per-planet contextual tokens at the
    current step, shape ``(B, P, d_model)``):
      * ``head_source`` → 1 logit per planet (B, P) for source softmax
      * ``head_target`` → 1 logit per planet (B, P) for target softmax

    CLS / global head (operates on the [CLS] read-out, shape
    ``(B, d_model)``):
      * ``head_acted`` → scalar (B,) for binary expert_acted prediction

    Padded planet slots are masked to ``-inf`` so softmax never picks
    them. The loss caller relies on ``ignore_index=-100`` to skip
    no-action rows.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.head_source = nn.Linear(d_model, 1)
        self.head_target = nn.Linear(d_model, 1)
        self.head_acted = nn.Linear(d_model, 1)
        # PPO heads
        self.head_value  = nn.Linear(2 * d_model, 1)   # reads [CLS, mean_pool(planets)]
        self.head_frac   = nn.Linear(d_model, 1)       # per-planet sigmoid → ship frac
        # Multi-horizon auxiliary value heads — predict n-step returns at
        # k ∈ CROSS_ENTITY_VALUE_HORIZONS. Trained against retrospective
        # targets in PPO; consumed only for auxiliary CLS regularisation.
        self.head_value_h = nn.ModuleDict({
            f"h{k}": nn.Linear(2 * d_model, 1)
            for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        for h in (self.head_value, self.head_frac, *self.head_value_h.values()):
            nn.init.zeros_(h.bias)
            nn.init.normal_(h.weight, std=1e-3)
        self.frac_log_std = nn.Parameter(torch.tensor(math.log(0.2)))
        self.value_bn = nn.LayerNorm(1)

    def forward(
        self,
        ctx_now: torch.Tensor,    # (B, P, d_model)
        mask_now: torch.Tensor,    # (B, P) bool — True = real planet
        glob: torch.Tensor,        # (B, d_model)
    ) -> dict[str, torch.Tensor]:
        src = self.head_source(ctx_now).squeeze(-1)            # (B, P)
        tgt = self.head_target(ctx_now).squeeze(-1)            # (B, P)
        neg_inf = torch.finfo(src.dtype).min
        # Guard: if a row has *all* planets masked (no real planets), fall back to
        # logit 0 at index 0 so downstream Categorical doesn't see all-(-inf) → NaN.
        any_valid = mask_now.any(dim=-1, keepdim=True)               # (B, 1)
        src = torch.where(any_valid, src.masked_fill(~mask_now, neg_inf),
                          src.masked_fill(~mask_now, 0.0))
        tgt = torch.where(any_valid, tgt.masked_fill(~mask_now, neg_inf),
                          tgt.masked_fill(~mask_now, 0.0))
        # For the batch-pool: if all-masked, falling back to index 0's data is
        # harmless — the mask validity is checked upstream anyway.
        mean_pool = (
            ctx_now.masked_fill(~mask_now.unsqueeze(-1), 0).sum(dim=1)
            / mask_now.sum(dim=1, keepdim=True).clamp(min=1)
        )
        value_input = torch.cat([glob, mean_pool], dim=-1)
        value = self.head_value(value_input).squeeze(-1)
        frac_logits = self.head_frac(ctx_now).squeeze(-1)      # (B, P)
        out = {
            "source_planet_logits": src,
            "target_planet_logits": tgt,
            "expert_acted_logit": self.head_acted(glob).squeeze(-1),
            "value": value,
            "frac_logits": frac_logits,
        }
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            out[f"value_h{k}"] = self.head_value_h[f"h{k}"](value_input).squeeze(-1)
        return out


class ContextualActionDecoder(nn.Module):
    """Per-planet action heads conditioned on the CLS global token.

    Drop-in replacement for :class:`ActionDecoder`. Each per-planet head
    reads ``[glob || ctx_planet_i]`` (shape ``2 * d_model``) instead of
    just ``ctx_planet_i``, so source / target / frac decisions can use
    the game-state summary the CLS encodes (winner, is_ahead, game_phase,
    ship balance, etc.).

    Output dict keys are identical to :class:`ActionDecoder` so
    :func:`compute_loss`, :func:`compute_action_log_prob`,
    :func:`evaluate`, ``runner.py`` and ``ppo.py`` all work unchanged.
    """

    def __init__(self, d_model: int = 64, hidden: int | None = None):
        super().__init__()
        in_dim = 2 * d_model
        h = hidden if hidden is not None else d_model

        def _mk_head() -> nn.Sequential:
            # 2-layer MLP — non-linear mix of glob (game-state) and
            # ctx (planet-state) before the per-planet 1-D logit.
            return nn.Sequential(
                nn.Linear(in_dim, h),
                nn.GELU(),
                nn.Linear(h, 1),
            )

        self.head_source = _mk_head()
        self.head_target = _mk_head()
        self.head_frac   = _mk_head()
        # CLS-only heads — same as ActionDecoder so PPO machinery is identical.
        self.head_acted = nn.Linear(d_model, 1)
        self.head_value = nn.Linear(2 * d_model, 1)
        # Multi-horizon auxiliary value heads. See ActionDecoder for the
        # rationale; output dict keys must stay parity with ActionDecoder
        # so PPO/runner code does not have to branch on decoder class.
        self.head_value_h = nn.ModuleDict({
            f"h{k}": nn.Linear(2 * d_model, 1)
            for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        # Match ActionDecoder's small-init for value/frac so PPO bring-up
        # doesn't diverge.
        nn.init.zeros_(self.head_value.bias)
        nn.init.normal_(self.head_value.weight, std=1e-3)
        nn.init.zeros_(self.head_frac[-1].bias)
        nn.init.normal_(self.head_frac[-1].weight, std=1e-3)
        for h in self.head_value_h.values():
            nn.init.zeros_(h.bias)
            nn.init.normal_(h.weight, std=1e-3)
        self.frac_log_std = nn.Parameter(torch.tensor(math.log(0.2)))
        self.value_bn = nn.LayerNorm(1)

    def forward(
        self,
        ctx_now: torch.Tensor,    # (B, P, d_model)
        mask_now: torch.Tensor,    # (B, P) bool — True = real planet
        glob: torch.Tensor,        # (B, d_model)
    ) -> dict[str, torch.Tensor]:
        B, P, d = ctx_now.shape
        # Broadcast glob across the planet dim, then concat per-planet.
        glob_b = glob.unsqueeze(1).expand(B, P, d)             # (B, P, d)
        joint  = torch.cat([glob_b, ctx_now], dim=-1)          # (B, P, 2d)

        src = self.head_source(joint).squeeze(-1)              # (B, P)
        tgt = self.head_target(joint).squeeze(-1)              # (B, P)
        frac_logits = self.head_frac(joint).squeeze(-1)        # (B, P)

        # Same -inf masking + all-masked guard as ActionDecoder so
        # downstream Categorical never sees all-(-inf) → NaN.
        neg_inf = torch.finfo(src.dtype).min
        any_valid = mask_now.any(dim=-1, keepdim=True)
        src = torch.where(any_valid, src.masked_fill(~mask_now, neg_inf),
                          src.masked_fill(~mask_now, 0.0))
        tgt = torch.where(any_valid, tgt.masked_fill(~mask_now, neg_inf),
                          tgt.masked_fill(~mask_now, 0.0))

        # Same value head as ActionDecoder: [glob, mean_pool(ctx)].
        mean_pool = (
            ctx_now.masked_fill(~mask_now.unsqueeze(-1), 0).sum(dim=1)
            / mask_now.sum(dim=1, keepdim=True).clamp(min=1)
        )
        value_input = torch.cat([glob, mean_pool], dim=-1)
        value = self.head_value(value_input).squeeze(-1)

        out = {
            "source_planet_logits": src,
            "target_planet_logits": tgt,
            "expert_acted_logit": self.head_acted(glob).squeeze(-1),
            "value": value,
            "frac_logits": frac_logits,
        }
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            out[f"value_h{k}"] = self.head_value_h[f"h{k}"](value_input).squeeze(-1)
        return out


def _build_action_decoder(d_model: int, *, contextual: bool) -> nn.Module:
    """Factory returning the requested action decoder flavor."""
    if contextual:
        return ContextualActionDecoder(d_model=d_model)
    return ActionDecoder(d_model=d_model)


class GlobalStateDecoder(nn.Module):
    """Auxiliary heads on the CLS global_token that maintain game-state
    supervision during action fine-tuning.

    All heads are direct ``nn.Linear`` (matching the existing per-head
    pattern). Outputs are prefixed ``"gd_"`` so they cannot collide with
    :class:`ActionDecoder` keys.

    Labels used (all already present in
    :class:`ActionSnapshotDataset` via the cross-entity CSV):

    * ``winner_seat``              — 4-way CE (who wins)
    * ``score_advantage_at_end_log`` — Huber regression
    * ``turns_until_episode_end``  — MSE regression (sigmoid-bounded)
    * ``is_ahead_t_plus_{k}``      — binary BCE per horizon, validity-masked
    * ``leader_seat_t_plus_{k}``   — 4-way CE per horizon, validity-masked
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        # Player-relative heads (current set).
        self.head_winner     = nn.Linear(d_model, ENTITY_N_OWNER_CLASSES)
        self.head_score_end  = nn.Linear(d_model, 1)
        self.head_turns_left = nn.Linear(d_model, 1)
        self.head_is_ahead_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, 1)
            for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        self.head_leader_k = nn.ModuleDict({
            f"k{k}": nn.Linear(d_model, ENTITY_N_OWNER_CLASSES)
            for k in CROSS_ENTITY_VALUE_HORIZONS
        })
        # Neutral (perspective-independent) heads — regularize the CLS
        # toward a true game-state representation, not just "am I winning".
        self.head_total_ships     = nn.Linear(d_model, 1)   # log(total ships)
        self.head_ship_entropy    = nn.Linear(d_model, 1)   # H(ship dist) in nats
        self.head_n_neutral       = nn.Linear(d_model, 1)   # neutral planet count
        self.head_game_phase      = nn.Linear(d_model, GAME_PHASE_N_CLASSES)

    def forward(self, glob: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            "gd_winner_seat":  self.head_winner(glob),
            "gd_score_end":    self.head_score_end(glob).squeeze(-1),
            "gd_turns_left":   torch.sigmoid(self.head_turns_left(glob).squeeze(-1)),
            "gd_total_ships_log":      self.head_total_ships(glob).squeeze(-1),
            "gd_ship_entropy":         self.head_ship_entropy(glob).squeeze(-1),
            "gd_n_neutral_planets":    self.head_n_neutral(glob).squeeze(-1),
            "gd_game_phase":           self.head_game_phase(glob),
        }
        for k in CROSS_ENTITY_VALUE_HORIZONS:
            out[f"gd_is_ahead_{k}"] = self.head_is_ahead_k[f"k{k}"](glob).squeeze(-1)
            out[f"gd_leader_{k}"]   = self.head_leader_k[f"k{k}"](glob)
        return out


def compute_action_log_prob(
    acted_logit: torch.Tensor,
    src_logits: torch.Tensor,
    tgt_logits: torch.Tensor,
    frac_logits: torch.Tensor,
    frac_log_std: torch.Tensor,
    acted: int,
    src_idx: int,
    tgt_idx: int,
    frac_z: float,
) -> torch.Tensor:
    """Joint log-prob of the factored action distribution.

    `frac_z` is the pre-sigmoid Normal sample stored in the rollout
    (see :class:`TransformerAgent._predict`). Used directly as the
    Normal observation; no `logit()` reconstruction needed.
    """
    acted_dist = Bernoulli(logits=acted_logit.unsqueeze(0))
    src_dist = Categorical(logits=src_logits.unsqueeze(0))
    tgt_dist = Categorical(logits=tgt_logits.unsqueeze(0))
    frac_std = torch.clamp(frac_log_std.exp(), min=0.30, max=1.0).to(
        device=frac_logits.device,
        dtype=frac_logits.dtype,
    )
    frac_dist = Normal(loc=frac_logits[src_idx].unsqueeze(0).unsqueeze(0), scale=frac_std)

    log_prob = acted_dist.log_prob(
        torch.tensor([float(acted)], device=acted_logit.device, dtype=acted_logit.dtype)
    )
    if acted:
        log_prob = log_prob + src_dist.log_prob(
            torch.tensor([src_idx], device=src_logits.device, dtype=torch.long)
        )
        log_prob = log_prob + tgt_dist.log_prob(
            torch.tensor([tgt_idx], device=tgt_logits.device, dtype=torch.long)
        )
        # `frac_z` is the pre-sigmoid Normal sample. Use directly so
        # the rollout-stored old log-prob matches the PPO-update
        # recomputation when the model is unchanged.
        frac_z_t = torch.tensor(
            [frac_z], device=frac_logits.device, dtype=frac_logits.dtype,
        )
        # Lower-truncated Normal correction: subtract log(1 - cdf(z_min))
        # so that log_p_frac matches the truncated distribution used at
        # rollout time (see FRAC_Z_MIN and runner.py _predict).
        z_min_t = torch.tensor(
            [FRAC_Z_MIN], device=frac_logits.device, dtype=frac_logits.dtype,
        )
        p_lo = frac_dist.cdf(z_min_t.unsqueeze(0))
        p_lo = torch.clamp(p_lo, max=1.0 - 1e-6)
        log_norm = torch.log1p(-p_lo)  # log(1 - p_lo) computed safely
        log_prob = log_prob + frac_dist.log_prob(frac_z_t.unsqueeze(0)) - log_norm

    return log_prob.squeeze(0)


class ActionTrainStack(nn.Module):
    """Bundle the full action stack for gradual unfreezing.

    Mirrors :class:`CrossEntityTrainStack` but with ``cross`` (the
    standalone :class:`CrossEntityAttention`) and ``action_decoder`` as
    the trainable top instead of ``cross_model``. Encoders below stay
    frozen until the gradual-unfreeze schedule thaws them.
    """

    def __init__(
        self,
        *,
        fleet_encoder: FleetEncoder,
        planet_encoder: PlanetEncoder,
        entity_encoder: PlanetEntityEncoder,
        cross_attention: CrossEntityAttention,
        action_decoder: ActionDecoder,
        global_decoder: GlobalStateDecoder | None = None,
    ):
        super().__init__()
        self.fleet_encoder = fleet_encoder
        self.planet_encoder = planet_encoder
        self.entity_encoder = entity_encoder
        self.cross = cross_attention
        self.action_decoder = action_decoder
        self.global_decoder = global_decoder or GlobalStateDecoder(
            d_model=cross_attention.d_model,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        entity_tokens, entity_mask = _entity_tokens_per_step(
            batch,
            self.fleet_encoder,
            self.planet_encoder,
            self.entity_encoder,
        )
        ctx, glob = self.cross(entity_tokens, entity_mask)
        ctx_now = ctx[:, -1] if ctx.dim() == 4 else ctx           # (B, P, d)
        if entity_mask.dim() == 3:
            mask_now = entity_mask[:, -1]                         # (B, P)
        else:
            mask_now = entity_mask
        action_out = self.action_decoder(ctx_now, mask_now, glob)
        global_out = self.global_decoder(glob)
        return {**action_out, **global_out}


# ---------- Loss ----------
def compute_global_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    aux_coef: float = 0.2,
) -> tuple[torch.Tensor, dict[str, tuple[float, int]]]:
    """Auxiliary losses for :class:`GlobalStateDecoder` heads.

    Mirrors the loss structure of ``CrossEntityPretrainModel`` (tier-3
    global labels) so the CLS token retains game-state supervision after
    the action decoder is added on top.  Each term is multiplied by
    ``aux_coef`` so global losses cannot dominate the action losses.

    Returns ``(scaled_total, per_head)`` where ``per_head[name] =
    (raw_loss_value, n_valid)`` — the raw (unscaled) value lets callers
    report readable magnitudes regardless of ``aux_coef``.
    """
    n_horizons = max(len(CROSS_ENTITY_VALUE_HORIZONS), 1)
    losses: dict[str, tuple[float, int]] = {}
    raw_total: torch.Tensor | None = None

    def add(name: str, term: torch.Tensor, n_valid: int) -> None:
        nonlocal raw_total
        raw_total = term if raw_total is None else raw_total + term
        losses[name] = (float(term.detach()), n_valid)

    B = preds["gd_winner_seat"].shape[0]

    add(
        "gd_winner_seat",
        F.cross_entropy(preds["gd_winner_seat"], targets["winner_seat"]) * 0.3,
        B,
    )
    add(
        "gd_score_end",
        F.huber_loss(preds["gd_score_end"], targets["score_advantage_at_end_log"], delta=1.0) * 0.1,
        B,
    )
    add(
        "gd_turns_left",
        F.mse_loss(preds["gd_turns_left"], targets["turns_until_episode_end"]) * 0.1,
        B,
    )

    # Neutral (perspective-independent) auxiliary losses. Smaller weights —
    # these are regularizers, not the primary objective.
    add(
        "gd_total_ships_log",
        F.huber_loss(preds["gd_total_ships_log"], targets["total_ships_in_play_log"], delta=1.0) * 0.1,
        B,
    )
    add(
        "gd_ship_entropy",
        F.mse_loss(preds["gd_ship_entropy"], targets["ship_distribution_entropy"]) * 0.1,
        B,
    )
    add(
        "gd_n_neutral_planets",
        F.huber_loss(preds["gd_n_neutral_planets"], targets["n_neutral_planets"], delta=1.0) * 0.05,
        B,
    )
    add(
        "gd_game_phase",
        F.cross_entropy(preds["gd_game_phase"], targets["game_phase"]) * 0.15,
        B,
    )

    per_k_weight = 0.2 / n_horizons
    for k in CROSS_ENTITY_VALUE_HORIZONS:
        valid = targets[f"valid_global_t_plus_{k}"]
        n_valid = int(valid.sum())
        if n_valid == 0:
            losses[f"gd_is_ahead_{k}"] = (0.0, 0)
            losses[f"gd_leader_{k}"] = (0.0, 0)
            continue
        bce = F.binary_cross_entropy_with_logits(
            preds[f"gd_is_ahead_{k}"],
            targets[f"is_ahead_t_plus_{k}"],
            reduction="none",
        )
        add(f"gd_is_ahead_{k}", (bce * valid).sum() / valid.sum().clamp(min=1) * per_k_weight, n_valid)

        ce = F.cross_entropy(
            preds[f"gd_leader_{k}"],
            targets[f"leader_seat_t_plus_{k}"],
            reduction="none",
        )
        add(f"gd_leader_{k}", (ce * valid).sum() / valid.sum().clamp(min=1) * (per_k_weight * 0.5), n_valid)

    assert raw_total is not None
    return raw_total * aux_coef, losses


def compute_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    aux_coef: float = 0.2,
    decoder_only: bool = False,
    action_only: bool = False,
) -> tuple[torch.Tensor, dict[str, tuple[float, int]]]:
    """Return ``(total_loss, per_head)`` where ``per_head[name] =
    (loss_value, n_valid)``. Caller weights training averages by
    ``n_valid`` so source/target metrics remain comparable to val/test
    even when many batches contain no valid action labels.

    For source/target heads ``n_valid`` is the number of rows whose
    label is not ``ACTION_IGNORE_INDEX``. For ``expert_acted`` it's the
    full batch size (every row carries a binary label).
    """
    losses: dict[str, tuple[float, int]] = {}
    total: torch.Tensor | None = None

    def masked_cross_entropy(
        logits: torch.Tensor, target: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        valid = target != ACTION_IGNORE_INDEX
        n_valid = int(valid.sum())
        if n_valid == 0:
            return logits.new_zeros(()), 0
        return (
            F.cross_entropy(logits, target, ignore_index=ACTION_IGNORE_INDEX),
            n_valid,
        )

    def add(name: str, term: torch.Tensor, n_valid: int) -> None:
        nonlocal total
        total = term if total is None else total + term
        losses[name] = (float(term.detach()), int(n_valid))

    if not decoder_only:
        # Action heads — source/target/expert_acted. Skipped when training
        # the global decoder in isolation.
        src_term, src_n = masked_cross_entropy(
            preds["source_planet_logits"], targets["source_planet_idx"],
        )
        add("source_planet", src_term, src_n)

        tgt_term, tgt_n = masked_cross_entropy(
            preds["target_planet_logits"], targets["target_planet_idx"],
        )
        add("target_planet", tgt_term, tgt_n)

        bce = F.binary_cross_entropy_with_logits(
            preds["expert_acted_logit"], targets["expert_acted"],
        )
        add("expert_acted", bce, int(targets["expert_acted"].numel()))

    # Global-state losses from GlobalStateDecoder.
    # In decoder-only mode the global terms are the WHOLE objective, so
    # use aux_coef=1.0 to avoid down-weighting the only signal we have.
    # In action-only mode, skip the global losses entirely — the action
    # heads are the whole objective.
    if not action_only:
        g_coef = 1.0 if decoder_only else aux_coef
        g_total, g_per_head = compute_global_loss(preds, targets, g_coef)
        if total is None:
            total = g_total
        else:
            total = total + g_total
        losses.update(g_per_head)

    assert total is not None
    return total, losses


# ---------- Eval ----------
@torch.no_grad()
def evaluate(
    stack: ActionTrainStack,
    loader: DataLoader,
    device: str,
) -> dict[str, dict[str, float]]:
    stack.eval()
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    correct_top1: dict[str, int] = defaultdict(int)
    correct_top3: dict[str, int] = defaultdict(int)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = stack(batch)

        for name, target_name in (
            ("source_planet", "source_planet_idx"),
            ("target_planet", "target_planet_idx"),
        ):
            logits = preds[f"{name}_logits"]
            target = batch[target_name]
            valid = target != ACTION_IGNORE_INDEX
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            ce = F.cross_entropy(
                logits, target,
                ignore_index=ACTION_IGNORE_INDEX, reduction="sum",
            )
            sums[name] += float(ce)
            counts[name] += n_valid
            argmax = logits.argmax(-1)
            correct_top1[name] += int(((argmax == target) & valid).sum())
            top_k = min(3, logits.shape[-1])
            top3 = logits.topk(top_k, dim=-1).indices                # (B, k)
            target_in_top3 = (top3 == target.unsqueeze(-1)).any(-1)
            correct_top3[name] += int((target_in_top3 & valid).sum())

        acted_logits = preds["expert_acted_logit"]
        acted_target = batch["expert_acted"]
        sums["expert_acted"] += float(
            F.binary_cross_entropy_with_logits(
                acted_logits, acted_target, reduction="sum",
            )
        )
        counts["expert_acted"] += int(acted_target.shape[0])
        pred_bin = acted_logits >= 0.0
        tgt_bin = acted_target > 0.5
        correct_top1["expert_acted"] += int((pred_bin == tgt_bin).sum())

        # Global decoder heads.
        B = preds["gd_winner_seat"].shape[0]
        sums["gd_winner_seat"] += float(
            F.cross_entropy(preds["gd_winner_seat"], batch["winner_seat"], reduction="sum")
        )
        counts["gd_winner_seat"] += B
        correct_top1["gd_winner_seat"] += int(
            (preds["gd_winner_seat"].argmax(-1) == batch["winner_seat"]).sum()
        )

        # Neutral (perspective-independent) global heads.
        sums["gd_total_ships_log"] += float(
            F.huber_loss(
                preds["gd_total_ships_log"], batch["total_ships_in_play_log"],
                delta=1.0, reduction="sum",
            )
        )
        counts["gd_total_ships_log"] += B
        sums["gd_ship_entropy"] += float(
            ((preds["gd_ship_entropy"] - batch["ship_distribution_entropy"]) ** 2).sum()
        )
        counts["gd_ship_entropy"] += B
        sums["gd_n_neutral_planets"] += float(
            F.huber_loss(
                preds["gd_n_neutral_planets"], batch["n_neutral_planets"],
                delta=1.0, reduction="sum",
            )
        )
        counts["gd_n_neutral_planets"] += B
        sums["gd_game_phase"] += float(
            F.cross_entropy(preds["gd_game_phase"], batch["game_phase"], reduction="sum")
        )
        counts["gd_game_phase"] += B
        correct_top1["gd_game_phase"] += int(
            (preds["gd_game_phase"].argmax(-1) == batch["game_phase"]).sum()
        )

        for k in CROSS_ENTITY_VALUE_HORIZONS:
            valid = batch[f"valid_global_t_plus_{k}"]
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            name_ah = f"gd_is_ahead_{k}"
            bce = F.binary_cross_entropy_with_logits(
                preds[name_ah],
                batch[f"is_ahead_t_plus_{k}"],
                reduction="none",
            )
            sums[name_ah] += float((bce * valid).sum())
            counts[name_ah] += n_valid
            pred_bin_ah = preds[name_ah] >= 0.0
            tgt_bin_ah = batch[f"is_ahead_t_plus_{k}"] > 0.5
            correct_top1[name_ah] += int(((pred_bin_ah == tgt_bin_ah) & valid.bool()).sum())

    summary: dict[str, dict[str, float]] = {}
    for name, total in sums.items():
        n = max(1, counts[name])
        entry: dict[str, float] = {"loss": total / n}
        # Only report accuracy for heads that actually incremented
        # ``correct_top1`` (i.e. the categorical / binary heads). Regression
        # heads (gd_total_ships_log, gd_ship_entropy, gd_n_neutral_planets,
        # gd_score_end, gd_turns_left) leave ``correct_top1`` at 0 by
        # default, which previously printed a misleading "acc=0.000".
        if name in correct_top1:
            entry["acc"] = correct_top1[name] / n
        if name in correct_top3 and n > 0:
            entry["acc_top3"] = correct_top3[name] / n
        summary[name] = entry
    stack.train()
    return summary


ACTION_HEAD_NAMES: tuple[str, ...] = (
    "source_planet", "target_planet", "expert_acted",
)


def _val_mean(
    val: dict[str, dict[str, float]],
    *,
    decoder_only: bool = False,
    action_only: bool = False,
) -> float:
    """Average val loss across the heads being actively trained.

    Best-checkpoint selection should track what's actually being optimized.
    If we average over all heads (including frozen ones whose CLS input is
    drifting), best-ckpt picks may be dominated by drift in heads that
    have no gradient — see the action_only training run where
    `gd_total_ships_log` and `gd_ship_entropy` degraded as cross adapted
    to action loss, dragging val_mean even though the action heads were
    improving.
    """
    if action_only:
        names = [n for n in ACTION_HEAD_NAMES if n in val]
    elif decoder_only:
        names = [n for n in val if n.startswith("gd_")]
    else:
        names = list(val)
    if not names:
        # Fallback to all heads if no match (e.g. a head was renamed).
        names = list(val)
    return sum(val[n]["loss"] for n in names) / max(1, len(names))


def _format_summary(summary: dict[str, dict[str, float]]) -> str:
    lines = []
    for name, m in summary.items():
        bits = [f"loss={m['loss']:.4f}"]
        if "acc" in m:
            bits.append(f"acc={m['acc']:.3f}")
        if "acc_top3" in m:
            bits.append(f"top3={m['acc_top3']:.3f}")
        lines.append(f"    {name:<24s}  " + "  ".join(bits))
    return "\n".join(lines)


# ---------- Trainable resolver / param groups ----------
# Mirrors cross_entity._resolve_trainable_modules but uses the action-stack
# alias map (no `cross_model`; `cross` is a top-level CrossEntityAttention).
def _resolve_action_modules(stack: ActionTrainStack, path: str) -> list[nn.Module]:
    aliases = {
        "fleet_encoder": stack.fleet_encoder,
        "planet_encoder": stack.planet_encoder,
        "entity_encoder": stack.entity_encoder,
        "cross": stack.cross,
        "action_decoder": stack.action_decoder,
        "global_decoder": stack.global_decoder,
    }
    if path in aliases:
        return [aliases[path]]
    root_name, dot, tail = path.partition(".")
    if root_name in aliases:
        root = aliases[root_name]
        return [_resolve_attr(root, tail)] if dot else [root]
    raise KeyError(f"unknown trainable path: {path}")


def set_trainable(
    stack: ActionTrainStack,
    *,
    freeze: list[str],
    unfreeze: list[str],
) -> None:
    for path in freeze:
        for module in _resolve_action_modules(stack, path):
            for p in module.parameters():
                p.requires_grad_(False)
    for path in unfreeze:
        for module in _resolve_action_modules(stack, path):
            for p in module.parameters():
                p.requires_grad_(True)


def build_param_groups(
    stack: ActionTrainStack,
    lr_table: dict[str, float],
) -> list[dict[str, Any]]:
    # Longest-first so specific leaves override broader matches.
    resolved: list[tuple[str, float, set[int]]] = []
    for path, lr in sorted(lr_table.items(), key=lambda kv: len(kv[0]), reverse=True):
        params: set[int] = set()
        for module in _resolve_action_modules(stack, path):
            for p in module.parameters():
                params.add(id(p))
        resolved.append((path, lr, params))

    grouped: dict[str, dict[str, Any]] = {}
    assigned: set[int] = set()
    for name, param in stack.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in assigned:
            continue
        match: tuple[str, float, set[int]] | None = None
        for item in resolved:
            if pid in item[2]:
                match = item
                break
        if match is None:
            raise ValueError(
                f"trainable param {name!r} is missing from the LR table"
            )
        path, lr, _ = match
        group = grouped.setdefault(path, {"params": [], "lr": lr, "name": path})
        group["params"].append(param)
        assigned.add(pid)
    return [g for g in grouped.values() if g["params"]]


# ---------- Schedule ----------
def _build_schedule(stage_epochs: list[int]) -> list[dict[str, Any]]:
    """Stage schedule for gradual unfreeze.

    Stage 0 is the ``frozen`` mode (separate code path). This builder
    starts at Stage 1 — the cross-attention thaws first, then entity
    encoder, then planet/fleet top-halves, then the full stack at low LR.
    """
    templates = [
        {
            "index": 1,
            "name": "cross-unfreeze",
            "trainable_paths": ["cross", "action_decoder", "global_decoder"],
            "aux_coef": 0.15,
            "lr_table": {
                "cross": 1e-4,
                "action_decoder": 1e-3,
                "global_decoder": 1e-3,
            },
        },
        {
            "index": 2,
            "name": "entity-unfreeze",
            "trainable_paths": ["cross", "action_decoder", "global_decoder", "entity_encoder"],
            "aux_coef": 0.15,
            "lr_table": {
                "cross": 1e-4,
                "action_decoder": 1e-3,
                "global_decoder": 1e-3,
                "entity_encoder": 1e-4,
            },
        },
        {
            "index": 3,
            "name": "top-half-unfreeze",
            "trainable_paths": [
                "cross", "action_decoder", "global_decoder", "entity_encoder",
                "fleet_encoder.fc2", "fleet_encoder.norm",
                "planet_encoder.scalar.fc2", "planet_encoder.traj.proj",
                "planet_encoder.gate", "planet_encoder.norm",
            ],
            "aux_coef": 0.10,
            "lr_table": {
                "cross": 1e-4,
                "action_decoder": 1e-3,
                "global_decoder": 1e-3,
                "entity_encoder": 1e-4,
                "fleet_encoder.fc2": 1e-5,
                "fleet_encoder.norm": 1e-5,
                "planet_encoder.scalar.fc2": 1e-5,
                "planet_encoder.traj.proj": 1e-5,
                "planet_encoder.gate": 1e-5,
                "planet_encoder.norm": 1e-5,
            },
        },
        {
            "index": 4,
            "name": "full-unfreeze",
            "trainable_paths": [
                "cross", "action_decoder", "global_decoder", "entity_encoder",
                "fleet_encoder", "planet_encoder",
            ],
            "aux_coef": 0.10,
            "lr_table": {
                "cross": 1e-4,
                "action_decoder": 1e-3,
                "global_decoder": 1e-3,
                "entity_encoder": 1e-4,
                "fleet_encoder.fc2": 1e-5,
                "fleet_encoder.norm": 1e-5,
                "fleet_encoder.fc1": 1e-6,
                "planet_encoder.scalar.fc2": 1e-5,
                "planet_encoder.traj.proj": 1e-5,
                "planet_encoder.gate": 1e-5,
                "planet_encoder.norm": 1e-5,
                "planet_encoder.scalar.fc1": 1e-6,
                "planet_encoder.traj.conv1": 1e-6,
                "planet_encoder.traj.conv2": 1e-6,
            },
        },
    ]

    if not stage_epochs:
        raise ValueError("stage_epochs must contain at least one entry")
    if len(stage_epochs) > len(templates):
        raise ValueError(
            f"got {len(stage_epochs)} stage epochs, but only {len(templates)} "
            "gradual-unfreeze stages exist"
        )

    schedule: list[dict[str, Any]] = []
    start_epoch = 1
    for template, n_epochs in zip(templates, stage_epochs):
        if n_epochs <= 0:
            continue
        stage = dict(template)
        stage["epochs"] = n_epochs
        stage["start_epoch"] = start_epoch
        stage["end_epoch"] = start_epoch + n_epochs - 1
        schedule.append(stage)
        start_epoch += n_epochs
    return schedule


def _stage_for_epoch(
    schedule: list[dict[str, Any]],
    epoch: int,
) -> dict[str, Any]:
    for stage in schedule:
        if stage["start_epoch"] <= epoch <= stage["end_epoch"]:
            return stage
    raise ValueError(f"epoch {epoch} is outside the gradual-unfreeze schedule")


# ---------- Checkpoint loading ----------
def _load_cross_attention(
    ckpt_path: Path,
    *,
    d_model: int,
    device: str,
) -> tuple[CrossEntityAttention, dict[str, Any]]:
    """Load ONLY the cross-attention sub-module from a cross-entity ckpt.

    Drops the multi-task heads from the cross-entity training run; we
    don't reuse them in the action pipeline.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("cross_model") or ckpt.get("model")
    if state is None:
        raise KeyError(
            f"{ckpt_path} contains no `cross_model` / `model` state"
        )
    cross_state = {
        k.removeprefix("cross."): v
        for k, v in state.items()
        if k.startswith("cross.")
    }
    if not cross_state:
        raise KeyError(
            f"{ckpt_path}'s cross_model state has no `cross.*` keys "
            "(needs to come from a cross-entity pretrain run)"
        )
    cross = CrossEntityAttention(d_model=d_model, n_heads=4, n_layers=3)
    cross.load_state_dict(cross_state)
    return cross, ckpt


def _load_action_checkpoint(
    ckpt_path: Path,
    *,
    d_model: int,
    device: str,
) -> tuple[CrossEntityAttention, nn.Module, GlobalStateDecoder, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "cross" not in ckpt or "action_decoder" not in ckpt:
        raise KeyError(
            f"{ckpt_path} is not an action checkpoint "
            "(missing `cross` or `action_decoder` keys)"
        )
    cross = CrossEntityAttention(d_model=d_model, n_heads=4, n_layers=3)
    cross.load_state_dict(ckpt["cross"])
    # Pick the decoder flavor that was saved with this ckpt; default to the
    # flat ActionDecoder for backward compatibility with pre-flag ckpts.
    contextual = bool(
        ckpt.get("config", {}).get("contextual_action_decoder", False)
    )
    decoder = _build_action_decoder(d_model, contextual=contextual)
    missing, unexpected = decoder.load_state_dict(
        ckpt["action_decoder"], strict=False,
    )
    ppo_heads = {
        "head_value", "head_frac", "frac_log_std", "value_bn",
        "head_value_h",
    }
    missing_ppo = [m for m in missing if any(m.startswith(p) for p in ppo_heads)]
    if missing_ppo:
        print(
            f"[expert_action] PPO heads not in ckpt {ckpt_path.name}: "
            f"{missing_ppo} (initialized from scratch)",
            flush=True,
        )
    # Architecture-mismatch warning: if a non-trivial number of head_*
    # weights were skipped, the saved ckpt almost certainly used the OTHER
    # decoder class. Surface that loud-and-clear so the user can decide.
    head_misses = [
        m for m in missing
        if m.startswith(("head_source", "head_target", "head_frac"))
        and not any(m.startswith(p) for p in ppo_heads)
    ]
    if head_misses:
        print(
            f"[expert_action] {ckpt_path.name}: action-head keys missing "
            f"({head_misses[:3]}{'...' if len(head_misses) > 3 else ''}) — "
            f"likely a decoder-class mismatch (loaded contextual={contextual}). "
            "These heads will train from scratch.",
            flush=True,
        )
    global_dec = GlobalStateDecoder(d_model=d_model)
    if "global_decoder" in ckpt:
        global_dec.load_state_dict(ckpt["global_decoder"])
    else:
        print(
            f"[expert_action] global_decoder not in ckpt {ckpt_path.name} "
            "(initialized from scratch)",
            flush=True,
        )
    return cross, decoder, global_dec, ckpt


def load_for_inference(
    ckpt_path: str | Path,
    *,
    device: str = "cpu",
    d_model: int = 64,
) -> ActionTrainStack:
    """Load a fully-wired :class:`ActionTrainStack` for inference.

    Reads the action checkpoint produced by :func:`train_frozen` /
    :func:`train_gradual_unfreeze` and rebuilds the full stack
    (fleet/planet/entity encoders + cross attention + action decoder).
    Encoder weights live IN the action ckpt (gradual-unfreeze updates
    them) so we don't need separate fleet/planet/entity ckpt paths
    here — except for the very first ``frozen-init`` ckpt, which may
    omit them. In that case the caller passes encoder ckpt dirs as
    fallback.
    """
    ckpt_path = Path(ckpt_path)
    cross, decoder, global_dec, ckpt = _load_action_checkpoint(
        ckpt_path, d_model=d_model, device=device,
    )

    # The action ckpt always includes encoder state dicts (saved by
    # :func:`_save_checkpoint`). Build empty modules and load.
    fleet_encoder = FleetEncoder(d_model=d_model)
    if "fleet_encoder" not in ckpt:
        raise KeyError(
            f"{ckpt_path} has no `fleet_encoder` state dict — "
            "this isn't a fully-saved action ckpt."
        )
    fleet_encoder.load_state_dict(ckpt["fleet_encoder"])

    planet_encoder = PlanetEncoder(d_model=d_model)
    planet_encoder.load_state_dict(ckpt["planet_encoder"])

    entity_encoder = PlanetEntityEncoder(d_model=d_model)
    entity_encoder.load_state_dict(ckpt["entity_encoder"])

    stack = ActionTrainStack(
        fleet_encoder=fleet_encoder,
        planet_encoder=planet_encoder,
        entity_encoder=entity_encoder,
        cross_attention=cross,
        action_decoder=decoder,
        global_decoder=global_dec,
    ).to(device).eval()
    for p in stack.parameters():
        p.requires_grad_(False)
    return stack


def _save_checkpoint(
    path: Path,
    *,
    stack: ActionTrainStack,
    epoch: int,
    stage_index: int | None,
    stage_name: str | None,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "cross": stack.cross.state_dict(),
        "action_decoder": stack.action_decoder.state_dict(),
        "global_decoder": stack.global_decoder.state_dict(),
        "fleet_encoder": stack.fleet_encoder.state_dict(),
        "planet_encoder": stack.planet_encoder.state_dict(),
        "entity_encoder": stack.entity_encoder.state_dict(),
        "epoch": epoch,
        "stage": stage_index,
        "stage_name": stage_name,
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# ---------- Dataset construction ----------
def _splits_cache_dir() -> Path:
    return ACTION_DATASET_DIR / "_snapshot_cache"


def _splits_cache_key(
    *,
    split: str,
    max_planets: int,
    max_fleets: int,
    n_history: int,
    max_episodes: int | None,
    stems: list[str],
) -> str:
    """Stable key for the snapshot cache. Encodes everything that can
    invalidate the materialized snapshots: tensor shapes, n_history,
    episode set, dataset version. Stems are hashed (sorted, joined) so
    rerunning featurization on the same set keeps the cache valid even
    if file mtimes shift.
    """
    import hashlib
    h = hashlib.sha1()
    h.update(f"v1|{split}|{max_planets}|{max_fleets}|{n_history}|".encode())
    h.update(f"max_eps={max_episodes}|".encode())
    h.update("\n".join(sorted(stems)).encode())
    return h.hexdigest()[:16]


def _build_splits(
    *,
    max_planets: int,
    max_fleets: int,
    max_episodes: int | None = None,
    num_load_workers: int | None = None,
    use_cache: bool = True,
) -> dict[str, ActionSnapshotDataset]:
    """Build train/val/test ActionSnapshotDatasets from the action manifest.

    Uses the *action* manifest as the source of truth, but filters each
    split to stems where all five CSVs (action, cross_entity, entity,
    planet, fleet) exist. That keeps the splits coherent even when the
    action featurizer was run on episodes the other featurizers haven't
    processed yet.

    ``max_episodes`` (per split) caps each split's stem count after
    filtering. Use this to fit the dataset into Colab T4 RAM — each
    train episode adds ~10 MB of resident snapshot tensors.

    ``use_cache=True`` (default) materializes each split once, then
    serializes the snapshot tensors to ``data/datasets/action/_snapshot_cache/``.
    Subsequent runs with the same ``(stems, max_planets, max_fleets,
    n_history, max_episodes)`` skip CSV parsing entirely — typically
    100x speedup on Colab where disk reads + Python CSV parse dominate.
    """
    manifest = json.loads((ACTION_DATASET_DIR / "manifest.json").read_text())

    def stems_of(split: str) -> list[str]:
        return [
            n.removeprefix("action_").removesuffix(".csv")
            for n in manifest[split]
        ]

    def filter_stems(stems: list[str]) -> tuple[list[str], int]:
        kept: list[str] = []
        for s in stems:
            if not all((
                (PLANET_DATASET_DIR / f"planet_{s}.csv").exists(),
                (FLEET_DATASET_DIR / f"fleet_{s}.csv").exists(),
                (ENTITY_DATASET_DIR / f"entity_{s}.csv").exists(),
                (CROSS_ENTITY_DATASET_DIR / f"cross_entity_{s}.csv").exists(),
                (ACTION_DATASET_DIR / f"action_{s}.csv").exists(),
            )):
                continue
            kept.append(s)
        return kept, len(stems) - len(kept)

    splits: dict[str, ActionSnapshotDataset] = {}
    kept_per_split: dict[str, int] = {}
    for split in ("train", "val", "test"):
        all_stems = stems_of(split)
        stems, dropped = filter_stems(all_stems)
        if max_episodes is not None and len(stems) > max_episodes:
            print(
                f"  [{split}] capping {len(stems)} → {max_episodes} stems "
                f"(--max-episodes)",
                flush=True,
            )
            stems = stems[:max_episodes]
        kept_per_split[split] = len(stems)
        if dropped:
            print(
                f"  [{split}] dropped {dropped} stems missing one or more "
                "co-required CSVs (planet/fleet/entity/cross_entity)",
                flush=True,
            )
        cache_path = None
        if use_cache:
            cache_key = _splits_cache_key(
                split=split,
                max_planets=max_planets,
                max_fleets=max_fleets,
                n_history=3,
                max_episodes=max_episodes,
                stems=stems,
            )
            cache_path = _splits_cache_dir() / f"{split}_{cache_key}.pt"
            if cache_path.exists():
                t_load = time.time()
                splits[split] = ActionSnapshotDataset.from_cache(cache_path)
                print(
                    f"  [{split}] loaded {len(splits[split])} snapshots "
                    f"from cache ({len(stems)} episodes, "
                    f"{time.time() - t_load:.1f}s) "
                    f"→ {cache_path.name}",
                    flush=True,
                )
                continue
        print(
            f"  [{split}] building dataset from {len(stems)} stems "
            f"({len(stems) * 5} CSVs)...",
            flush=True,
        )
        t_split = time.time()
        splits[split] = ActionSnapshotDataset(
            planet_csv_paths=[PLANET_DATASET_DIR / f"planet_{s}.csv" for s in stems],
            fleet_csv_paths=[FLEET_DATASET_DIR / f"fleet_{s}.csv" for s in stems],
            entity_csv_paths=[ENTITY_DATASET_DIR / f"entity_{s}.csv" for s in stems],
            cross_entity_csv_paths=[
                CROSS_ENTITY_DATASET_DIR / f"cross_entity_{s}.csv" for s in stems
            ],
            action_csv_paths=[
                ACTION_DATASET_DIR / f"action_{s}.csv" for s in stems
            ],
            max_planets=max_planets,
            max_fleets=max_fleets,
            num_load_workers=num_load_workers,
        )
        print(
            f"  {split}: {len(splits[split])} snapshots "
            f"({len(stems)} episodes, {time.time() - t_split:.1f}s)",
            flush=True,
        )
        if cache_path is not None:
            t_cache = time.time()
            splits[split].save_cache(cache_path)
            print(
                f"    cached → {cache_path.name} "
                f"({time.time() - t_cache:.1f}s)",
                flush=True,
            )

    # Hard preflight: refuse to train on a split that's been silently
    # gutted by the all-5-CSVs intersection. Picking the best ckpt off
    # one or two snapshots is meaningless.
    MIN_EPISODES_PER_SPLIT = 3
    bad = [s for s in ("train", "val", "test") if kept_per_split[s] < MIN_EPISODES_PER_SPLIT]
    if bad:
        details = ", ".join(f"{s}={kept_per_split[s]}" for s in ("train", "val", "test"))
        raise SystemExit(
            f"action splits too small after intersecting CSVs ({details}); "
            f"each split needs ≥ {MIN_EPISODES_PER_SPLIT} episodes. "
            "Rerun `python scripts/build_encoder_dataset.py "
            "--datasets fleet,planet,entity,cross_entity,action` to "
            "expand coverage."
        )
    if any(len(splits[s]) == 0 for s in ("train", "val", "test")):
        raise SystemExit(
            "action splits contain 0 snapshots — featurization is "
            "incomplete; rerun build_encoder_dataset.py."
        )
    return splits


# ---------- Frozen training (Stage 0) ----------
def train_frozen(
    *,
    out_dir: Path,
    fleet_run_dir: Path,
    planet_run_dir: Path,
    entity_run_dir: Path,
    resume_cross_pt: Path,
    d_model: int = 64,
    batch_size: int = 64,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_every: int = 1,
    max_planets: int = 64,
    max_fleets: int = 256,
    max_episodes: int | None = None,
    num_load_workers: int | None = None,
    num_workers: int = 0,
    device: str | None = None,
    seed: int = 0,
    decoder_only: bool = False,
    action_only: bool = False,
    unfreeze_cross: bool = False,
    contextual_action_decoder: bool = False,
    early_stop_patience: int | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # Build the stack + save an initial checkpoint BEFORE the slow CSV
    # parse so a Colab kill / OOM during dataset build still leaves a
    # resumable action_last.pt on disk.
    print(f"[action-frozen] loading upstream encoders ({device}) ...", flush=True)
    fenc, penc, eenc = _load_frozen_encoders(
        fleet_run_dir, planet_run_dir, entity_run_dir, device=device,
    )
    print(f"[action-frozen] loading cross-attention from {resume_cross_pt}", flush=True)
    cross, _ = _load_cross_attention(
        resume_cross_pt, d_model=d_model, device=device,
    )
    cross.to(device)
    decoder = _build_action_decoder(
        d_model, contextual=contextual_action_decoder,
    ).to(device)
    global_dec = GlobalStateDecoder(d_model=d_model).to(device)
    stack = ActionTrainStack(
        fleet_encoder=fenc, planet_encoder=penc, entity_encoder=eenc,
        cross_attention=cross, action_decoder=decoder,
        global_decoder=global_dec,
    ).to(device)
    if decoder_only and action_only:
        raise ValueError(
            "--decoder-only and --action-only are mutually exclusive"
        )

    _freeze_all(stack)
    if decoder_only:
        unfreeze_paths = ["global_decoder"]
    elif action_only:
        unfreeze_paths = ["action_decoder"]
    else:
        unfreeze_paths = ["action_decoder", "global_decoder"]
    if unfreeze_cross:
        unfreeze_paths.append("cross")
    set_trainable(stack, freeze=[], unfreeze=unfreeze_paths)
    aux_coef = 0.2
    lr_table: dict[str, float] = {}
    if not action_only:
        lr_table["global_decoder"] = lr
    if not decoder_only:
        lr_table["action_decoder"] = lr
    if unfreeze_cross:
        # Cross-attention LR is one decade lower — typical pattern when
        # thawing a layer below the trainable head (see gradual-unfreeze
        # schedule).
        lr_table["cross"] = lr * 0.1
    opt = torch.optim.AdamW(
        build_param_groups(stack, lr_table),
        weight_decay=weight_decay,
    )
    print(
        f"[action-frozen] decoder_only={decoder_only}  action_only={action_only}  "
        f"unfreeze_cross={unfreeze_cross}  "
        f"action_decoder params: {sum(p.numel() for p in decoder.parameters()):,}  "
        f"global_decoder params: {sum(p.numel() for p in global_dec.parameters()):,}  "
        f"unfreeze: {unfreeze_paths}  trainable groups: {len(opt.param_groups)}",
        flush=True,
    )

    config = {
        "train_mode": "frozen",
        "decoder_only": decoder_only,
        "action_only": action_only,
        "unfreeze_cross": unfreeze_cross,
        "contextual_action_decoder": contextual_action_decoder,
        "d_model": d_model, "lr": lr, "weight_decay": weight_decay,
        "batch_size": batch_size, "epochs": epochs, "aux_coef": aux_coef,
        "fleet_run_dir": str(fleet_run_dir),
        "planet_run_dir": str(planet_run_dir),
        "entity_run_dir": str(entity_run_dir),
        "resume_cross_pt": str(resume_cross_pt),
    }

    log: list[dict[str, Any]] = []
    best_val = float("inf")
    epochs_since_improvement = 0
    best_path = out_dir / "action_best.pt"
    last_path = out_dir / "action_last.pt"

    _save_checkpoint(
        last_path, stack=stack, epoch=0,
        stage_index=0, stage_name="frozen-init", config=config,
    )
    print(
        f"[action-frozen] saved initial {last_path.name} "
        f"(pre-dataset, resumable)",
        flush=True,
    )

    print("[action-frozen] loading CSVs ...", flush=True)
    splits = _build_splits(
        max_planets=max_planets,
        max_fleets=max_fleets,
        max_episodes=max_episodes,
        num_load_workers=num_load_workers,
    )

    train_loader = DataLoader(
        splits["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        splits["val"], batch_size=batch_size, num_workers=num_workers,
    )
    test_loader = DataLoader(
        splits["test"], batch_size=batch_size, num_workers=num_workers,
    )

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        print(f"[action-frozen] epoch {epoch}/{epochs} starting...", flush=True)
        stack.train()
        running_total = 0.0
        n_batches = 0
        running_loss_sum: dict[str, float] = defaultdict(float)
        running_valid_sum: dict[str, int] = defaultdict(int)
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = stack(batch)
            total_loss, per_head = compute_loss(
                preds, batch, aux_coef,
                decoder_only=decoder_only, action_only=action_only,
            )
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            running_total += float(total_loss.detach())
            # Weight by valid-row count so train loss is comparable to
            # val/test (which normalize by valid rows). Source/target
            # heads contribute 0 rows on no-action batches; expert_acted
            # always contributes batch size.
            for k, (loss_v, n_valid) in per_head.items():
                running_loss_sum[k] += loss_v * n_valid
                running_valid_sum[k] += n_valid
            n_batches += 1

        train_total = running_total / max(1, n_batches)
        train_per_head = {
            k: running_loss_sum[k] / max(1, running_valid_sum[k])
            for k in running_loss_sum
        }
        elapsed = round(time.time() - t0, 2)
        entry: dict[str, Any] = {
            "epoch": epoch, "train_total": train_total,
            "train_per_head": train_per_head,
            "train_per_head_n_valid": dict(running_valid_sum),
            "elapsed_s": elapsed,
        }

        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate(stack, val_loader, device)
            mean = _val_mean(
                val, decoder_only=decoder_only, action_only=action_only,
            )
            entry["val_mean_loss"] = mean
            entry["val"] = val
            print(
                f"[ep {epoch:>2}/{epochs}]  train_total={train_total:.4f}  "
                f"val_mean={mean:.4f}  ({elapsed}s)"
            )
            if mean < best_val:
                best_val = mean
                epochs_since_improvement = 0
                _save_checkpoint(
                    best_path, stack=stack, epoch=epoch,
                    stage_index=0, stage_name="frozen", config=config,
                )
            else:
                epochs_since_improvement += 1

        log.append(entry)
        _save_checkpoint(
            last_path, stack=stack, epoch=epoch,
            stage_index=0, stage_name="frozen", config=config,
        )
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

        if (
            early_stop_patience is not None
            and epochs_since_improvement >= early_stop_patience
        ):
            print(
                f"[action-frozen] early stop at epoch {epoch}: "
                f"val hasn't improved in {early_stop_patience} epochs "
                f"(best_val={best_val:.4f})",
                flush=True,
            )
            break

    print("\n[action-frozen] evaluating best on test ...")
    cross_b, decoder_b, global_dec_b, _ = _load_action_checkpoint(
        best_path, d_model=d_model, device=device,
    )
    stack.cross.load_state_dict(cross_b.state_dict())
    stack.action_decoder.load_state_dict(decoder_b.state_dict())
    stack.global_decoder.load_state_dict(global_dec_b.state_dict())
    test = evaluate(stack, test_loader, device)
    print(_format_summary(test))
    (out_dir / "test_summary.json").write_text(json.dumps(test, indent=2))
    print(f"\n[action-frozen] outputs in {out_dir}")
    return best_path


# ---------- Gradual-unfreeze training (Stages 1-4) ----------
def train_gradual_unfreeze(
    *,
    out_dir: Path,
    fleet_run_dir: Path,
    planet_run_dir: Path,
    entity_run_dir: Path,
    resume_action_pt: Path,
    d_model: int = 64,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    eval_every: int = 1,
    max_planets: int = 64,
    max_fleets: int = 256,
    max_episodes: int | None = None,
    num_load_workers: int | None = None,
    num_workers: int = 0,
    device: str | None = None,
    seed: int = 0,
    stage_epochs: list[int] | None = None,
    action_only: bool = False,
    early_stop_patience: int | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    stage_epochs = stage_epochs or [5]
    schedule = _build_schedule(stage_epochs)
    total_epochs = schedule[-1]["end_epoch"]

    # Build the stack + save an initial checkpoint BEFORE the slow CSV
    # parse so a Colab kill / OOM during dataset build still leaves a
    # resumable action_last.pt on disk.
    print(f"[action-gradual] loading upstream encoders ({device}) ...", flush=True)
    fenc, penc, eenc = _load_frozen_encoders(
        fleet_run_dir, planet_run_dir, entity_run_dir, device=device,
    )
    print(f"[action-gradual] loading action ckpt {resume_action_pt}", flush=True)
    cross, decoder, global_dec, resume_ckpt = _load_action_checkpoint(
        resume_action_pt, d_model=d_model, device=device,
    )
    cross.to(device)
    decoder.to(device)
    global_dec.to(device)

    # If the resumed action ckpt has fine-tuned encoder weights, load them.
    # Otherwise the upstream encoder ckpts stay as-is.
    if resume_ckpt.get("fleet_encoder"):
        fenc.load_state_dict(resume_ckpt["fleet_encoder"])
    if resume_ckpt.get("planet_encoder"):
        penc.load_state_dict(resume_ckpt["planet_encoder"])
    if resume_ckpt.get("entity_encoder"):
        eenc.load_state_dict(resume_ckpt["entity_encoder"])

    stack = ActionTrainStack(
        fleet_encoder=fenc, planet_encoder=penc, entity_encoder=eenc,
        cross_attention=cross, action_decoder=decoder,
        global_decoder=global_dec,
    ).to(device)

    config = {
        "train_mode": "gradual-unfreeze",
        # Persist the decoder flavor so future loads instantiate the right
        # class. Inferred from the type of `decoder` since gradual-unfreeze
        # always inherits the choice from the resumed action ckpt.
        "contextual_action_decoder": isinstance(decoder, ContextualActionDecoder),
        "d_model": d_model, "weight_decay": weight_decay,
        "batch_size": batch_size,
        "fleet_run_dir": str(fleet_run_dir),
        "planet_run_dir": str(planet_run_dir),
        "entity_run_dir": str(entity_run_dir),
        "resume_action_pt": str(resume_action_pt),
        "stage_epochs": list(stage_epochs),
        "schedule": [
            {
                "index": s["index"], "name": s["name"],
                "start_epoch": s["start_epoch"], "end_epoch": s["end_epoch"],
                "epochs": s["epochs"], "lr_table": dict(s["lr_table"]),
            }
            for s in schedule
        ],
    }

    print("[action-gradual] schedule:", flush=True)
    for stage in schedule:
        print(
            f"  stage {stage['index']}: {stage['name']}  "
            f"epochs={stage['start_epoch']}..{stage['end_epoch']}",
            flush=True,
        )

    log: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = out_dir / "action_best.pt"
    last_path = out_dir / "action_last.pt"
    current_stage_idx: int | None = None
    opt: torch.optim.Optimizer | None = None

    _save_checkpoint(
        last_path, stack=stack, epoch=0,
        stage_index=None, stage_name="gradual-init", config=config,
    )
    print(
        f"[action-gradual] saved initial {last_path.name} "
        f"(pre-dataset, resumable)",
        flush=True,
    )

    print("[action-gradual] loading CSVs ...", flush=True)
    splits = _build_splits(
        max_planets=max_planets,
        max_fleets=max_fleets,
        max_episodes=max_episodes,
        num_load_workers=num_load_workers,
    )

    train_loader = DataLoader(
        splits["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        splits["val"], batch_size=batch_size, num_workers=num_workers,
    )
    test_loader = DataLoader(
        splits["test"], batch_size=batch_size, num_workers=num_workers,
    )

    t0 = time.time()
    epochs_since_improvement = 0
    early_stopped_at: int | None = None
    for epoch in range(1, total_epochs + 1):
        stage = _stage_for_epoch(schedule, epoch)
        if stage["index"] != current_stage_idx:
            # In action-only mode the global decoder stays frozen — strip
            # it from this stage's trainable_paths and lr_table.
            stage_trainable = list(stage["trainable_paths"])
            stage_lr_table = dict(stage["lr_table"])
            if action_only:
                stage_trainable = [p for p in stage_trainable if p != "global_decoder"]
                stage_lr_table.pop("global_decoder", None)
            _freeze_all(stack)
            set_trainable(
                stack, freeze=[], unfreeze=stage_trainable,
            )
            opt = torch.optim.AdamW(
                build_param_groups(stack, stage_lr_table),
                weight_decay=weight_decay,
            )
            current_stage_idx = stage["index"]
            print(
                f"[stage] {stage['index']} {stage['name']}  "
                f"epochs={stage['start_epoch']}..{stage['end_epoch']}  "
                f"param_groups={len(opt.param_groups)}  "
                f"action_only={action_only}",
                flush=True,
            )

        assert opt is not None
        print(
            f"[action-gradual] epoch {epoch}/{total_epochs} starting "
            f"(stage {stage['index']} {stage['name']})...",
            flush=True,
        )
        stack.train()
        running_total = 0.0
        n_batches = 0
        running_loss_sum: dict[str, float] = defaultdict(float)
        running_valid_sum: dict[str, int] = defaultdict(int)
        stage_aux_coef = stage.get("aux_coef", 0.1)
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = stack(batch)
            total_loss, per_head = compute_loss(
                preds, batch, stage_aux_coef, action_only=action_only,
            )
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            running_total += float(total_loss.detach())
            for k, (loss_v, n_valid) in per_head.items():
                running_loss_sum[k] += loss_v * n_valid
                running_valid_sum[k] += n_valid
            n_batches += 1

        train_total = running_total / max(1, n_batches)
        train_per_head = {
            k: running_loss_sum[k] / max(1, running_valid_sum[k])
            for k in running_loss_sum
        }
        elapsed = round(time.time() - t0, 2)
        entry: dict[str, Any] = {
            "epoch": epoch, "stage": stage["index"], "stage_name": stage["name"],
            "train_total": train_total, "train_per_head": train_per_head,
            "train_per_head_n_valid": dict(running_valid_sum),
            "elapsed_s": elapsed,
        }

        if epoch % eval_every == 0 or epoch == total_epochs:
            val = evaluate(stack, val_loader, device)
            mean = _val_mean(val, action_only=action_only)
            entry["val_mean_loss"] = mean
            entry["val"] = val
            print(
                f"[ep {epoch:>2}/{total_epochs}]  stage={stage['index']} "
                f"{stage['name']}  train_total={train_total:.4f}  "
                f"val_mean={mean:.4f}  ({elapsed}s)"
            )
            if mean < best_val:
                best_val = mean
                epochs_since_improvement = 0
                _save_checkpoint(
                    best_path, stack=stack, epoch=epoch,
                    stage_index=stage["index"], stage_name=stage["name"],
                    config=config,
                )
            else:
                epochs_since_improvement += 1

        if epoch == stage["end_epoch"]:
            stage_path = out_dir / f"action_stage{stage['index']}_epoch{epoch}.pt"
            _save_checkpoint(
                stage_path, stack=stack, epoch=epoch,
                stage_index=stage["index"], stage_name=stage["name"],
                config=config,
            )

        log.append(entry)
        _save_checkpoint(
            last_path, stack=stack, epoch=epoch,
            stage_index=stage["index"], stage_name=stage["name"],
            config=config,
        )
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))

        # Early stop: bail if val hasn't improved in `early_stop_patience`
        # consecutive eval epochs. action_last.pt + action_best.pt are
        # already saved above so no work is lost.
        if (
            early_stop_patience is not None
            and epochs_since_improvement >= early_stop_patience
        ):
            early_stopped_at = epoch
            print(
                f"[action-gradual] early stop at epoch {epoch}: "
                f"val hasn't improved in {early_stop_patience} epochs "
                f"(best_val={best_val:.4f})",
                flush=True,
            )
            break

    print("\n[action-gradual] evaluating best on test ...")
    cross_b, decoder_b, global_dec_b, ckpt_b = _load_action_checkpoint(
        best_path, d_model=d_model, device=device,
    )
    stack.cross.load_state_dict(cross_b.state_dict())
    stack.action_decoder.load_state_dict(decoder_b.state_dict())
    stack.global_decoder.load_state_dict(global_dec_b.state_dict())
    if ckpt_b.get("fleet_encoder"):
        stack.fleet_encoder.load_state_dict(ckpt_b["fleet_encoder"])
    if ckpt_b.get("planet_encoder"):
        stack.planet_encoder.load_state_dict(ckpt_b["planet_encoder"])
    if ckpt_b.get("entity_encoder"):
        stack.entity_encoder.load_state_dict(ckpt_b["entity_encoder"])
    test = evaluate(stack, test_loader, device)
    print(_format_summary(test))
    (out_dir / "test_summary.json").write_text(json.dumps(test, indent=2))
    print(f"\n[action-gradual] outputs in {out_dir}")
    return best_path


# ---------- CLI ----------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-mode",
        choices=("frozen", "gradual-unfreeze"),
        default="frozen",
        help="`frozen` trains only the action decoder on top of a fixed "
             "cross-entity ckpt; `gradual-unfreeze` resumes from an "
             "action ckpt and thaws lower layers in stages.",
    )
    parser.add_argument("--fleet-run-dir", type=Path, default=None,
                        help="Default: latest under data/runs/fleet/")
    parser.add_argument("--planet-run-dir", type=Path, default=None)
    parser.add_argument("--entity-run-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-cross-pt", type=Path, default=None,
        help="Cross-entity checkpoint (.pt) to seed cross-attention from. "
             "Default: latest `cross_entity_best.pt` under "
             "data/runs/cross_entity/. Used in `--train-mode frozen`.",
    )
    parser.add_argument(
        "--resume-action-pt", type=Path, default=None,
        help="Action checkpoint (.pt) to resume from for "
             "`--train-mode gradual-unfreeze`. Default: latest "
             "`action_best.pt` under data/runs/action/.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--stage-epochs", type=str, default="5",
        help="Comma-separated epoch counts for gradual-unfreeze stages "
             "(starts at Stage 1). Examples: `5`, `5,5`, `5,5,5`, `5,5,5,5`.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-planets", type=int, default=64)
    parser.add_argument("--max-fleets", type=int, default=256)
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Cap each split's episode count. Default: no cap. "
             "Halve the corpus with e.g. `--max-episodes 240` (with the "
             "current 474-train manifest).",
    )
    parser.add_argument(
        "--num-load-workers", type=int, default=None,
        help="Process-pool workers for parallel CSV parsing during "
             "dataset construction. Default: min(8, cpu_count-1). "
             "Set 1 to disable parallelism.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--decoder-only", action="store_true",
        help="Train ONLY the GlobalStateDecoder (skip ActionDecoder + its losses). "
             "Use to grow the CLS representation without coupling to action heads.",
    )
    parser.add_argument(
        "--action-only", action="store_true",
        help="Train ONLY the action heads (skip GlobalStateDecoder + its losses). "
             "Useful when fine-tuning a contextual ActionDecoder on top of an "
             "already-trained CLS without further auxiliary supervision.",
    )
    parser.add_argument(
        "--unfreeze-cross", action="store_true",
        help="In `--train-mode frozen`, also unfreeze the cross-attention "
             "alongside the decoder(s) at LR = lr * 0.1.",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=None,
        help="If set, stop training when val_mean hasn't improved for N "
             "consecutive eval epochs. action_best.pt holds the best ckpt.",
    )
    parser.add_argument(
        "--contextual-action-decoder", action="store_true",
        help="Use ContextualActionDecoder (per-planet heads read [glob, ctx]) "
             "instead of the default ActionDecoder. Strongly recommended once "
             "the GlobalStateDecoder has trained the CLS to a useful state. "
             "In `--train-mode frozen` this picks the class on construction; "
             "in `--train-mode gradual-unfreeze` it's inferred from the "
             "resumed checkpoint.",
    )
    args = parser.parse_args()

    fleet_run = args.fleet_run_dir or _latest_run_dir(
        FLEET_RUNS_DIR, "fleet_encoder_best.pt",
    )
    planet_run = args.planet_run_dir or _latest_run_dir(
        PLANET_RUNS_DIR, "planet_encoder_best.pt",
    )
    entity_run = args.entity_run_dir or _latest_run_dir(
        ENTITY_RUNS_DIR, "entity_encoder_best.pt",
    )
    out_dir = args.out_dir or (ACTION_RUNS_DIR / time.strftime("%Y%m%d-%H%M%S"))

    if args.train_mode == "frozen":
        resume_cross_pt = args.resume_cross_pt
        if resume_cross_pt is None:
            resume_cross_pt = (
                _latest_run_dir(CROSS_ENTITY_RUNS_DIR, "cross_entity_best.pt")
                / "cross_entity_best.pt"
            )
        train_frozen(
            out_dir=out_dir,
            fleet_run_dir=fleet_run,
            planet_run_dir=planet_run,
            entity_run_dir=entity_run,
            resume_cross_pt=resume_cross_pt,
            d_model=args.d_model,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            max_planets=args.max_planets,
            max_fleets=args.max_fleets,
            max_episodes=args.max_episodes,
            num_load_workers=args.num_load_workers,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            decoder_only=args.decoder_only,
            action_only=args.action_only,
            unfreeze_cross=args.unfreeze_cross,
            contextual_action_decoder=args.contextual_action_decoder,
            early_stop_patience=args.early_stop_patience,
        )
        return

    stage_epochs = [
        int(part.strip())
        for part in args.stage_epochs.split(",")
        if part.strip()
    ]
    resume_action_pt = args.resume_action_pt
    if resume_action_pt is None:
        # Prefer best, fall back to last — useful right after a fresh
        # frozen run on a tiny dataset where val never improves and
        # action_best.pt never gets written.
        try:
            resume_action_pt = (
                _latest_run_dir(ACTION_RUNS_DIR, "action_best.pt")
                / "action_best.pt"
            )
        except FileNotFoundError:
            try:
                resume_action_pt = (
                    _latest_run_dir(ACTION_RUNS_DIR, "action_last.pt")
                    / "action_last.pt"
                )
                print(
                    f"[action] no action_best.pt found; "
                    f"resuming from {resume_action_pt}"
                )
            except FileNotFoundError:
                raise SystemExit(
                    "no action checkpoint to resume from; run "
                    "`--train-mode frozen` first to produce an initial "
                    "action_best.pt / action_last.pt."
                )
    train_gradual_unfreeze(
        out_dir=out_dir,
        fleet_run_dir=fleet_run,
        planet_run_dir=planet_run,
        entity_run_dir=entity_run,
        resume_action_pt=resume_action_pt,
        d_model=args.d_model,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        max_planets=args.max_planets,
        max_fleets=args.max_fleets,
        max_episodes=args.max_episodes,
        num_load_workers=args.num_load_workers,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        stage_epochs=stage_epochs,
        action_only=args.action_only,
        early_stop_patience=args.early_stop_patience,
    )


if __name__ == "__main__":
    main()
