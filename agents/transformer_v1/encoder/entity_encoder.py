"""Per-entity encoder: planet/comet self-state + query-conditioned
fleet pooling, **Plan B** layout.

Earlier (Plan A) we partitioned inbound fleets per planet into
``num_owner_slots × n_eta_buckets`` groups, attention-pooled each cell,
and concatenated everything with the planet self-token. That gave
``num_groups = 28`` cells per planet and a fuse-input of ~1900 dims —
sparse and expensive.

This file implements the simpler refinement:

  1. **Group only by ``(planet, player_slot)``** — drops the ETA bucket
     dimension. ``num_groups`` per planet falls 28 → 4 (≈7× compute and
     parameter reduction in the attention block).

  2. **Compensate the lost ETA discreteness with continuous timing
     stats** — each ``(planet, player)`` cell emits ``min``, ``mean``,
     ``max`` and ``std`` of normalized ETAs. The same temporal
     information PCA-style, without forcing every horizon into its own
     bucket.

  3. **Inject source / origin context into fleet keys** — each fleet
     token is concatenated with its *source planet's* token (a free
     output of the planet encoder we already run) and projected back
     to ``d_model`` before pooling. Lets the attention attend to "this
     fleet came from a heavily-defended ally" vs "from an enemy
     stronghold," which encodes strategic causality the raw fleet
     features can't express.

  4. **Action-conditioned decoder** — :class:`LaunchOutcomeDecoder`
     (separate module, can be trained on top of frozen entity tokens)
     consumes a ``(source_token, target_token, ships)`` triple and
     predicts launch outcomes (capture probability, signed-log capture
     margin, ETA). Useful for the agent: at decision time, score every
     candidate ``(src, tgt)`` pair by feeding both planets' entity
     tokens through this head.

The pool itself (:class:`QueryConditionedPool`) is unchanged from Plan
A — additive attention with masked softmax and a learned
``empty_token`` for fully-masked rows.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Pool (unchanged from Plan A) ----------
class QueryConditionedPool(nn.Module):
    """Bahdanau-style additive attention pooling over a set of keys.

    Score:    s_i = vᵀ tanh(W_q q + W_k k_i)
    Weights:  α_i = softmax(s_i, mask)
    Output:   z = Σ α_i v_i

    Shapes: ``query`` is broadcast over the leading dims of ``keys``.
    Empty groups (no valid keys for some leading row) get a learned
    ``empty_token`` rather than NaN, so downstream layers stay numerically
    safe and the network can learn what "no fleets" should look like.
    """

    def __init__(self, d_model: int, d_attn: int | None = None):
        super().__init__()
        d_attn = d_attn or d_model
        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.v = nn.Linear(d_attn, 1, bias=False)
        self.empty_token = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        query: torch.Tensor,                     # (..., d)
        keys: torch.Tensor,                      # (..., N, d)
        values: torch.Tensor | None = None,      # (..., N, d)
        mask: torch.Tensor | None = None,        # (..., N) bool
    ) -> torch.Tensor:
        if values is None:
            values = keys
        q = self.W_q(query).unsqueeze(-2)
        k = self.W_k(keys)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)

        if mask is not None:
            any_valid = mask.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~mask, float("-inf"))
            scores = torch.where(any_valid, scores, torch.zeros_like(scores))
            attn = F.softmax(scores, dim=-1)
        else:
            any_valid = None
            attn = F.softmax(scores, dim=-1)

        pooled = (attn.unsqueeze(-1) * values).sum(dim=-2)

        if any_valid is not None:
            empty = self.empty_token.expand_as(pooled)
            pooled = torch.where(any_valid, pooled, empty)
        return pooled


# ---------- Entity encoder (Plan B) ----------
N_STATS_PER_CELL = 7   # count, ships_sum_log, ships_max_log, eta_min/mean/max/std


class PlanetEntityEncoder(nn.Module):
    """Fuse planet self-tokens with per-(planet, player) inbound-fleet
    summaries, optionally conditioned on each fleet's source planet.

    Forward inputs (batched on ``B``):

      planet_tokens      (B, P, d)   from ``PlanetEncoder``.
      fleet_tokens       (B, F, d)   from ``FleetEncoder``.
      fleet_target_idx   (B, F) long — index in ``[0, P)`` of the planet
                                       this fleet is aimed at; ``-1`` =
                                       no target / lost-in-transit.
      fleet_source_idx   (B, F) long — index in ``[0, P)`` of the
                                       launching planet, or ``-1`` if
                                       unknown (drops the source-fusion
                                       contribution for that fleet).
      fleet_owner_slot   (B, F) long — owner slot relative to the learner
                                       (``0..num_owner_slots-1``).
      fleet_ships_log    (B, F) float — ``log1p(ships)`` already
                                        normalized into ~[0, 1].
      fleet_eta_norm     (B, F) float — normalized ETA (typically
                                        ``min(1, eta/50)``); used both
                                        for stats and as a sanity check
                                        on per-cell timing.
      fleet_mask         (B, F) bool — True for real fleets.
      planet_mask        (B, P) bool — True for real planets (zeroes
                                       padded output rows).

    Output: ``entity_tokens`` (B, P, d_model).
    """

    def __init__(
        self,
        d_model: int = 64,
        *,
        num_owner_slots: int = 4,
        d_attn: int | None = None,
        d_hidden: int | None = None,
        ships_norm: float = 1.0,
        count_norm: float = 32.0,
        use_source_fusion: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_owner_slots = num_owner_slots
        self.ships_norm = ships_norm
        self.count_norm = count_norm
        self.use_source_fusion = use_source_fusion

        self.pool = QueryConditionedPool(d_model, d_attn=d_attn)

        # Source-token fusion projects ``[fleet_token ‖ source_token]``
        # back to d_model so the pool sees a single key per fleet. When
        # disabled (or source unknown) the fleet token passes through
        # unchanged via a residual.
        if use_source_fusion:
            self.source_fuse = nn.Linear(2 * d_model, d_model)

        fuse_in = d_model + num_owner_slots * (d_model + N_STATS_PER_CELL)
        d_hidden = d_hidden or 2 * d_model
        # Two-layer fusion MLP. Same shape as Plan A but ~7× narrower
        # input because of the group reduction.
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def _fuse_source(
        self,
        fleet_tokens: torch.Tensor,        # (B, F, d)
        planet_tokens: torch.Tensor,        # (B, P, d)
        fleet_source_idx: torch.Tensor,     # (B, F) long
    ) -> torch.Tensor:
        """Concat each fleet token with its source-planet token and
        project back to d_model. Fleets with ``source_idx == -1`` get a
        zero source contribution (i.e., ``source_fuse([fleet, 0])``)."""
        B, P, d = planet_tokens.shape
        F = fleet_tokens.shape[1]
        valid = fleet_source_idx >= 0                                    # (B, F)
        # Clamp -1 to 0 for the gather (we mask the result back to 0).
        idx = fleet_source_idx.clamp(min=0).unsqueeze(-1).expand(B, F, d)
        src_tokens = torch.gather(planet_tokens, dim=1, index=idx)        # (B, F, d)
        src_tokens = src_tokens * valid.unsqueeze(-1).float()
        return self.source_fuse(torch.cat([fleet_tokens, src_tokens], dim=-1))

    def forward(
        self,
        planet_tokens: torch.Tensor,           # (B, P, d)
        fleet_tokens: torch.Tensor,            # (B, F, d)
        fleet_target_idx: torch.Tensor,        # (B, F)
        fleet_source_idx: torch.Tensor,        # (B, F)
        fleet_owner_slot: torch.Tensor,        # (B, F)
        fleet_ships_log: torch.Tensor,         # (B, F)
        fleet_eta_norm: torch.Tensor,          # (B, F)
        fleet_mask: torch.Tensor,              # (B, F) bool
        planet_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, P, d = planet_tokens.shape
        Nf = fleet_tokens.shape[1]
        device = planet_tokens.device
        K = self.num_owner_slots

        # ---- Source/origin fusion (optional) ----
        if self.use_source_fusion:
            fleet_keys = self._fuse_source(fleet_tokens, planet_tokens, fleet_source_idx)
        else:
            fleet_keys = fleet_tokens

        # ---- Per-(planet, player) full mask ----
        target_match = (
            fleet_target_idx.unsqueeze(1)
            == torch.arange(P, device=device).view(1, P, 1)
        )                                                                # (B, P, F)
        owner_match = (
            fleet_owner_slot.unsqueeze(1)
            == torch.arange(K, device=device).view(1, K, 1)
        )                                                                # (B, K, F)
        full_mask = (
            target_match.unsqueeze(2)                                    # (B, P, 1, F)
            & owner_match.unsqueeze(1)                                   # (B, 1, K, F)
            & fleet_mask.view(B, 1, 1, Nf)
        )                                                                # (B, P, K, F)

        # ---- Pool per cell ----
        query = planet_tokens.unsqueeze(2).expand(B, P, K, d)
        keys = fleet_keys.view(B, 1, 1, Nf, d).expand(B, P, K, Nf, d)
        pooled = self.pool(query, keys, mask=full_mask)                  # (B, P, K, d)

        # ---- Per-cell stats: count, ships(sum/max), eta(min/mean/max/std) ----
        m = full_mask.float()                                             # (B, P, K, F)
        count = m.sum(-1)                                                 # (B, P, K)

        ships_b = fleet_ships_log.view(B, 1, 1, Nf)
        ships_sum = (m * ships_b).sum(-1)
        ships_for_max = ships_b.expand(B, P, K, Nf).masked_fill(~full_mask, float("-inf"))
        ships_max = ships_for_max.amax(-1)
        ships_max = torch.where(count > 0, ships_max, torch.zeros_like(ships_max))

        eta_b = fleet_eta_norm.view(B, 1, 1, Nf)
        eta_for_min = eta_b.expand(B, P, K, Nf).masked_fill(~full_mask, float("inf"))
        eta_min = eta_for_min.amin(-1)
        # Empty cell ⇒ "no fleet" marker = 1.0 (saturated normalized ETA).
        eta_min = torch.where(count > 0, eta_min, torch.ones_like(eta_min))
        eta_for_max = eta_b.expand(B, P, K, Nf).masked_fill(~full_mask, float("-inf"))
        eta_max = eta_for_max.amax(-1)
        eta_max = torch.where(count > 0, eta_max, torch.zeros_like(eta_max))
        eta_sum = (m * eta_b).sum(-1)
        eta_mean = eta_sum / count.clamp(min=1.0)
        # Variance with safe division; clamp negatives that arise from
        # fp drift (sum-of-squares minus square-of-sum is fine here).
        eta_var = (m * (eta_b - eta_mean.unsqueeze(-1)).pow(2)).sum(-1) / count.clamp(min=1.0)
        eta_std = eta_var.clamp(min=0.0).sqrt()

        stats = torch.stack(
            [
                count / self.count_norm,
                ships_sum / self.ships_norm,
                ships_max,
                eta_min,
                eta_mean,
                eta_max,
                eta_std,
            ],
            dim=-1,
        )                                                                 # (B, P, K, 7)

        # ---- Concat + fuse ----
        pooled_flat = pooled.reshape(B, P, K * d)
        stats_flat = stats.reshape(B, P, K * N_STATS_PER_CELL)
        fused_in = torch.cat([planet_tokens, pooled_flat, stats_flat], dim=-1)
        out = self.fuse(fused_in)

        if planet_mask is not None:
            out = out * planet_mask.unsqueeze(-1).float()
        return out


# ---------- Action-conditioned decoder ----------
class LaunchOutcomeDecoder(nn.Module):
    """Predict launch outcome from a ``(source, target)`` entity pair.

    Inputs:
      source_token  (B, d) — entity-encoder output for the launching planet
      target_token  (B, d) — entity-encoder output for the target planet
      ships_log     (B,)   — ``log1p(ships) / SHIPS_LOG_MAX`` of the
                             proposed fleet size

    Outputs (dict, all shape ``(B,)``):
      capture_logit       — pre-sigmoid binary head; >0 ⇒ predicts
                            ownership flips on arrival.
      capture_margin_log  — predicted ``signed_log1p(ships - garrison)``
                            at arrival; positive ⇒ over-budget capture.
      eta_norm            — predicted ``eta / ETA_NORM`` for the fleet.

    Trained downstream of pretraining, on labels we already store
    (``will_change_owner_on_arrival``, ``capture_margin_signed_log``,
    derived ETA). The whole point of this decoder is to give the agent
    a fast "score this candidate launch" head: at decision time, run
    the planet/entity encoders once per turn, then evaluate every
    ``(src, tgt)`` pair through this decoder for ~free.
    """

    def __init__(self, d_model: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head_capture = nn.Linear(hidden, 1)
        self.head_margin = nn.Linear(hidden, 1)
        self.head_eta = nn.Linear(hidden, 1)

    def forward(
        self,
        source_token: torch.Tensor,
        target_token: torch.Tensor,
        ships_log: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat(
            [source_token, target_token, ships_log.unsqueeze(-1)],
            dim=-1,
        )
        h = self.net(x)
        return {
            "capture_logit": self.head_capture(h).squeeze(-1),
            "capture_margin_log": self.head_margin(h).squeeze(-1),
            "eta_norm": torch.sigmoid(self.head_eta(h)).squeeze(-1),
        }


# ---------- Routing helper ----------
def build_fleet_routing(
    fleet_records,
    planet_records,
    *,
    learner_slot: int,
    num_players: int,
    num_owner_slots: int = 4,
    ships_log_max: float | None = None,
    eta_norm: float = 50.0,
    pad_to: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build the routing tensors :class:`PlanetEntityEncoder` expects.

    Per fleet:
      * ``fleet_target_idx`` — index in ``planet_records`` of the
        first-hit planet (or -1 if no target / planet not in list).
      * ``fleet_source_idx``  — index in ``planet_records`` of the
        launching planet (-1 if not present, e.g., destroyed).
      * ``fleet_owner_slot``  — ``(owner - learner) % num_players``.
      * ``fleet_eta_bucket``  — kept for backward compat with Plan A
        callers; no longer required by Plan B encoder.
      * ``fleet_eta_norm``    — ``min(1, eta/eta_norm)``, ``1.0`` if no
        target (saturated marker).
      * ``fleet_ships_log``   — ``log1p(ships) / SHIPS_LOG_MAX``.
      * ``fleet_mask``        — True for real fleets.

    All tensors are 1-D over fleets; wrap with ``unsqueeze(0)`` (or
    pad+stack) to use the encoder's batched API.
    """
    import math
    from ..featurizer.fleet_featurizer import (
        SHIPS_LOG_MAX as DEFAULT_SHIPS_LOG_MAX,
    )
    if ships_log_max is None:
        ships_log_max = DEFAULT_SHIPS_LOG_MAX

    pid_to_idx = {p.planet_id: i for i, p in enumerate(planet_records)}
    F = len(fleet_records)
    if pad_to is not None and pad_to < F:
        raise ValueError(f"pad_to={pad_to} < n_fleets={F}")
    F_out = pad_to or F

    target_idx = torch.full((F_out,), -1, dtype=torch.long)
    source_idx = torch.full((F_out,), -1, dtype=torch.long)
    owner_slot = torch.zeros(F_out, dtype=torch.long)
    eta_bucket = torch.zeros(F_out, dtype=torch.long)
    eta_norm_t = torch.ones(F_out, dtype=torch.float32)   # 1.0 = saturated
    ships_log = torch.zeros(F_out, dtype=torch.float32)
    mask = torch.zeros(F_out, dtype=torch.bool)

    for i, rec in enumerate(fleet_records):
        if rec.target_planet_id is not None and rec.target_planet_id in pid_to_idx:
            target_idx[i] = pid_to_idx[rec.target_planet_id]
        if rec.source_planet_id is not None and rec.source_planet_id in pid_to_idx:
            source_idx[i] = pid_to_idx[rec.source_planet_id]
        if 0 <= rec.owner_id < num_players:
            owner_slot[i] = (rec.owner_id - learner_slot) % num_players
        eta_bucket[i] = max(0, min(num_owner_slots * 7 - 1, rec.eta_bucket))
        if rec.eta_to_target is not None:
            eta_norm_t[i] = min(1.0, float(rec.eta_to_target) / eta_norm)
        ships_log[i] = math.log1p(max(0, rec.ships)) / ships_log_max
        mask[i] = True

    return {
        "fleet_target_idx": target_idx,
        "fleet_source_idx": source_idx,
        "fleet_owner_slot": owner_slot,
        "fleet_eta_bucket": eta_bucket,
        "fleet_eta_norm": eta_norm_t,
        "fleet_ships_log": ships_log,
        "fleet_mask": mask,
    }
