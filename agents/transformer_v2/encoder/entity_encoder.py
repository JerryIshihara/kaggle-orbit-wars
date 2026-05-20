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


# ---------- Entity encoder (cross-attention over relation-aware fleets) ----------
# ``N_STATS_PER_CELL`` is kept exported for the few legacy callers that
# import the constant; the cross-attention design no longer computes
# per-cell stats internally.
N_STATS_PER_CELL = 7


class PlanetEntityEncoder(nn.Module):
    """Cross-attention from planets to relation-aware fleets.

    Replaces the previous per-(planet, owner_slot) Bahdanau-pool +
    per-cell stats + dense fuse layout with a single cross-attention
    step. Per fleet we build a relational representation that carries
    the fleet's own token plus the tokens of its source and target
    planets::

        fleet_repr[i] = proj([fleet_tok[i]
                              ‖ source_planet_tok[i]
                              ‖ target_planet_tok[i]])

    Per planet, one cross-attention pass over the full fleet_repr set
    produces a context vector; this is concatenated with the planet
    self-token and fused through a small MLP to yield the final entity
    token::

        ctx[p]    = MultiheadAttention(query=planet_tok[p],
                                       key=value=fleet_repr,
                                       key_padding_mask=~fleet_mask)
        entity[p] = Fuse([planet_tok[p] ‖ ctx[p]])

    The cross-attention learns which fleets matter for a given planet
    directly from the source/target context baked into each fleet's
    representation — no hand-crafted owner-slot pooling, no per-cell
    stats.

    Forward inputs (batched on ``B``):

      planet_tokens      (B, P, d)   unified planet/comet self-tokens.
      fleet_tokens       (B, F, d)   from ``FleetEncoder``.
      fleet_target_idx   (B, F) long — index in ``[0, P)`` of the target
                                       planet; ``-1`` zeroes the target
                                       slot of ``fleet_repr``.
      fleet_source_idx   (B, F) long — same for source planet.
      fleet_owner_slot   accepted-but-unused (positional compat for
                         older callers built around the dense layout).
      fleet_ships_log    accepted-but-unused (same reason).
      fleet_eta_norm     accepted-but-unused (same reason).
      fleet_mask         (B, F) bool — True for real fleets.
      planet_mask        (B, P) bool — True for real planets (zeroes
                                       padded output rows).

    Output: ``entity_tokens`` (B, P, d_model).
    """

    def __init__(
        self,
        d_model: int = 256,
        *,
        n_heads: int = 8,
        dropout: float = 0.0,
        # Accepted-but-unused constructor kwargs from the dense-fusion
        # design; kept so existing call sites like ``PlanetEntityEncoder(
        # d_model=128, num_owner_slots=4)`` keep working.
        num_owner_slots: int | None = None,
        d_attn: int | None = None,
        d_hidden: int | None = None,
        ships_norm: float | None = None,
        count_norm: float | None = None,
        use_source_fusion: bool | None = None,
    ):
        super().__init__()
        del num_owner_slots, d_attn, d_hidden
        del ships_norm, count_norm, use_source_fusion
        self.d_model = d_model
        self.n_heads = n_heads
        # Per-fleet relational representation:
        #   [fleet_tok ‖ source_planet_tok ‖ target_planet_tok] → d_model.
        self.fleet_repr_proj = nn.Linear(3 * d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout,
            batch_first=True,
        )
        # Fuse the planet self-token with the attended context. A short
        # 2-Linear MLP + LayerNorm matches the depth the downstream
        # ``CrossEntityAttention`` is tuned for.
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _gather_planet_for_fleets(
        planet_tokens: torch.Tensor,        # (B, P, d)
        idx: torch.Tensor,                  # (B, F) long; -1 = unknown
    ) -> torch.Tensor:
        """Gather one planet token per fleet by index; zero the rows
        where ``idx == -1`` so fleets with unknown source / no target
        contribute zero to that slot of ``fleet_repr`` (rather than
        polluting it with whatever planet sits at index 0)."""
        B, P, d = planet_tokens.shape
        F = idx.shape[1]
        valid = (idx >= 0).unsqueeze(-1).float()                # (B, F, 1)
        safe = idx.clamp(min=0).unsqueeze(-1).expand(B, F, d)
        return torch.gather(planet_tokens, dim=1, index=safe) * valid

    def forward(
        self,
        planet_tokens: torch.Tensor,           # (B, P, d)
        fleet_tokens: torch.Tensor,            # (B, F, d)
        fleet_target_idx: torch.Tensor,        # (B, F)
        fleet_source_idx: torch.Tensor,        # (B, F)
        fleet_owner_slot: torch.Tensor | None = None,    # backward compat
        fleet_ships_log: torch.Tensor | None = None,     # backward compat
        fleet_eta_norm: torch.Tensor | None = None,      # backward compat
        fleet_mask: torch.Tensor | None = None,          # (B, F) bool
        planet_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The dense-fusion encoder consumed these three; the
        # cross-attention design folds the same information into the
        # fleet's relational representation via source/target gather.
        del fleet_owner_slot, fleet_ships_log, fleet_eta_norm
        if fleet_mask is None:
            raise ValueError("PlanetEntityEncoder.forward: fleet_mask is required")

        # Relation-aware fleet representation.
        src_tok = self._gather_planet_for_fleets(planet_tokens, fleet_source_idx)
        tgt_tok = self._gather_planet_for_fleets(planet_tokens, fleet_target_idx)
        fleet_repr = self.fleet_repr_proj(
            torch.cat([fleet_tokens, src_tok, tgt_tok], dim=-1)
        )                                                       # (B, F, d)

        # Cross-attention from each planet over the fleet_repr set.
        # PyTorch's ``key_padding_mask`` uses True = "ignore", which is
        # the inverse of ``fleet_mask`` semantics here.
        ctx, _ = self.cross_attn(
            query=planet_tokens,
            key=fleet_repr, value=fleet_repr,
            key_padding_mask=~fleet_mask,
            need_weights=False,
        )                                                       # (B, P, d)
        # Snapshots with zero valid fleets produce NaN from softmax over
        # an all-masked row; clamp to zero so downstream layers stay
        # numerically safe.
        ctx = torch.nan_to_num(ctx, nan=0.0)

        entity_tokens = self.fuse(
            torch.cat([planet_tokens, ctx], dim=-1)
        )                                                       # (B, P, d)
        if planet_mask is not None:
            entity_tokens = entity_tokens * planet_mask.unsqueeze(-1).float()
        return entity_tokens


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
