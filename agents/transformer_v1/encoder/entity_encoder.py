"""Per-entity encoder: planet/comet self-state + query-conditioned
fleet pooling.

The fleet and planet encoders produce per-token embeddings independently
(``fleet_encoder.py``, ``planet_encoder.py``). To feed a downstream
deep-set or transformer aggregator, we want a single token per entity
(planet/comet) that already summarizes the inbound-fleet picture from
*its own perspective*.

Why query-conditioned: globally "important" fleets ≠ "important to this
planet". A 100-ship reinforcement aimed at planet 5 is critical for
planet 5 and irrelevant for planet 12. A pure self-attention over all
fleets would let global-prominence dilute the per-target signal we
actually care about.

The pool follows the Bahdanau-style additive form:

    s_i = vᵀ tanh(W_q · q + W_k · k_i)
    α_i = softmax(s_i)         over fleets that match the (planet, group) cell
    z_{j,g} = Σ α_i v_i

For each planet ``j`` we compute one pooled summary per
``(owner_slot, eta_bucket)`` group ``g``, plus three scalar stats
(count, ships-log sum, ships-log max). All summaries + stats are
concatenated with the planet's self-token and run through a small MLP
to produce the final ``(B, P, d_model)`` entity tokens.

Why preserve groups instead of one global pool per planet: the (owner,
eta_bucket) groups carry "who" and "when" structure that a single pool
would smooth out. A planet under attack from one player at h=3 looks
very different from being reinforced by an ally at h=20, even if the
total ship count is the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Pool ----------
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
        # Output for "no valid keys"; shape broadcasts to (..., d_model).
        self.empty_token = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        query: torch.Tensor,                     # (..., d)
        keys: torch.Tensor,                      # (..., N, d)
        values: torch.Tensor | None = None,      # (..., N, d), default = keys
        mask: torch.Tensor | None = None,        # (..., N) bool
    ) -> torch.Tensor:
        """Return pooled (..., d). ``query`` and ``keys/values`` must
        broadcast on all leading dims. Mask is True for valid entries.
        """
        if values is None:
            values = keys
        q = self.W_q(query).unsqueeze(-2)        # (..., 1, d_attn)
        k = self.W_k(keys)                        # (..., N, d_attn)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)   # (..., N)

        if mask is not None:
            any_valid = mask.any(dim=-1, keepdim=True)   # (..., 1)
            # masked_fill -inf for invalid, then zero out fully-empty rows
            # so softmax doesn't NaN.
            scores = scores.masked_fill(~mask, float("-inf"))
            scores = torch.where(any_valid, scores, torch.zeros_like(scores))
            attn = F.softmax(scores, dim=-1)
        else:
            any_valid = None
            attn = F.softmax(scores, dim=-1)

        pooled = (attn.unsqueeze(-1) * values).sum(dim=-2)   # (..., d)

        if any_valid is not None:
            # any_valid: (..., 1) → broadcast over feature dim
            empty = self.empty_token.expand_as(pooled)
            pooled = torch.where(any_valid, pooled, empty)
        return pooled


# ---------- Entity encoder (Plan A: grouped pooling) ----------
class PlanetEntityEncoder(nn.Module):
    """Fuse planet self-tokens with per-planet, per-group fleet summaries.

    Forward inputs (batched on ``B``):

      planet_tokens      (B, P, d)   from ``PlanetEncoder``
      fleet_tokens       (B, F, d)   from ``FleetEncoder``
      fleet_target_idx   (B, F) long — index in ``[0, P)`` of the planet
                                       this fleet is aimed at; ``-1`` =
                                       "no target" / lost-in-transit.
      fleet_owner_slot   (B, F) long — owner slot relative to the learner
                                       (``0..num_owner_slots-1``).
      fleet_eta_bucket   (B, F) long — ETA bucket id (``0..n_eta_buckets-1``).
      fleet_ships_log    (B, F) float — ``log1p(ships)`` already
                                       normalized into ~[0, 1].
      fleet_mask         (B, F) bool — True for real fleets.
      planet_mask        (B, P) bool — True for real planets (optional;
                                       only used to zero-out padded rows
                                       in the output).

    Output: ``entity_tokens`` (B, P, d_model).

    Caller responsibilities:
      * map ``fleet.target_planet_id`` to the index in the planet list,
        not the raw planet id (``build_fleet_routing(...)`` below does
        this).
      * keep ``fleet_owner_slot`` / ``fleet_eta_bucket`` clipped into
        valid ranges; out-of-range values just won't match any group
        and contribute nothing — safe but wasted.

    Memory note: the per-(planet, group) attention materializes a
    ``(B, P, n_groups, F)`` mask. For typical Orbit Wars sizes
    (P≈32, n_groups=28, F≈128, B≤16) this is small, but watch out if
    you scale ``F`` or ``num_owner_slots`` aggressively.
    """

    def __init__(
        self,
        d_model: int = 64,
        *,
        num_owner_slots: int = 4,
        n_eta_buckets: int = 7,
        d_attn: int | None = None,
        d_hidden: int | None = None,
        ships_norm: float = 1.0,        # divisor for ships_sum scalar stat
        count_norm: float = 32.0,       # divisor for count scalar stat
    ):
        super().__init__()
        self.d_model = d_model
        self.num_owner_slots = num_owner_slots
        self.n_eta_buckets = n_eta_buckets
        self.n_groups = num_owner_slots * n_eta_buckets
        self.ships_norm = ships_norm
        self.count_norm = count_norm

        self.pool = QueryConditionedPool(d_model, d_attn=d_attn)

        # Per group we emit (d_model pooled summary) + (3 scalar stats).
        n_stats = 3
        fuse_in = d_model + self.n_groups * (d_model + n_stats)
        d_hidden = d_hidden or 2 * d_model
        # Two-layer fusion MLP. The first linear is the heavy one
        # (``fuse_in → d_hidden``); a deeper stack would only help if we
        # saw the per-group structure being underexploited.
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        planet_tokens: torch.Tensor,          # (B, P, d)
        fleet_tokens: torch.Tensor,           # (B, F, d)
        fleet_target_idx: torch.Tensor,       # (B, F) long
        fleet_owner_slot: torch.Tensor,       # (B, F) long
        fleet_eta_bucket: torch.Tensor,       # (B, F) long
        fleet_ships_log: torch.Tensor,        # (B, F) float
        fleet_mask: torch.Tensor,             # (B, F) bool
        planet_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, P, d = planet_tokens.shape
        Nf = fleet_tokens.shape[1]
        device = planet_tokens.device

        # ---- Build per-(planet, group, fleet) mask ----
        # Combined group index: g = owner * n_eta + eta_bucket ∈ [0, n_groups).
        E = self.n_eta_buckets
        fleet_group = fleet_owner_slot * E + fleet_eta_bucket            # (B, F)

        # target_match[b, j, i] = (fleet i in batch b targets planet j)
        target_match = (
            fleet_target_idx.unsqueeze(1)
            == torch.arange(P, device=device).view(1, P, 1)
        )                                                                # (B, P, F)

        # group_match[b, g, i] = (fleet i is in group g)
        group_match = (
            fleet_group.unsqueeze(1)
            == torch.arange(self.n_groups, device=device).view(1, self.n_groups, 1)
        )                                                                # (B, n_g, F)

        # full_mask[b, j, g, i] = target_match & group_match & fleet_mask
        full_mask = (
            target_match.unsqueeze(2)                                    # (B, P, 1, F)
            & group_match.unsqueeze(1)                                   # (B, 1, n_g, F)
            & fleet_mask.view(B, 1, 1, Nf)
        )                                                                # (B, P, n_g, F)

        # ---- Per-(planet, group) attention pooling ----
        # We expand both query and keys to (B, P, n_groups, ...) so the
        # pool's broadcasting handles the per-cell softmax. Expansion
        # is virtual; the pool's tanh(q+k) is the actual allocation.
        query = planet_tokens.unsqueeze(2).expand(B, P, self.n_groups, d)
        keys = fleet_tokens.view(B, 1, 1, Nf, d).expand(
            B, P, self.n_groups, Nf, d,
        )
        pooled = self.pool(query, keys, mask=full_mask)                  # (B, P, n_g, d)

        # ---- Per-(planet, group) scalar stats ----
        m = full_mask.float()                                             # (B, P, n_g, F)
        count = m.sum(-1)                                                 # (B, P, n_g)
        # Use full_mask to gate ships_log; "max" over a fully-masked group
        # would return -inf otherwise.
        ships_log_b = fleet_ships_log.view(B, 1, 1, Nf)
        ships_w = m * ships_log_b
        ships_sum = ships_w.sum(-1)                                       # (B, P, n_g)
        # Replace masked positions with -inf for the max, then snap any
        # all-empty group's max back to 0.
        ships_for_max = ships_log_b.expand(B, P, self.n_groups, Nf)
        ships_for_max = ships_for_max.masked_fill(~full_mask, float("-inf"))
        ships_max = ships_for_max.amax(-1)                                # (B, P, n_g)
        ships_max = torch.where(
            count > 0, ships_max, torch.zeros_like(ships_max),
        )
        stats = torch.stack(
            [
                count / self.count_norm,
                ships_sum / self.ships_norm,
                ships_max,
            ],
            dim=-1,
        )                                                                 # (B, P, n_g, 3)

        # ---- Concat + fuse ----
        pooled_flat = pooled.reshape(B, P, self.n_groups * d)
        stats_flat = stats.reshape(B, P, self.n_groups * 3)
        fused_in = torch.cat([planet_tokens, pooled_flat, stats_flat], dim=-1)
        out = self.fuse(fused_in)                                         # (B, P, d)

        if planet_mask is not None:
            out = out * planet_mask.unsqueeze(-1).float()
        return out


# ---------- Routing helper ----------
def build_fleet_routing(
    fleet_records,
    planet_records,
    *,
    learner_slot: int,
    num_players: int,
    num_owner_slots: int = 4,
    n_eta_buckets: int = 7,
    ships_log_max: float | None = None,
    pad_to: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build the per-fleet routing tensors the entity encoder expects.

    ``fleet_records`` are the ``FleetFeaturizer`` records returned by
    ``featurize_fleets``; ``planet_records`` are the ``PlanetFeaturizer``
    records from ``featurize_planets``. We need to:
      * map each fleet's ``target_planet_id`` to its **index** in
        ``planet_records`` (planet IDs aren't always contiguous and
        comet entries make this strictly not 1:1)
      * compute the relative owner slot
      * pull ``eta_bucket``
      * compute ``log1p(ships) / SHIPS_LOG_MAX`` once

    Returns a dict of unbatched 1-D tensors plus an ``F``-length mask.
    Wrap with ``unsqueeze(0)`` (or pad+stack) to use with the encoder's
    batched API. ``pad_to`` zero-pads to a fixed F if supplied (mask
    handles the padding).
    """
    import math
    from ..featurizer.fleet_featurizer import SHIPS_LOG_MAX as DEFAULT_SHIPS_LOG_MAX
    if ships_log_max is None:
        ships_log_max = DEFAULT_SHIPS_LOG_MAX

    pid_to_idx = {p.planet_id: i for i, p in enumerate(planet_records)}
    F = len(fleet_records)
    if pad_to is not None and pad_to < F:
        raise ValueError(f"pad_to={pad_to} < n_fleets={F}")
    F_out = pad_to or F

    target_idx = torch.full((F_out,), -1, dtype=torch.long)
    owner_slot = torch.zeros(F_out, dtype=torch.long)
    eta_bucket = torch.zeros(F_out, dtype=torch.long)
    ships_log = torch.zeros(F_out, dtype=torch.float32)
    mask = torch.zeros(F_out, dtype=torch.bool)

    for i, rec in enumerate(fleet_records):
        if rec.target_planet_id is not None and rec.target_planet_id in pid_to_idx:
            target_idx[i] = pid_to_idx[rec.target_planet_id]
        if 0 <= rec.owner_id < num_players:
            owner_slot[i] = (rec.owner_id - learner_slot) % num_players
        # else: leave 0 — fleet without owner is degenerate; mask handles it.
        eta_bucket[i] = max(0, min(n_eta_buckets - 1, rec.eta_bucket))
        ships_log[i] = math.log1p(max(0, rec.ships)) / ships_log_max
        mask[i] = True

    return {
        "fleet_target_idx": target_idx,
        "fleet_owner_slot": owner_slot,
        "fleet_eta_bucket": eta_bucket,
        "fleet_ships_log": ships_log,
        "fleet_mask": mask,
    }
