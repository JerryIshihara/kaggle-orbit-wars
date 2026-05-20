"""Pair-score scorer with a shared 2-layer trunk + L1-conditioned FiLM + 2 heads.

After the L1-L4 entity stack, this head produces TWO predictions per
snapshot from the same ``(B, P, P, trunk_hidden)`` broadcast tensor:

  * ``pair_logits  (B, P, P)``  per-cell source→target compatibility
  * ``pair_frac    (B, P, P)``  fraction of source's ships sent to t,
                                 raw logit (caller sigmoids in loss);
                                 supervised on positive cells only.

The earlier auxiliary heads (``source_act`` / ``target_aim`` /
``glob_act``) were dropped: at inference the runner only consumes
``pair_logits`` + ``pair_frac``, and the per-planet / snapshot heads
fought the joint optimizer for capacity without informing actions.

FiLM conditioning between trunk and heads
=========================================

Empirically the trunk output ``h[s, t]`` carries the role-aware
representation from L4 + the role-agnostic context from L2, but it has
no direct view of the **tactical state** around the pair — e.g. "this
source already has outgoing fleets" or "this target is being contested
by an enemy inbound". That information lives in L1's per-planet tokens,
which fuse the planet's own L0 representation with the fleet relation
context (inbound/outbound fleets, ship counts, ETAs).

The FiLM block uses the L1 source + L1 target tokens plus an explicit
27-way pair-type embedding to produce per-cell affine modulation
parameters ``γ, β`` and apply them to ``h``:

    h_film[s, t] = h[s, t] + α · (γ[s, t] · h[s, t] + β[s, t])

Initialization is **identity**: ``γ`` and ``β`` start at 0, so
``h_film = h`` exactly at step 0. The residual scale ``α`` starts at
1.0; because the FiLM output is zero-initialized, the branch is still a
no-op initially, but gradients flow into the conditioner immediately.

Trunk + FiLM + head structure (default ``d_pair = d_model = 256``):

    pair_feat  (B, P, P, 6·d_pair = 1536)
       │ Linear(1536 → trunk_hidden), GELU
       │ Linear(trunk_hidden → trunk_hidden), GELU
       ▼
    h          (B, P, P, trunk_hidden)
                                                ↑ FiLM conditioner ↑
                                                [L1_src ‖ L1_tgt ‖
                                                 pair_type_emb]
                                                Linear → GELU → Linear → (γ, β)
       │ h_film = h + α · (γ · h + β)            α: learnable scalar, init=1
       ▼
    h_film     (B, P, P, trunk_hidden)
       │
       ├─ pair_head        : Linear(trunk_hidden → 1) → pair_logits (B, P, P)
       └─ pair_frac_head   : Linear(trunk_hidden → 1) → pair_frac   (B, P, P)

Backward compatibility with the 5-head ckpts
============================================

``load_state_dict`` filters out the legacy ``source_act_head`` /
``target_aim_head`` / ``glob_act_head`` keys and accepts missing or
shape-incompatible FiLM keys by falling through to the identity-init
defaults. Old ckpts therefore load cleanly into the new 2-head + FiLM
module — the trunk + pair heads keep their trained weights, and FiLM
behaves as a no-op at load time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: FiLM pair-type category count:
#:
#:   source physical type       ∈ {0=static, 1=orbital, 2=comet}
#:   target owner relation      ∈ {0=enemy, 1=neutral, 2=own}
#:   target physical type       ∈ {0=static, 1=orbital, 2=comet}
#:
#: Category id = ``source_type * 9 + target_relation * 3 + target_type``.
PAIR_TYPE_NUM_CLASSES = 27
PAIR_TYPE_EMBED_DIM = 32


class PairHead(nn.Module):
    """Trunk + L1-conditioned FiLM + 2 heads.

    Forward inputs:
      source_joint  (B, P, d_model)   from :class:`JointRoleAttention`.
      target_joint  (B, P, d_model)   from :class:`JointRoleAttention`.
      ctx_now       (B, P, d_model)   from :class:`CrossEntityAttention`.
      l1_tokens     (B, P, d_model)   from :class:`PlanetEntityEncoder`
                                       (the current-step L1 output).
      is_comet      (B, P) bool       per-planet comet flag from L0;
                                       used only as fallback when
                                       ``pair_type_ids`` is omitted.
      pair_type_ids (B, P, P) long    27-way source/target category.
      pair_valid    (B, P, P) bool    optional; kept for API symmetry.

    Returns ``dict[str, Tensor]``:
      ``pair_logits  (B, P, P)``,
      ``pair_frac    (B, P, P)`` raw logit (caller sigmoids in loss).
    """

    HEAD_NAMES: tuple[str, ...] = ("pair_logits", "pair_frac")

    def __init__(
        self,
        d_model: int = 256,
        *,
        d_pair: int | None = None,
        trunk_hidden: int = 256,
        conditioner_hidden: int = 256,
        conditioner_n_layers: int = 1,
        head_n_layers: int = 1,
        pair_type_num_classes: int = PAIR_TYPE_NUM_CLASSES,
        pair_type_embed_dim: int = PAIR_TYPE_EMBED_DIM,
        c_scalars: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_pair is None:
            d_pair = d_model
        if conditioner_n_layers < 1:
            raise ValueError(
                f"conditioner_n_layers must be >= 1; got {conditioner_n_layers}"
            )
        if head_n_layers < 1:
            raise ValueError(
                f"head_n_layers must be >= 1; got {head_n_layers}"
            )
        self.d_model = d_model
        self.d_pair = d_pair
        self.trunk_hidden = trunk_hidden
        self.conditioner_hidden = conditioner_hidden
        self.conditioner_n_layers = int(conditioner_n_layers)
        self.head_n_layers = int(head_n_layers)
        self.pair_type_num_classes = pair_type_num_classes
        self.pair_type_embed_dim = pair_type_embed_dim
        self.c_scalars = c_scalars

        # Three independent projections — source, target, ctx — same as
        # before. At d_pair == d_model these are square (no width change).
        self.src_proj = nn.Linear(d_model, d_pair)
        self.tgt_proj = nn.Linear(d_model, d_pair)
        self.ctx_proj = nn.Linear(d_model, d_pair)

        feat_dim = 6 * d_pair + c_scalars
        # 2-layer shared trunk.
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, trunk_hidden),
            nn.GELU(),
            *([nn.Dropout(dropout)] if dropout > 0 else []),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.GELU(),
            *([nn.Dropout(dropout)] if dropout > 0 else []),
        )

        # ---- FiLM conditioner -------------------------------------------------
        # Inputs per (s, t): L1_src ‖ L1_tgt ‖ pair_type_emb.
        # ``pair_type_emb`` is the 27-way source/target category described
        # by ``PAIR_TYPE_NUM_CLASSES`` above. Depth is parametrized by
        # ``conditioner_n_layers`` (number of hidden layers):
        #
        #   n=1 (default):  Linear(cond_in → H) → GELU → Linear(H → 2·trunk_h)
        #   n=2:            …+ extra Linear(H → H) → GELU before final
        #   n=k:            k hidden layers then final 2·trunk_h projection
        #
        # The final Linear is zero-init so γ=β=0 at start regardless of depth.
        self.pair_type_embed = nn.Embedding(
            pair_type_num_classes, pair_type_embed_dim,
        )
        cond_in_dim = 2 * d_model + pair_type_embed_dim
        film_layers: list[nn.Module] = [
            nn.Linear(cond_in_dim, conditioner_hidden), nn.GELU(),
        ]
        if dropout > 0:
            film_layers.append(nn.Dropout(dropout))
        for _ in range(self.conditioner_n_layers - 1):
            film_layers += [
                nn.Linear(conditioner_hidden, conditioner_hidden), nn.GELU(),
            ]
            if dropout > 0:
                film_layers.append(nn.Dropout(dropout))
        film_layers.append(nn.Linear(conditioner_hidden, 2 * trunk_hidden))  # γ + β
        self.film_proj = nn.Sequential(*film_layers)

        # Identity-init the FiLM output: final Linear weights + biases at 0
        # so γ ~ 0, β ~ 0 at the very first forward.
        nn.init.zeros_(self.film_proj[-1].weight)
        nn.init.zeros_(self.film_proj[-1].bias)
        # Learnable scalar scale. Keep it non-zero: if both alpha and the
        # zero-initialized FiLM output start at 0, the conditioner receives
        # zero gradient forever. With alpha=1 and γ=β=0, the first forward is
        # still exactly identity, while gradients flow into film_proj.
        self.film_alpha = nn.Parameter(torch.ones(1))
        self.register_load_state_dict_post_hook(
            self._upgrade_dead_film_alpha_after_load,
        )

        # ---- Two heads --------------------------------------------------------
        # ``head_n_layers`` controls depth of each per-head decoder MLP:
        #   n=1 (default): single Linear(trunk_hidden → 1) — backward-compat
        #   n=k: (k-1) × [Linear(H → H) → GELU (→ Dropout)] + Linear(H → 1)
        # The final Linear keeps the same small-std init as the n=1 path so
        # initial pair_logits are roughly chance.
        self.pair_head = self._build_head(trunk_hidden, head_n_layers, dropout)
        self.pair_frac_head = self._build_head(trunk_hidden, head_n_layers, dropout)
        for head in (self.pair_head, self.pair_frac_head):
            final = head if isinstance(head, nn.Linear) else head[-1]
            nn.init.zeros_(final.bias)
            nn.init.normal_(final.weight, std=0.02)

    @staticmethod
    def _build_head(
        hidden: int, n_layers: int, dropout: float,
    ) -> nn.Module:
        if n_layers == 1:
            return nn.Linear(hidden, 1)
        layers: list[nn.Module] = []
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        return nn.Sequential(*layers)

    def forward(
        self,
        source_joint: torch.Tensor,        # (B, P, d_model)
        target_joint: torch.Tensor,        # (B, P, d_model)
        ctx_now: torch.Tensor,             # (B, P, d_model)
        l1_tokens: torch.Tensor,           # (B, P, d_model)
        is_comet: torch.Tensor,            # (B, P) bool
        pair_type_ids: torch.Tensor | None = None,  # (B, P, P) long
        pair_valid: torch.Tensor | None = None,    # noqa: ARG002 (API symmetry)
        pair_scalars: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, P, d = source_joint.shape
        if d != self.d_model:
            raise ValueError(
                f"source_joint d={d} but PairHead built for d_model={self.d_model}"
            )

        # ---- Trunk ----
        src_r = self.src_proj(source_joint)        # (B, P, d_pair)
        tgt_r = self.tgt_proj(target_joint)
        ctx_r = self.ctx_proj(ctx_now)

        src_b  = src_r.unsqueeze(2).expand(B, P, P, self.d_pair)
        tgt_b  = tgt_r.unsqueeze(1).expand(B, P, P, self.d_pair)
        ctxs_b = ctx_r.unsqueeze(2).expand(B, P, P, self.d_pair)
        ctxt_b = ctx_r.unsqueeze(1).expand(B, P, P, self.d_pair)
        st     = src_b * tgt_b
        ctx_st = ctxs_b * ctxt_b

        feats: list[torch.Tensor] = [src_b, ctxs_b, tgt_b, ctxt_b, st, ctx_st]
        if pair_scalars is not None:
            if pair_scalars.shape[-1] != self.c_scalars:
                raise ValueError(
                    f"pair_scalars last-dim={pair_scalars.shape[-1]} "
                    f"but PairHead built for c_scalars={self.c_scalars}"
                )
            feats.append(pair_scalars)
        feat = torch.cat(feats, dim=-1)            # (B, P, P, 6·d_pair + c_sc)
        h = self.trunk(feat)                       # (B, P, P, trunk_hidden)

        # ---- FiLM conditioner ----
        # Broadcast L1 tokens + 27-way pair-type embedding across (P, P).
        l1_src = l1_tokens.unsqueeze(2).expand(B, P, P, self.d_model)
        l1_tgt = l1_tokens.unsqueeze(1).expand(B, P, P, self.d_model)
        if pair_type_ids is None:
            # Back-compat fallback for older tests/callers: only comet-ness
            # is known, so non-comets collapse to "static" and targets use
            # the enemy relation bucket.
            phys = torch.where(
                is_comet.to(torch.bool),
                torch.full_like(is_comet, 2, dtype=torch.long),
                torch.zeros_like(is_comet, dtype=torch.long),
            )
            pair_type_ids = (
                phys.unsqueeze(2) * 9
                + phys.unsqueeze(1)
            )
        if pair_type_ids.shape != (B, P, P):
            raise ValueError(
                f"pair_type_ids shape={tuple(pair_type_ids.shape)}; "
                f"expected {(B, P, P)}"
            )
        type_emb = self.pair_type_embed(
            pair_type_ids.to(torch.long).clamp(
                min=0, max=self.pair_type_num_classes - 1,
            )
        )                                                         # (B,P,P,type_d)
        cond_in = torch.cat([l1_src, l1_tgt, type_emb], dim=-1)   # (B,P,P, 2d+type_d)
        film = self.film_proj(cond_in)                                 # (B,P,P, 2·th)
        gamma, beta = film.chunk(2, dim=-1)

        # Identity-init residual: h_film = h + α · (γ · h + β).
        # At γ=β=0 → h_film == h exactly. Alpha starts non-zero so the
        # conditioner is trainable from the first backward pass.
        h_film = h + self.film_alpha * (gamma * h + beta)

        pair_logits = self.pair_head(h_film).squeeze(-1)       # (B, P, P)
        pair_frac   = self.pair_frac_head(h_film).squeeze(-1)  # (B, P, P) raw

        return {"pair_logits": pair_logits, "pair_frac": pair_frac}

    # ------------------------------------------------------------------ #
    # Backward-compat state_dict loader                                  #
    # ------------------------------------------------------------------ #
    def _prune_legacy_state_dict(self, state_dict, prefix: str = "") -> None:
        """Remove obsolete or shape-incompatible keys before load.

        This is needed both for direct ``PairHead.load_state_dict`` and for
        parent ``EntityPretrainModel.load_state_dict`` calls: PyTorch does
        not call child ``load_state_dict`` overrides during recursive load.
        """
        legacy_prefixes = (
            "source_act_head", "target_aim_head", "glob_act_head",
        )
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            local = key[len(prefix):]
            if any(local.startswith(p) for p in legacy_prefixes):
                del state_dict[key]

        def drop_film_branch() -> None:
            for key in list(state_dict.keys()):
                if (
                    key.startswith(prefix + "film_proj.")
                    or key == prefix + "film_alpha"
                    or key == prefix + "pair_type_embed.weight"
                ):
                    del state_dict[key]

        # The pair-type embedding changes film_proj.0 input width
        # versus the older is_comet-bit conditioner. If that old tensor is
        # present, drop the whole FiLM branch so it reverts to identity-init
        # instead of mixing random new input weights with learned old output
        # weights.
        first_weight_key = prefix + "film_proj.0.weight"
        if (
            first_weight_key in state_dict
            and tuple(state_dict[first_weight_key].shape)
            != tuple(self.film_proj[0].weight.shape)
        ):
            drop_film_branch()

        emb_key = prefix + "pair_type_embed.weight"
        if (
            emb_key in state_dict
            and tuple(state_dict[emb_key].shape)
            != tuple(self.pair_type_embed.weight.shape)
        ):
            # 18-way → 27-way keeps film_proj.0 at the same width because
            # the embedding dim is unchanged, but the category semantics and
            # table shape changed. Loading the old FiLM MLP with a fresh
            # random embedding would produce arbitrary modulation, so reset
            # the whole FiLM branch to identity-init.
            drop_film_branch()

        has_film_key = any(k.startswith(prefix + "film_proj.") for k in state_dict)
        if has_film_key and emb_key not in state_dict:
            # Defensive: a partially-saved FiLM ckpt without the type table
            # is not semantically loadable.
            drop_film_branch()

        # Depth mismatch — when ``--conditioner-n-layers`` differs between
        # the saved ckpt and the new instantiation, the film_proj Sequential
        # has a different number of Linear sub-modules. Count the saved
        # Linear weights vs the current module's; if they differ, drop the
        # whole branch so PairHead initializes identity FiLM at the new
        # depth instead of half-loading mismatched weights.
        saved_film_layer_count = sum(
            1 for k in state_dict
            if k.startswith(prefix + "film_proj.") and k.endswith(".weight")
        )
        current_film_layer_count = sum(
            1 for _ in self.film_proj if isinstance(_, nn.Linear)
        )
        if saved_film_layer_count and saved_film_layer_count != current_film_layer_count:
            drop_film_branch()

        # Head depth mismatch — the per-head decoders are nn.Linear when
        # ``head_n_layers == 1`` and nn.Sequential otherwise. Going from
        # n=1 (saved key ``pair_head.weight``) to n>=2 (saved keys
        # ``pair_head.0.weight``, ``pair_head.2.weight`` … ) the saved key
        # names don't match the current module's parameter names, so
        # PyTorch's strict=False would leave the new Sequential at random
        # init anyway. Strip the mismatched keys to keep the diagnostic
        # output clean (no "unexpected" / "missing" noise for known cases).
        for head_name in ("pair_head", "pair_frac_head"):
            head_module = getattr(self, head_name, None)
            if head_module is None:
                continue
            saved_keys = [
                k for k in state_dict
                if k.startswith(prefix + head_name + ".")
                or k == prefix + head_name + ".weight"
                or k == prefix + head_name + ".bias"
            ]
            if not saved_keys:
                continue
            saved_is_sequential = any(
                k.startswith(prefix + head_name + ".") and k != prefix + head_name + ".weight"
                and k != prefix + head_name + ".bias"
                for k in saved_keys
            )
            current_is_sequential = isinstance(head_module, nn.Sequential)
            if saved_is_sequential != current_is_sequential:
                for k in saved_keys:
                    del state_dict[k]
                continue
            if saved_is_sequential and current_is_sequential:
                saved_linear_count = sum(
                    1 for k in saved_keys if k.endswith(".weight")
                )
                current_linear_count = sum(
                    1 for m in head_module if isinstance(m, nn.Linear)
                )
                if saved_linear_count != current_linear_count:
                    for k in saved_keys:
                        del state_dict[k]

    def _upgrade_dead_film_alpha_after_load(self, module, incompatible_keys) -> None:
        """Repair checkpoints saved with the dead ``alpha=0, γ=β=0`` init.

        A FiLM checkpoint produced before this fix is numerically
        identical after setting alpha to 1 when the final FiLM projection
        is still all-zero: ``γ=β=0`` keeps ``h_film == h``. The change only
        restores gradient flow if that checkpoint is resumed for training.
        """
        del incompatible_keys
        if module is not self:
            return
        with torch.no_grad():
            final = self.film_proj[-1]
            if (
                bool(torch.all(self.film_alpha == 0))
                and bool(torch.all(final.weight == 0))
                and bool(torch.all(final.bias == 0))
            ):
                self.film_alpha.fill_(1.0)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Drop legacy aux-head keys + tolerate missing FiLM keys.

        Old 5-head ckpts (``source_act_head`` / ``target_aim_head`` /
        ``glob_act_head``) load cleanly into the new 2-head + FiLM
        module: the legacy aux head weights are discarded silently, the
        trunk + pair heads keep their trained values, and the new FiLM
        keys (``film_proj.*``, ``film_alpha``) fall back to the
        identity-init defaults set in :meth:`__init__`.
        """
        filtered = dict(state_dict)
        self._prune_legacy_state_dict(filtered, prefix="")
        # If the incoming dict has no FiLM keys at all, the caller is loading
        # a pre-FiLM ckpt. Use strict=False so PyTorch reports "missing keys"
        # for film_proj / film_alpha without raising — our init handles it.
        has_film = any(k.startswith("film_") for k in filtered)
        has_pair_type = "pair_type_embed.weight" in filtered
        effective_strict = bool(strict) and has_film and has_pair_type
        return super().load_state_dict(filtered, strict=effective_strict)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._prune_legacy_state_dict(state_dict, prefix=prefix)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
