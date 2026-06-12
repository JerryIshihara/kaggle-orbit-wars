"""Cross-entity self-attention over per-planet entity tokens.

Sits one level above ``encoder.entity_encoder.PlanetEntityEncoder``.
Each entity token already carries its own inbound-fleet picture; this
layer lets every token see every *other* token, contextualizing each
planet by the global game state (frontiers, neighbors, sector
balance).

The implementation is deliberately a thin wrapper around
``nn.TransformerEncoder``:

  * Pre-LN encoder layers (more stable than post-LN for shallow
    stacks; we run 2 layers without warmup — v2 dropped from 3→2).
  * A learned ``[CLS]`` token prepended at sequence position 0,
    never masked, that the encoder gradually fills with a global
    snapshot summary. Snapshot-level heads (winner classification,
    expert-acted, …) read it out from index 0.
  * Dynamic ``P`` (planets coming and going as comets spawn/die)
    handled natively via ``src_key_padding_mask`` — the upstream
    ``EntitySnapshotDataset`` already pads to ``max_planets``, so
    ``mask = ~entity_mask`` flows straight through.

See ``README.md`` for the design rationale and the label menu the
pretrain pipeline will train this layer against.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossEntityAttention(nn.Module):
    """Multi-layer self-attention over ``(B, T, P, d_model)`` entity tokens.

    Accepts either ``(B, T, P, d)`` (multi-step) or ``(B, P, d)`` (single
    step) inputs — single-step is treated as ``T=1``. Multi-step
    flattens the time axis into the sequence dim so attention can
    relate any (step, planet) pair to any other.

    Inputs:
      entity_tokens  (B, T, P, d) or (B, P, d)  per-planet entity tokens
      entity_mask    (B, T, P)    or (B, P)     True = real entity

    Returns:
      contextual_tokens  (B, T, P, d) or (B, P, d) — same rank as input,
                                                    post-attention values.
      global_token       (B, d)                   — CLS read-out.

    Multi-step adds a learned per-step embedding (positional encoding
    along time) so attention can distinguish ``e_{t-1}^j`` from
    ``e_t^j``. Step ``i`` (with ``i = 0`` being the oldest history
    frame and ``i = T-1`` the current turn) gets ``step_embed[i]``
    added to every planet token at that step.

    Cold-start handling: when an episode lacks ``T-1`` past turns, the
    caller passes zero tokens with ``entity_mask`` all-False for the
    missing slots; the masked softmax in the encoder ignores them.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.0,
        n_steps: int = 9,
        n_player_tokens: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_steps = n_steps
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        # Optional per-player readout tokens (transformer_v3 l2_tokens
        # player state; replaces the PlayerConsolidator). They sit in the
        # sequence as PURE READERS: an asymmetric attention mask keeps the
        # global CLS and every planet token blind to them, so enabling
        # them leaves all other outputs bit-identical — a warm-started
        # model reproduces its pre-player-token forward exactly. Slot
        # order is learner-relative (0 = self). Read out via
        # ``self.last_player_tokens`` after forward (stash, not return,
        # to keep the v2 call signature).
        self.n_player_tokens = int(n_player_tokens)
        if self.n_player_tokens > 0:
            self.player_tokens = nn.Parameter(
                torch.zeros(1, self.n_player_tokens, d_model))
            nn.init.trunc_normal_(self.player_tokens, std=0.02)
        else:
            self.player_tokens = None
        self.last_player_tokens: torch.Tensor | None = None
        # Per-relative-step positional encoding. Step ``i`` of ``n_steps``
        # gets ``step_embed[i]`` added to every planet token at that step.
        # Same role as positional embeddings, but along time rather than
        # position-in-sequence.
        self.step_embed = nn.Parameter(torch.zeros(n_steps, d_model))
        nn.init.trunc_normal_(self.step_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
            norm_first=True,
        )
        # Pre-LN disables PyTorch's nested-tensor fast path; setting this
        # explicitly avoids a noisy warning on every Colab run.
        try:
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=n_layers, enable_nested_tensor=False,
            )
        except TypeError:
            # Older PyTorch versions do not expose enable_nested_tensor.
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        entity_tokens: torch.Tensor,        # (B, T, P, d) or (B, P, d)
        entity_mask: torch.Tensor,          # (B, T, P) or (B, P) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize to (B, T, P, d) / (B, T, P).
        if entity_tokens.dim() == 3:
            entity_tokens = entity_tokens.unsqueeze(1)
            entity_mask = entity_mask.unsqueeze(1)
            squeeze_t = True
        else:
            squeeze_t = False
        B, T, P, d = entity_tokens.shape
        if d != self.d_model:
            raise ValueError(
                f"entity_tokens has d={d} but module was built for "
                f"d_model={self.d_model}"
            )
        if T > self.n_steps:
            raise ValueError(
                f"got T={T} steps but module was built for "
                f"n_steps={self.n_steps}"
            )

        # Step embeddings — pick the last T entries so that index 0 is
        # the OLDEST history frame and index T-1 is the CURRENT turn.
        # This way single-step input always gets the "current" embedding.
        step_emb = self.step_embed[-T:].view(1, T, 1, d)
        seq = entity_tokens + step_emb                                 # (B, T, P, d)

        # Flatten (T, P) into one sequence dim.
        seq = seq.reshape(B, T * P, d)
        mask_flat = entity_mask.reshape(B, T * P)                      # bool

        # Prepend CLS (never masked) and, when enabled, the per-player
        # reader tokens right after it.
        n_pt = self.n_player_tokens
        cls_tok = self.cls.expand(B, 1, d)
        if n_pt > 0:
            seq = torch.cat(
                [cls_tok, self.player_tokens.expand(B, n_pt, d), seq], dim=1,
            )                                                  # (B, 1+n_pt+T*P, d)
        else:
            seq = torch.cat([cls_tok, seq], dim=1)             # (B, 1 + T*P, d)
        lead_unmasked = torch.zeros(
            B, 1 + n_pt, dtype=torch.bool, device=entity_tokens.device,
        )
        key_padding = torch.cat([lead_unmasked, ~mask_flat], dim=1)    # True = MASKED

        attn_mask = None
        if n_pt > 0:
            # Asymmetric mask: queries OTHER than the player tokens must
            # not attend to the player-token keys (cols 1..n_pt). Player
            # rows attend everywhere. This keeps CLS + planet outputs
            # bit-identical to the no-player-token forward.
            L = seq.shape[1]
            attn_mask = torch.zeros(
                L, L, dtype=seq.dtype, device=seq.device,
            )
            attn_mask[:, 1:1 + n_pt] = float("-inf")
            attn_mask[1:1 + n_pt, 1:1 + n_pt] = 0.0
            attn_mask[1:1 + n_pt, :] = 0.0

        out = self.encoder(
            seq, mask=attn_mask, src_key_padding_mask=key_padding,
        )                                                       # (B, 1+n_pt+T*P, d)

        global_token = out[:, 0]                                        # (B, d)
        if n_pt > 0:
            self.last_player_tokens = out[:, 1:1 + n_pt]                # (B, n_pt, d)
        contextual = out[:, 1 + n_pt:].reshape(B, T, P, d)              # (B, T, P, d)
        if squeeze_t:
            contextual = contextual.squeeze(1)
        return contextual, global_token


class TemporalCrossEntityAttention(nn.Module):
    """Block-causal L2 with PER-FRAME summary tokens.

    Where :class:`CrossEntityAttention` carries a single ``[CLS]`` for the
    whole spacetime sequence and is fully bidirectional, this variant adds
    a small set of **summary tokens per frame** — one ``[CLS]`` plus
    ``n_players`` owner/player slot tokens — and enforces **block-causal**
    attention in time: a token at frame ``t`` may attend only to tokens at
    frames ``<= t``. This makes each frame's summary an honest "state as of
    frame t given history up to t", so the value/outcome heads can be
    supervised at every frame (deep, dense supervision) without leaking the
    future. Callers may pass 4 slots for real players only, or 5 slots for
    learner-relative ``[self, opp1, opp2, opp3, neutral]``.

    Per-frame layout (frame-major blocks of width ``1 + n_players + P``)::

        [CLS_0 P0_0..P3_0  e_0^0 .. e_0^{P-1}]  [CLS_1 P0_1..  ...]  ...

    Inputs:
      entity_tokens  (B, T, P, d)
      entity_mask    (B, T, P)      True = real entity
      owner_oh       (B, T, P, 5)   learner-relative owner one-hot, added as
                                    a gentle per-frame tag to entity tokens.

    Returns:
      ctx           (B, T, P, d)    per-(frame, planet) contextual tokens
      glob          (B, T, d)       per-frame CLS read-out
      player_state  (B, T, n_players, d)
                                    per-frame slot/player state

    The deployed critic / heads read the LAST frame (``[:, -1]``); the
    earlier frames are auxiliary deep-supervision targets at train time.
    """

    _N_OWNER_CHANNELS: int = 5

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.0,
        n_steps: int = 10,
        n_players: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_steps = n_steps
        self.n_players = n_players
        self.n_summary = 1 + n_players                       # CLS + players

        # Shared CLS (one learned vector, reused at every frame; frames are
        # distinguished by the per-frame step embedding added below).
        self.cls = nn.Parameter(torch.zeros(1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        # Player slot identity (learner-relative: 0=self, 1-3=opponents).
        self.player_cls = nn.Parameter(torch.zeros(n_players, d_model))
        nn.init.trunc_normal_(self.player_cls, std=0.02)
        # Per-relative-step temporal positional encoding (shared by entity
        # tokens AND summary tokens of the same frame).
        self.step_embed = nn.Parameter(torch.zeros(n_steps, d_model))
        nn.init.trunc_normal_(self.step_embed, std=0.02)

        # Owner additive tag on entity tokens (no bias → zero one-hot = no
        # signal). Scaled small so L1/L2 perception dominates at init.
        self.owner_enc = nn.Linear(self._N_OWNER_CHANNELS, d_model, bias=False)
        nn.init.trunc_normal_(self.owner_enc.weight, std=0.02)
        self.owner_scale = nn.Parameter(torch.tensor(0.1))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            batch_first=True, activation="gelu", dropout=dropout,
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=n_layers, enable_nested_tensor=False,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        entity_tokens: torch.Tensor,        # (B, T, P, d)
        entity_mask: torch.Tensor,          # (B, T, P) bool
        owner_oh: torch.Tensor,             # (B, T, P, 5)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if entity_tokens.dim() != 4:
            raise ValueError(
                "TemporalCrossEntityAttention requires (B, T, P, d) input; "
                f"got shape {tuple(entity_tokens.shape)}"
            )
        B, T, P, d = entity_tokens.shape
        if d != self.d_model:
            raise ValueError(f"d={d} != d_model={self.d_model}")
        if T > self.n_steps:
            raise ValueError(f"T={T} > n_steps={self.n_steps}")
        S = self.n_summary
        blk = S + P                                              # tokens per frame
        device = entity_tokens.device

        # Step embedding: index 0 = oldest, T-1 = current.
        step = self.step_embed[-T:].view(1, T, 1, d)            # (1, T, 1, d)

        # Entity stream: owner tag + step.
        owner_tag = self.owner_scale * self.owner_enc(owner_oh.to(entity_tokens.dtype))
        ent = entity_tokens + owner_tag + step                  # (B, T, P, d)

        # Summary stream: [CLS, P0..P3] per frame, + step.
        summ = torch.cat(
            [self.cls.view(1, d), self.player_cls], dim=0,
        ).view(1, 1, S, d).expand(B, T, S, d)                   # (B, T, S, d)
        summ = summ + step

        # Frame-major blocks: [summary(S) ‖ entities(P)] per frame.
        block = torch.cat([summ, ent], dim=2)                   # (B, T, S+P, d)
        seq = block.reshape(B, T * blk, d)                      # (B, T*(S+P), d)

        # Key padding: summary tokens never masked; entities by entity_mask.
        summ_unmasked = torch.zeros(B, T, S, dtype=torch.bool, device=device)
        pad = torch.cat([summ_unmasked, ~entity_mask], dim=2).reshape(B, T * blk)

        # Block-causal attention mask (S_total, S_total): a query at frame
        # t_q may attend to a key at frame t_k iff t_k <= t_q. True = BLOCK.
        frame_idx = torch.arange(T * blk, device=device) // blk        # (S_total,)
        attn_mask = frame_idx.unsqueeze(0) > frame_idx.unsqueeze(1)    # (Sq, Sk)

        out = self.encoder(
            seq, mask=attn_mask, src_key_padding_mask=pad,
        )                                                              # (B, T*blk, d)
        out = out.reshape(B, T, blk, d)

        glob = out[:, :, 0]                                            # (B, T, d)
        player_state = out[:, :, 1:S]                                  # (B, T, n_players, d)
        ctx = out[:, :, S:]                                            # (B, T, P, d)

        # Summary tokens are never fully masked (each frame attends to its
        # own unmasked summary tokens), but guard against NaN from any
        # fully-masked entity query rows in edge snapshots.
        glob = torch.nan_to_num(glob, nan=0.0)
        player_state = torch.nan_to_num(player_state, nan=0.0)
        return ctx, glob, player_state
