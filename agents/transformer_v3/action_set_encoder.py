"""Action-SET token: a permutation-invariant encoder over a turn's launches.

Self-supervised pretrain (NO outcome — the Q_set value head comes later) with
two joint tasks that teach the set token the JOINT structure of an action set:

  A) MASKED DENOISE — mask a random subset of the fired launches; predict the
     masked sources' targets (and frac) from the state + the un-masked launches.
     Teaches conditional set completion (coalitions, complementary targets).
  B) AUTOENCODER    — encode the FULL set into one bottleneck `set_token`, then
     reconstruct every owned source's target/HOLD from that token. Teaches a
     compressed permutation-invariant representation.

Both decode through the SAME per-source head conditioned on
(set_token ⊕ source features ⊕ glob ⊕ player) so the set token has to carry the
joint information. Launch token = h_film[s,t] ⊕ frac_embed (glob/player are
global → injected at decode, not duplicated per launch).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FracEmbed(nn.Module):
    """Fourier features of the launch fraction f∈(0,1) → d_frac (a lone scalar
    is invisible next to a 256-d token; Fourier gives it non-linear range)."""

    def __init__(self, d_frac: int = 32, n_freq: int = 8):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * torch.pi)
        self.proj = nn.Linear(2 * n_freq + 1, d_frac)

    def forward(self, f: torch.Tensor) -> torch.Tensor:        # (...,) -> (..., d_frac)
        a = f.unsqueeze(-1) * self.freqs                       # (..., n_freq)
        return self.proj(torch.cat([f.unsqueeze(-1), a.sin(), a.cos()], -1))


class ActionSetEncoder(nn.Module):
    """Launch tokens (B,K,d_launch) → set_token (B,d) + contextualized tokens.

    A learned ``[SET]`` CLS pools the set; ``mask_tok`` replaces masked launches
    (task A). ``pad_mask`` (True=pad) ignores padded slots."""

    def __init__(self, d_model: int, d_launch: int, n_layers: int = 2,
                 n_heads: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(d_launch, d_model)
        self.set_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mask_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True,
            activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)

    def forward(self, launch_tok, pad_mask, mask_sel=None):
        # launch_tok (B,K,d_launch); pad_mask (B,K) True=pad; mask_sel (B,K) True=masked
        x = self.in_proj(launch_tok)                            # (B,K,d)
        if mask_sel is not None:
            x = torch.where(mask_sel.unsqueeze(-1), self.mask_tok.to(x.dtype), x)
        B = x.size(0)
        x = torch.cat([self.set_cls.expand(B, -1, -1), x], 1)   # prepend CLS
        kp = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device),
                        pad_mask], 1)
        out = self.enc(x, src_key_padding_mask=kp)
        return out[:, 0], out[:, 1:]                            # set_token, launch_ctx


class SetReconHead(nn.Module):
    """Per-source target decoder conditioned on the set token + situation.

    For each owned source s, predict a distribution over [targets ... HOLD]
    from ``[set_token ⊕ source_joint[s] ⊕ glob ⊕ player]``. The actual pair
    logits come from a learned bilinear score against each target's L4 token,
    so reconstruction is over the real P-target grid (not a fixed vocab)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(4 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.tgt_proj = nn.Linear(d_model, d_model)
        self.hold = nn.Linear(d_model, 1)                       # per-source HOLD logit

    def forward(self, set_token, source_joint, target_joint, glob, player):
        # set_token (B,d); source/target_joint (B,P,d); glob (B,d); player (B,d)
        B, P, d = source_joint.shape
        cond = self.cond(torch.cat([
            set_token[:, None, :].expand(B, P, d),
            source_joint,
            glob[:, None, :].expand(B, P, d),
            player[:, None, :].expand(B, P, d)], -1))           # (B,P,d) per source
        tgt = self.tgt_proj(target_joint)                       # (B,P,d)
        pair = torch.einsum("bsd,btd->bst", cond, tgt)          # (B,P,P) recon logits
        hold = self.hold(cond).squeeze(-1)                      # (B,P) HOLD logit
        return pair, hold


def set_pretrain_loss(pair_logits, hold_logits, pair_labels, owned_mask,
                      pred_rows):
    """CE over [targets ... HOLD] for the rows we must predict.

    pair_logits (B,P,P), hold_logits (B,P): recon scores.
    pair_labels (B,P,P) bool: expert fired pairs. owned_mask (B,P) bool: owned
    launchable sources. pred_rows (B,P) bool: which source rows to score
    (B=all owned for the AE task; the MASKED owned rows for the denoise task).
    Label per row = the fired target, or HOLD if the source held."""
    B, P, _ = pair_logits.shape
    device = pair_logits.device
    row = pred_rows & owned_mask
    if not bool(row.any()):
        return pair_logits.sum() * 0.0, {"n": 0}
    fired = pair_labels & owned_mask.unsqueeze(-1)
    has = fired.any(-1)                                          # (B,P) launched?
    tgt = fired.float().argmax(-1)                               # (B,P) fired target idx
    # logits over [P targets ... HOLD] per row
    logits = torch.cat([pair_logits, hold_logits.unsqueeze(-1)], -1)  # (B,P,P+1)
    label = torch.where(has, tgt, torch.full_like(tgt, P))      # HOLD = index P
    rl = logits[row]                                            # (N,P+1)
    lab = label[row]                                            # (N,)
    loss = F.cross_entropy(rl, lab)
    with torch.no_grad():
        acc = (rl.argmax(-1) == lab).float().mean().item()
        lr = has[row]
        lacc = ((rl.argmax(-1) == lab) & lr).float().sum() / lr.sum().clamp(min=1)
    return loss, {"ce": float(loss.detach()), "acc": acc,
                  "launch_acc": float(lacc.detach()), "n": int(row.sum())}
