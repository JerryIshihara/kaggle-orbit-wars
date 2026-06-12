"""Merge a stage-A (L2-only) checkpoint into a joint-stage warm start.

    .venv/bin/python scripts/merge_l2_stage_into_joint.py \
        --base  <v3dual joint ckpt: full model incl. L3/L4>  \
        --l2    <stage-A l2_best.pt: trained dual L2 + player tokens> \
        --out   joint_warm_merged.pt

Output = base ckpt with every ``cross.*`` tensor replaced by the stage-A
version (branches, fusions, owner proj, player tokens). L3/L4/PairHead
come from the base (stage A froze PairHead — its tensors equal the
base's anyway); value/short heads are dropped (they retrain fresh in the
joint stage); consolidator is dropped (removed in v3.1). config merges
the v3.1 self-description from the stage-A ckpt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--l2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    l2 = torch.load(args.l2, map_location="cpu", weights_only=False)
    bsd = dict(base["model"])
    lsd = l2["model"]

    n_drop = 0
    for k in list(bsd):
        if k.startswith(("value_heads.", "short_heads.", "consolidator.")):
            del bsd[k]
            n_drop += 1
    n_cross = 0
    for k, v in lsd.items():
        if k.startswith("cross."):
            bsd[k] = v
            n_cross += 1
    cfg = dict(base.get("config", {}))
    cfg.update({k: v for k, v in l2.get("config", {}).items()
                if k in ("arch", "n_steps", "history_offsets",
                         "long_history_offsets", "short_history_offsets",
                         "with_consolidator", "player_state_source")})
    cfg["l2_stage_ckpt"] = str(args.l2)
    torch.save({"model": bsd, "config": cfg, "epoch": l2.get("epoch")}, args.out)
    print(f"merged: {n_cross} cross.* tensors from stage A over the base; "
          f"dropped {n_drop} value/short/consolidator tensors; "
          f"{len(bsd)} tensors total -> {args.out}")


if __name__ == "__main__":
    main()
