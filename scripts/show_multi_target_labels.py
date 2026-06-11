"""Show the bernoulli_select_multinomial_alloc_v2 pretrain labels.

Reads a pair cache (mmap — safe for the 19 GB Ebi cache on a 24 GB Mac),
builds the stage-1 select bits + stage-2 allocation-multinomial targets via
``agents.transformer_v2.pretrain.alloc_labels``, prints worked examples and
aggregate label statistics. Read-only; nothing is trained or written.

Usage:
    .venv/bin/python scripts/show_multi_target_labels.py \
        --cache data/datasets/_pair_cache/Ebi_T6/Ebi_T6_p64_f512_acted.pt \
        --n-snapshots 1500 --n-examples 3
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.transformer_v2.pretrain.alloc_labels import (  # noqa: E402
    ALLOC_STAT_KEYS,
    build_alloc_targets,
)
from agents.transformer_v2.pretrain.entity_encoder import (  # noqa: E402
    _owned_source_rows,
)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _print_example(snap: dict[str, torch.Tensor], tag: str) -> None:
    row_mask, target = build_alloc_targets(snap)
    pair_labels = snap["pair_labels"].bool()
    pair_valid = snap["pair_valid"].bool()
    ships = snap["pair_ships"].float()
    pos = pair_labels & pair_valid
    fired = pos & (ships > 0)
    P = pair_labels.shape[0]
    batch1 = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in snap.items()}
    owned = _owned_source_rows(batch1, pair_valid.unsqueeze(0))[0]

    print(f"\n--- example snapshot {tag} "
          f"(owned rows: {int(owned.sum())}, acting rows: {int((owned & pos.any(1)).sum())}) ---")
    shown_hold = False
    for s in range(P):
        if not bool(owned[s]):
            continue
        n_legal = int(pair_valid[s].sum())
        if bool(row_mask[s]):
            cells = fired[s].nonzero(as_tuple=False).flatten().tolist()
            hold_share = float(target[s, P])
            total = ships[s, fired[s]].sum() / max(1e-9, 1.0 - hold_share) \
                if hold_share < 1.0 else ships[s, fired[s]].sum()
            parts = [
                f"t={t:2d} {int(ships[s, t]):4d} ships ({float(target[s, t]):.3f})"
                for t in cells
            ]
            kept = int(round(float(total) * hold_share))
            print(f"  src {s:2d}  N≈{int(round(float(total))):4d}  "
                  f"SELECT: {len(cells)} fired / {n_legal} legal")
            print(f"          ALLOC target: {' | '.join(parts)} "
                  f"| HOLD {kept:4d} ships ({hold_share:.3f})")
        elif not shown_hold and n_legal > 0:
            print(f"  src {s:2d}  held — SELECT bits all-zero over {n_legal} legal "
                  f"cells (stage-1 supervision only; no alloc row)")
            shown_hold = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--n-snapshots", type=int, default=1500,
                    help="acted snapshots to scan for the aggregate stats")
    ap.add_argument("--n-examples", type=int, default=3)
    args = ap.parse_args()

    print(f"[{_ts()}] mmap-loading {args.cache} ...", flush=True)
    payload = torch.load(args.cache, map_location="cpu",
                         weights_only=False, mmap=True)
    snaps = payload["snapshots"]
    config = payload["config"]
    acted = payload.get("acted_indices")
    if acted is None:
        acted = [i for i, sn in enumerate(snaps) if bool(sn["pair_labels"].any())]
    cfg_brief = {
        k: (f"<{len(v)} entries>" if isinstance(v, (list, dict)) and len(str(v)) > 200
            else v)
        for k, v in config.items()
    }
    print(f"[{_ts()}] config: {cfg_brief}")
    print(f"[{_ts()}] snapshots: {len(snaps)} total, {len(acted)} acted", flush=True)
    if "pair_ships" not in snaps[acted[0]]:
        sys.exit("cache has no pair_ships — rebuild with the 3-tuple-edge builder")

    # Spread the sample across the whole cache (games are stored contiguously).
    stride = max(1, len(acted) // args.n_snapshots)
    sample = acted[::stride][: args.n_snapshots]

    stats: Counter[str] = Counter()
    hold_shares: list[float] = []
    cell_shares: list[float] = []
    targets_per_row: Counter[int] = Counter()
    legal_per_owned: list[int] = []
    t0 = time.time()
    for j, i in enumerate(sample):
        snap = snaps[i]
        row_mask, target = build_alloc_targets(snap, stats=stats)
        P = target.shape[0]
        if bool(row_mask.any()):
            hold_shares.extend(target[row_mask, P].tolist())
            fired_counts = (target[:, :P] > 0).sum(dim=1)[row_mask]
            for c in fired_counts.tolist():
                targets_per_row[int(c)] += 1
            cell_shares.extend(target[row_mask][:, :P][target[row_mask][:, :P] > 0].tolist())
        pv = snap["pair_valid"].bool()
        b1 = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in snap.items()}
        owned = _owned_source_rows(b1, pv.unsqueeze(0))[0]
        legal_per_owned.extend(pv[owned].sum(dim=1).tolist())
        if (j + 1) % 250 == 0:
            print(f"[{_ts()}] scanned {j + 1}/{len(sample)} snapshots "
                  f"({(j + 1) / (time.time() - t0):.0f}/s)", flush=True)

    hs = torch.tensor(hold_shares)
    cs = torch.tensor(cell_shares)
    lp = torch.tensor(legal_per_owned, dtype=torch.float32)

    print(f"\n[{_ts()}] ===== aggregate label stats ({len(sample)} snapshots) =====")
    for k in ALLOC_STAT_KEYS:
        print(f"  {k:24s} {stats[k]:>9,d}")
    print(f"\n  stage-1 SELECT (per owned row): legal cells mean {lp.mean():.1f} "
          f"p95 {lp.quantile(0.95):.0f}; "
          f"acting rows / owned rows = {stats['acted_rows']}/{stats['owned_rows']} "
          f"= {stats['acted_rows'] / max(1, stats['owned_rows']):.4f} "
          f"(positive-cell rate ≈ {stats['fired_cells'] / max(1.0, lp.sum()):.5f})")
    print("\n  stage-2 ALLOC fired-targets-per-acting-row histogram:")
    for k in sorted(targets_per_row):
        n = targets_per_row[k]
        frac = n / max(1, stats["supervised_rows"])
        print(f"    {k:2d} target(s): {n:>7,d} rows ({frac:.1%})")
    q = lambda t, p: float(t.quantile(p)) if t.numel() else float("nan")  # noqa: E731
    print(f"\n  HOLD share (the diagonal's new label): mean {hs.mean():.3f} | "
          f"p10 {q(hs, 0.10):.3f}  p50 {q(hs, 0.50):.3f}  p90 {q(hs, 0.90):.3f} | "
          f"==0: {(hs == 0).float().mean():.1%}  >0.5: {(hs > 0.5).float().mean():.1%}")
    print(f"  per-target share: mean {cs.mean():.3f} | "
          f"p10 {q(cs, 0.10):.3f}  p50 {q(cs, 0.50):.3f}  p90 {q(cs, 0.90):.3f}")

    # Worked examples: prefer snapshots with a multi-target row.
    shown = 0
    for i in sample:
        snap = snaps[i]
        row_mask, target = build_alloc_targets(snap)
        P = target.shape[0]
        if shown < args.n_examples - 1:
            multi = ((target[:, :P] > 0).sum(dim=1) >= 2) & row_mask
            if not bool(multi.any()):
                continue
        if bool(row_mask.any()):
            ep, turn = payload["keys"][i]
            _print_example(snap, f"#{i} (episode={ep} turn={turn})")
            shown += 1
            if shown >= args.n_examples:
                break
    print(f"\n[{_ts()}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
