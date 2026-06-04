"""Add P1 value-pretrain labels to an existing (tensorized) cross-entity cache.

Additive pass — no CSV re-parse. Per snapshot computes the 5 relative P1 signals
sᵢ(t), then gathers the actual FUTURE levels sᵢ(t+K) (K∈P1_FWD_HORIZONS) and
BACKWARD deltas sᵢ(t)−sᵢ(t−K) (K∈P1_BACK_HORIZONS) by (episode,turn±K) lookup,
plus survives(t+K). Writes columns: p1_now, p1_future, p1_valid, p1_back,
valid_back, survives_future. (win/rank come from existing final_rank/player_valid.)

Usage: PYTHONPATH=. .venv/bin/python -m scripts.add_p1_value_labels IN.pt [OUT.pt]
"""
import sys
import torch

from agents.transformer_v2.pretrain.consolidator_heads import compute_current_state_labels
from agents.transformer_v2.pretrain.value_signals import (
    N_P1_SIGNALS, P1_BACK_HORIZONS, P1_FWD_HORIZONS, compute_p1_signals,
)

INPUT_KEYS = (
    "planet_features", "planet_mask", "comet_features", "is_comet",
    "fleet_features", "fleet_mask", "fleet_target_idx", "fleet_source_idx",
    "fleet_owner_slot", "fleet_ships_log", "fleet_eta_norm",
)


def main():
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp
    print(f"[p1] loading {inp}", flush=True)
    d = torch.load(inp, map_location="cpu", weights_only=False)
    assert d.get("config", {}).get("cache_format") == "tensorized", "need tensorized cache"
    cols = d["columns"]
    keys = d["keys"]
    N = next(iter(cols.values())).shape[0]
    O = 4
    NF, NB = len(P1_FWD_HORIZONS), len(P1_BACK_HORIZONS)
    print(f"[p1] {N} snapshots | fwd={P1_FWD_HORIZONS} back={P1_BACK_HORIZONS}", flush=True)

    p1 = torch.zeros(N, O, N_P1_SIGNALS)
    alive = torch.zeros(N, O)
    CH = 4096
    for s in range(0, N, CH):
        e = min(N, s + CH)
        batch = {k: cols[k][s:e] for k in INPUT_KEYS if k in cols}
        with torch.no_grad():
            p1[s:e] = compute_p1_signals(batch)
            lab = compute_current_state_labels(
                batch["planet_features"], batch["planet_mask"],
                batch["fleet_features"], batch["fleet_mask"])
            alive[s:e] = lab.alive
        if s % (CH * 8) == 0:
            print(f"  signals {e}/{N}", flush=True)

    idx = {(ep, int(t)): i for i, (ep, t) in enumerate(keys)}
    fidx = torch.full((N, NF), -1, dtype=torch.long)
    bidx = torch.full((N, NB), -1, dtype=torch.long)
    for i, (ep, t) in enumerate(keys):
        t = int(t)
        for k, K in enumerate(P1_FWD_HORIZONS):
            j = idx.get((ep, t + K))
            if j is not None:
                fidx[i, k] = j
        for k, K in enumerate(P1_BACK_HORIZONS):
            j = idx.get((ep, t - K))
            if j is not None:
                bidx[i, k] = j

    fvalid = (fidx >= 0)
    bvalid = (bidx >= 0)
    # forward levels: (N,NF,O,5) -> (N,O,5,NF), zeroed where invalid
    p1_future = p1[fidx.clamp(min=0)].permute(0, 2, 3, 1).contiguous()
    p1_future = p1_future * fvalid[:, None, None, :].float()
    surv = alive[fidx.clamp(min=0)].permute(0, 2, 1).contiguous()        # (N,O,NF)
    surv = surv * fvalid[:, None, :].float()
    # backward deltas sᵢ(t) − sᵢ(t−K): (N,NB,O,5) -> (N,O,5,NB)
    back = (p1.unsqueeze(1) - p1[bidx.clamp(min=0)]).permute(0, 2, 3, 1).contiguous()
    back = back * bvalid[:, None, None, :].float()

    cols["p1_now"] = p1
    cols["p1_future"] = p1_future
    cols["p1_valid"] = fvalid.float()
    cols["p1_back"] = back
    cols["valid_back"] = bvalid.float()
    cols["survives_future"] = surv
    d["config"]["p1_fwd_horizons"] = list(P1_FWD_HORIZONS)
    d["config"]["p1_back_horizons"] = list(P1_BACK_HORIZONS)
    d["config"]["p1_signals"] = N_P1_SIGNALS

    print(f"[p1] valid horizons: fwd mean {fvalid.float().mean(0).tolist()} "
          f"| back mean {bvalid.float().mean(0).tolist()}", flush=True)
    print(f"[p1] saving {out}", flush=True)
    torch.save(d, out)
    print("[p1] done", flush=True)


if __name__ == "__main__":
    main()
