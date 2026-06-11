#!/bin/bash
# 20/20 gate: deploy-eval a PPO policy_vK.pt (pulled from the pod) against the
# frozen v2 baseline, 10 stratified panel seeds x both seats = 20 games.
#
#   bash scripts/gate_beat_baseline.sh <policy_vK.pt> [num_seeds]
#
# The policy plays through the production runner with its contract auto-mode
# (v2 -> alloc_softmax, config inherited from the baseline ckpt); the baseline
# plays the same way. Both deterministic -> a clean 20/20 target read.
set -euo pipefail
POLICY="$1"
SEEDS="${2:-10}"
BASE=data/runs/joint/joint_mt_alloc_d256_T10_head3_20260610-164152/joint_best.pt
TMP=/tmp/ow_gate
mkdir -p "$TMP"
RUNNER_CKPT="$TMP/runner_$(basename "${POLICY%.pt}").pt"

.venv/bin/python - "$POLICY" "$BASE" "$RUNNER_CKPT" <<'EOF'
import sys
sys.path.insert(0, '.')
from pathlib import Path
import torch
from scripts.eval_ppo_threshold_vs_physical import build_runner_ckpt
build_runner_ckpt(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
# Propagate the POLICY's training contract into the runner ckpt config so the
# runner auto-selects the right deploy decode (v3 -> topk_self; v2 ->
# alloc_softmax). The base config would otherwise impose ITS contract.
pol = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
contract = (pol.get("ppo_trial") or {}).get("action_contract")
if contract:
    rc = torch.load(sys.argv[3], map_location="cpu", weights_only=False)
    rc["config"]["action_contract"] = contract
    if (pol.get("ppo_trial") or {}).get("select_k_max"):
        rc["config"]["select_k_max"] = int(pol["ppo_trial"]["select_k_max"])
    torch.save(rc, sys.argv[3])
    print(f"[gate] runner ckpt contract <- {contract}")
EOF

.venv/bin/python scripts/eval_runner_ckpt_vs_agent.py \
  --ckpt "$RUNNER_CKPT" \
  --opponent-ckpt "$BASE" \
  --num-seeds "$SEEDS"
