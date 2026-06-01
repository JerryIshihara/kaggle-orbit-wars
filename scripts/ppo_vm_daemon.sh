#!/usr/bin/env bash
# Launch the PPO rollout poll-daemon on the VM under nohup (survives the SSH
# session that started it). It watches GCS for new policy_vK head-deltas and
# runs rollout_worker for each. Tail ~/orbit-wars/ppo_daemon.log to watch.
#
# Required env vars:
#   RUN_ID         the PPO run id (matches Colab)
#   BASE_VERSION   the frozen-backbone id the head-deltas sit on
# Optional:
#   BUCKET=gs://orbit-wars-shipping  VM_ID=$(hostname)  WORK=~/orbit-wars
#   HISTORY_WINDOW=10  EPISODES=16  POLL_S=15
set -euo pipefail

BUCKET="${BUCKET:-gs://orbit-wars-shipping}"
WORK="${WORK:-$HOME/orbit-wars}"
VM_ID="${VM_ID:-$(hostname)}"
RUN_ID="${RUN_ID:?set RUN_ID}"
BASE_VERSION="${BASE_VERSION:?set BASE_VERSION}"
HISTORY_WINDOW="${HISTORY_WINDOW:-10}"
EPISODES="${EPISODES:-16}"
POLL_S="${POLL_S:-15}"

cd "$WORK"
export PYTHONPATH="$WORK:${PYTHONPATH:-}"
LOG="$WORK/ppo_daemon.log"
echo "[daemon-launch] run=$RUN_ID vm=$VM_ID hw=$HISTORY_WINDOW episodes=$EPISODES → $LOG"

# kill any prior daemon so re-launching never double-rolls the same run
pkill -f 'agents.transformer_v2.ppo.vm_daemon' 2>/dev/null && sleep 1 || true

nohup python3 -u -m agents.transformer_v2.ppo.vm_daemon \
  --bucket "$BUCKET" --run-id "$RUN_ID" --vm-id "$VM_ID" --poll-s "$POLL_S" \
  --ckpt "$WORK/ckpts/entity/entity_encoder_best.pt" \
  --fleet-run-dir "$WORK/ckpts/fleet" \
  --planet-run-dir "$WORK/ckpts/planet" \
  --comet-run-dir "$WORK/ckpts/comet" \
  --base-version "$BASE_VERSION" \
  --history-window "$HISTORY_WINDOW" --episodes "$EPISODES" \
  > "$LOG" 2>&1 &

echo "[daemon-launch] pid $! — tail -f $LOG"
