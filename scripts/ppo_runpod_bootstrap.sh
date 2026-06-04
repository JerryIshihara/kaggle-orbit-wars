#!/bin/bash
# Bootstrap a RunPod GPU pod (RunPod Pytorch image, RTX 3090, 24GB VRAM; tested
# on both 32-vCPU and 256-vCPU pods) and run a single-machine PPO run via
# train_local_trial.
#
# NOTE ON DEVICE: the default here is GPU inference-server rollout. CPU workers
# own env stepping + T=10 featurization only; the parent process owns the single
# CUDA PPOActorCritic instance and batches ready seat observations from all
# workers into GPU forwards. The old CPU fork pool is available only by setting
# `ROLLOUT=pool DEVICE=cpu ALLOW_CPU_FORWARD_ROLLOUT=1`.
#
# Inputs expected in ~/ow (push via ssh-pipe from the controller, or gcloud-pull
# if the pod is authed): code.tgz, weights.tgz (L0 encoders), joint_best.pt.
# Pass run args after the script name, e.g.:
#   trial:   GCS_OUT=gs://orbit-wars-shipping/ppo/runpod_trial \
#            bash ppo_runpod_bootstrap.sh --iters 2 --games 64 --unfreeze head
#   10-iter: bash ppo_runpod_bootstrap.sh --iters 10 --games 64 --unfreeze L2 \
#              --out-dir ~/ow/ppo_run10
set -e
OW="${OW:-$HOME/ow}"
cd "$OW"

# Optional self-logging. When OUT_DIR is set, the log file is created before
# dependency install, artifact extraction, preflight checks, or rollout launch.
# The controller can start `tail -n +1 -F "$OUT_DIR/nohup.out"` immediately
# after spawning this script and see the entire run from the first boot line.
if [ -z "${BOOTSTRAP_LOG_ACTIVE:-}" ] && [ -n "${OUT_DIR:-}" ]; then
  mkdir -p "$OUT_DIR"
  export BOOTSTRAP_LOG_ACTIVE=1
  exec >>"$OUT_DIR/nohup.out" 2>&1
fi

echo "=== ppo_runpod_bootstrap: $(date -u) host=$(hostname) nproc=$(nproc) ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
python --version

# Deps — start early and overlap with artifact download/extraction. The
# --break-system-packages path is needed on newer RunPod/Pytorch images; the
# fallback keeps older/local images working.
install_deps() {
  python -m pip install -q --break-system-packages --ignore-installed \
    "kaggle_environments==1.28.1" psutil numpy google-cloud-storage google-cloud-firestore 2>&1 | tail -1 || \
  python -m pip install -q --ignore-installed \
    "kaggle_environments==1.28.1" psutil numpy google-cloud-storage google-cloud-firestore
}
if [ "${SKIP_DEPS:-0}" = "1" ]; then
  echo "[boot] dependency install skipped (SKIP_DEPS=1)"
  DEPS_PID=""
else
  echo "[boot] dependency install started in background"
  install_deps &
  DEPS_PID=$!
fi

# Optional GCS pull (only if the pod has gcloud + auth; otherwise inputs are
# expected pre-staged in ~/ow via ssh-pipe).
BUCKET="${BUCKET:-gs://orbit-wars-shipping/entity}"
CKPT_RUN="${CKPT_RUN:-joint_actval_d256_T10_20260603-115028}"
if command -v gcloud >/dev/null 2>&1 && [ "${GCS_PULL:-0}" = "1" ]; then
  echo "[boot] pulling inputs from $BUCKET in parallel"
  PULL_PIDS=()
  gcloud storage cp "$BUCKET/code.tgz" . &
  PULL_PIDS+=($!)
  gcloud storage cp "$BUCKET/weights.tgz" . &
  PULL_PIDS+=($!)
  gcloud storage cp "$BUCKET/runs/$CKPT_RUN/joint_best.pt" . &
  PULL_PIDS+=($!)
  for pid in "${PULL_PIDS[@]}"; do
    if ! wait "$pid"; then
      echo "[boot] WARNING one GCS pull failed; continuing if file is already staged"
    fi
  done
fi

EXTRACT_PIDS=()
if [ -f code.tgz ]; then
  (tar xzf code.tgz && echo "[boot] extracted code.tgz") &
  EXTRACT_PIDS+=($!)
fi
if [ -f weights.tgz ]; then
  (tar xzf weights.tgz && echo "[boot] extracted weights.tgz") &
  EXTRACT_PIDS+=($!)
fi
for pid in "${EXTRACT_PIDS[@]}"; do
  wait "$pid"
done
echo "[boot] waiting for dependency install"
if [ -n "$DEPS_PID" ]; then
  wait "$DEPS_PID"
fi
echo "[boot] dependency install done"

# Stage L0 encoders into the run-dir layout train_local_trial expects.
mkdir -p ckpts/planet ckpts/fleet ckpts/comet
cp -f planet_encoder_best.pt ckpts/planet/ 2>/dev/null || true
cp -f fleet_encoder_best.pt  ckpts/fleet/  2>/dev/null || true
cp -f comet_past_best.pt     ckpts/comet/  2>/dev/null || true

for f in joint_best.pt ckpts/planet/planet_encoder_best.pt ckpts/fleet/fleet_encoder_best.pt ckpts/comet/comet_past_best.pt; do
  [ -f "$f" ] || { echo "[boot] FATAL missing $f"; exit 3; }
done

python - <<'PY'
import inspect
import torch
from agents.transformer_v2.ppo.actor_critic import PPOActorCritic
from agents.transformer_v2.ppo import train_local_trial

assert "reward_decomp" in inspect.signature(PPOActorCritic).parameters, (
    "stale code.tgz: PPOActorCritic lacks reward_decomp")
assert hasattr(train_local_trial, "apply_phi_shaping"), (
    "stale code.tgz: train_local_trial lacks apply_phi_shaping")

s = torch.load("joint_best.pt", map_location="cpu", weights_only=False)
m = s.get("model", s)
n_vh = sum(k.startswith("value_heads.") for k in m)
assert n_vh, "joint_best.pt has no value_heads.* tensors; --reward-decomp cannot run"
print(f"[boot] reward-decomp preflight OK: value_heads tensors={n_vh}")
PY

export PYTHONPATH="$OW" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -n 1048576 2>/dev/null || true

ROLLOUT="${ROLLOUT:-infserver}"
DEVICE="${DEVICE:-cuda}"
POST_PROCS="${POST_PROCS:-4}"
[ "$POST_PROCS" -lt 0 ] && POST_PROCS=0
MAX_PLANETS="${MAX_PLANETS:-64}"
MAX_FLEETS="${MAX_FLEETS:-512}"
NPROC="$(nproc)"
if [ -n "${PROCS:-}" ]; then
  DEFAULT_PROCS="$PROCS"
elif [ "$MAX_FLEETS" -ge 512 ] && [ "$NPROC" -ge 128 ]; then
  # 512 fleet slots materially increase rollout forward memory. Use a safer
  # high-CPU default on 24GB GPUs; pass PROCS explicitly for throughput tests.
  DEFAULT_PROCS=48
elif [ "$NPROC" -ge 128 ]; then
  # On large CPU pods, use enough env workers to fill the single CUDA inference
  # batches without spawning hundreds of mostly idle Python envs.
  DEFAULT_PROCS=200
else
  DEFAULT_PROCS="$(( NPROC - POST_PROCS ))"
fi
[ "$DEFAULT_PROCS" -lt 1 ] && DEFAULT_PROCS=1
if [ -n "${MINIBATCH_SIZE:-}" ]; then
  DEFAULT_MINIBATCH_SIZE="$MINIBATCH_SIZE"
elif [ "$MAX_FLEETS" -ge 512 ]; then
  # 512 fleet slots quadruple the per-minibatch fleet tensor footprint versus
  # the memory-safe 128-fleet debug run. Keep update VRAM bounded; rollout
  # packing itself stays on CPU in train_local_trial.
  DEFAULT_MINIBATCH_SIZE=64
else
  DEFAULT_MINIBATCH_SIZE=256
fi
INFSERVER_STALL_TIMEOUT_S="${INFSERVER_STALL_TIMEOUT_S:-180}"
INFSERVER_SPOOL_DIR="${INFSERVER_SPOOL_DIR:-/dev/shm/ow_infserver_spool}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "$DEVICE" = "cuda" ]; then
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "DEVICE=cuda requested but CUDA is unavailable"
print("[boot] CUDA OK:", torch.cuda.get_device_name(0))
PY
fi

ARGS=(
  --ckpt "$OW/joint_best.pt"
  --planet-run-dir "$OW/ckpts/planet"
  --fleet-run-dir "$OW/ckpts/fleet"
  --comet-run-dir "$OW/ckpts/comet"
  --history-window 10
  --rollout "$ROLLOUT"
  --procs "$DEFAULT_PROCS"
  --post-procs "$POST_PROCS"
  --reward-decomp
  --max-planets "$MAX_PLANETS"
  --max-fleets "$MAX_FLEETS"
  --device "$DEVICE"
  --minibatch-size "$DEFAULT_MINIBATCH_SIZE"
  --infserver-stall-timeout-s "$INFSERVER_STALL_TIMEOUT_S"
)
if [ -n "${INFSERVER_SPOOL_DIR:-}" ]; then
  ARGS+=(--infserver-spool-dir "$INFSERVER_SPOOL_DIR")
fi
if [ -n "${GCS_OUT:-}" ]; then
  ARGS+=(--gcs-out "$GCS_OUT")
fi
if [ -n "${OUT_DIR:-}" ]; then
  ARGS+=(--out-dir "$OUT_DIR")
fi
if [ -n "${STREAM_URL:-}" ]; then
  ARGS+=(--stream-url "$STREAM_URL")
fi
if [ -n "${RUN_ID:-}" ]; then
  ARGS+=(--run-id "$RUN_ID")
fi
if [ "${ALLOW_CPU_FORWARD_ROLLOUT:-0}" = "1" ]; then
  ARGS+=(--allow-cpu-forward-rollout)
fi
if [ "${PACKED_ROLLOUT_ARTIFACTS_ONLY:-1}" = "1" ]; then
  ARGS+=(--packed-rollout-artifacts-only)
fi
if [ "${DELETE_LOCAL_ROLLOUTS_AFTER_UPLOAD:-1}" = "1" ]; then
  ARGS+=(--delete-local-rollouts-after-upload)
fi
if [ "${DELETE_INFSERVER_SPOOL_AFTER_POST:-1}" = "1" ]; then
  ARGS+=(--delete-infserver-spool-after-post)
fi

echo "[boot] launching train_local_trial rollout=$ROLLOUT device=$DEVICE procs=$DEFAULT_PROCS post_procs=$POST_PROCS max_fleets=$MAX_FLEETS minibatch_size=$DEFAULT_MINIBATCH_SIZE ..."
exec python -u -m agents.transformer_v2.ppo.train_local_trial \
  "${ARGS[@]}" \
  "$@"
