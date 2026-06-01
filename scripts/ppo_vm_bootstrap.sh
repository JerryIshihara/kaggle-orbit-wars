#!/usr/bin/env bash
# One-time bootstrap for a PPO rollout VM (Debian/Ubuntu, GCP c4/n2/e2).
#
# Stages the FROZEN base once: repo code, L0 encoders, and the supervised entity
# ckpt. Per-iteration the daemon then pulls only the <1 MB head-delta.
# Run once on a fresh VM (or as the instance startup-script).
#
# Parameterize via env vars (all have sensible defaults):
#   BUCKET        gs://orbit-wars-shipping
#   ENTITY_CKPT   gs URL of the supervised entity (L1+actor) ckpt           [required]
#   WORK          ~/orbit-wars
#
# The L0 encoders come from weights.tgz (pack_for_gpu.sh layout). After this,
# launch either the legacy GCS poll daemon (scripts/ppo_vm_daemon.sh) or the
# current Firestore STATE actor daemon (ppo_actor_daemon).
set -euo pipefail

BUCKET="${BUCKET:-gs://orbit-wars-shipping}"
WORK="${WORK:-$HOME/orbit-wars}"
ENTITY_CKPT="${ENTITY_CKPT:?set ENTITY_CKPT=gs://.../entity_encoder_best.pt}"

echo "[bootstrap] swapfile (e2-medium has 4 GB; swap guards T=10 bursts)"
if [ ! -f /swapfile ]; then
  sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
  sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
fi

echo "[bootstrap] python deps (Debian 12 is PEP 668 externally-managed -> break-system-packages)"
sudo apt-get update -y -q
sudo apt-get install -y -q python3-pip
export PIP_BREAK_SYSTEM_PACKAGES=1
python3 -m pip install -q --break-system-packages --upgrade pip
python3 -m pip install -q --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -q --break-system-packages \
  kaggle_environments numpy psutil google-cloud-firestore google-cloud-storage

echo "[bootstrap] stage repo + weights + base ckpts into $WORK"
mkdir -p "$WORK" && cd "$WORK"
gcloud storage cp "$BUCKET/entity/code.tgz" .    && tar xzf code.tgz
gcloud storage cp "$BUCKET/entity/weights.tgz" . && tar xzf weights.tgz   # flat ./{planet,fleet,comet}_*_best.pt
mkdir -p ckpts/entity ckpts/fleet ckpts/planet ckpts/comet
# weights.tgz unpacks the three L0 encoder *_best.pt at the cwd (flat layout).
[ -f planet_encoder_best.pt ] && cp planet_encoder_best.pt ckpts/planet/
[ -f fleet_encoder_best.pt ]  && cp fleet_encoder_best.pt  ckpts/fleet/
[ -f comet_past_best.pt ]     && cp comet_past_best.pt     ckpts/comet/
gcloud storage cp "$ENTITY_CKPT" ckpts/entity/entity_encoder_best.pt

echo "[bootstrap] sanity: import + encoder check"
export PYTHONPATH="$WORK:${PYTHONPATH:-}"
python3 - <<'PY'
import torch
from agents.transformer_v2.ppo import shards   # import-only sanity
c = torch.load("ckpts/entity/entity_encoder_best.pt", map_location="cpu", weights_only=False)
print("  entity d_model=", c.get('config',{}).get('d_model'))
print("[bootstrap] OK")
PY
echo "[bootstrap] DONE. PYTHONPATH=$WORK"
echo "[bootstrap] current STATE protocol: run python3 -u -m agents.transformer_v2.ppo.ppo_actor_daemon ..."
echo "[bootstrap] legacy poll protocol:   run scripts/ppo_vm_daemon.sh"
