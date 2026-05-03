#!/usr/bin/env bash
# Pull training-run artifacts from the GPU VM back to local.
#
# Usage:
#   bash scripts/sync_from_gpu.sh logs    # JSON logs only (small, fast)
#   bash scripts/sync_from_gpu.sh full    # everything incl. .pt ckpts
#
# Defaults assume a Compute Engine VM named "orbit-wars-gpu" in
# us-central1-b training the cross-entity pretrain. Override via env:
#
#   VM_NAME=my-vm ZONE=us-east1-b RUN_KIND=entity bash scripts/sync_from_gpu.sh full

set -euo pipefail

VM_NAME="${VM_NAME:-orbit-wars-gpu}"
ZONE="${ZONE:-us-central1-b}"
RUN_KIND="${RUN_KIND:-cross_entity}"   # subdir of data/runs/
MODE="${1:-logs}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="$REPO_ROOT/data/runs/$RUN_KIND"
mkdir -p "$LOCAL_DIR"

# Find the latest run dir on the VM.
LATEST=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command \
  "ls -td orbit-wars/data/runs/$RUN_KIND/*/ 2>/dev/null | head -1" \
  | tr -d '\r' | sed 's:/*$::')

if [ -z "$LATEST" ]; then
  echo "no run dir found on $VM_NAME under data/runs/$RUN_KIND/" >&2
  exit 1
fi

RUN_NAME=$(basename "$LATEST")
LOCAL_RUN_DIR="$LOCAL_DIR/$RUN_NAME"
mkdir -p "$LOCAL_RUN_DIR"

case "$MODE" in
  logs)
    PATTERNS=("*.json")
    ;;
  full)
    PATTERNS=("*.json" "*.pt")
    ;;
  *)
    echo "usage: $0 [logs|full]" >&2
    exit 1
    ;;
esac

echo "[sync] $VM_NAME:~/$LATEST → $LOCAL_RUN_DIR"
for p in "${PATTERNS[@]}"; do
  gcloud compute scp \
    --zone="$ZONE" \
    --recurse \
    "$VM_NAME:$LATEST/$p" \
    "$LOCAL_RUN_DIR/" 2>/dev/null || echo "  (no files matched $p — skipping)"
done

ls -lh "$LOCAL_RUN_DIR"
