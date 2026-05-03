#!/usr/bin/env bash
# Pack code, data, and weights into three tarballs for upload to GCS.
# See docs/GPU_TRAINING.md for the full GPU workflow.
#
# Output:  /tmp/orbit-pack/{code,data,weights}.tgz
# Prints:  ready-to-paste gsutil cp commands.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/orbit-pack}"
BUCKET="${BUCKET:-gs://orbit-wars-shipping}"

mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"

# ---- code: agents/, scripts/, utils/, requirements.txt, app/, run.py ----
# Excludes __pycache__, ckpts, datasets — these go in their own tar.
echo "[pack] code.tgz ..."
tar czf "$OUT_DIR/code.tgz" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  agents scripts utils app requirements.txt run.py

# ---- data: every CSV dataset the pretrain scripts read ----
echo "[pack] data.tgz ..."
tar czf "$OUT_DIR/data.tgz" \
  --exclude='__pycache__' \
  data/datasets/fleet \
  data/datasets/planet \
  data/datasets/entity \
  data/datasets/cross_entity \
  data/datasets/action

# ---- weights: only the BEST checkpoints from each prior run ----
# These are frozen for cross-entity training. Best-only because last-
# epoch / log files aren't needed at training time. Optionally include
# the latest (or explicitly specified) cross-entity checkpoint so the
# later gradual-unfreeze run can resume from it.
echo "[pack] weights.tgz ..."
WEIGHT_FILES=()
INCLUDE_CROSS_ENTITY="${INCLUDE_CROSS_ENTITY:-0}"
CROSS_ENTITY_RESUME="${CROSS_ENTITY_RESUME:-}"
# Map encoder kind → expected ckpt filename. Skips probe-* / experiment
# run dirs that don't carry the canonical encoder ckpt.
for sub in fleet planet entity; do
  case "$sub" in
    fleet)  TARGET="fleet_encoder_best.pt" ;;
    planet) TARGET="planet_encoder_best.pt" ;;
    entity) TARGET="entity_encoder_best.pt" ;;
  esac
  RUN_DIR=""
  # Iterate dirs newest→oldest, pick the first that actually has the ckpt.
  for d in $(ls -td data/runs/$sub/*/ 2>/dev/null); do
    if [ -f "${d}${TARGET}" ]; then
      RUN_DIR="$d"
      break
    fi
  done
  if [ -z "$RUN_DIR" ]; then
    echo "  warning: no $sub run dir contains $TARGET" >&2
    continue
  fi
  WEIGHT_FILES+=("${RUN_DIR}${TARGET}")
  echo "  $sub: ${RUN_DIR}${TARGET}"
done

if [ "$INCLUDE_CROSS_ENTITY" = "1" ]; then
  TARGET=""
  if [ -n "$CROSS_ENTITY_RESUME" ]; then
    if [ ! -f "$CROSS_ENTITY_RESUME" ]; then
      echo "  warning: CROSS_ENTITY_RESUME does not exist: $CROSS_ENTITY_RESUME" >&2
    else
      TARGET="$CROSS_ENTITY_RESUME"
    fi
  else
    for d in $(ls -td data/runs/cross_entity/*/ 2>/dev/null); do
      if [ -f "${d}cross_entity_best.pt" ]; then
        TARGET="${d}cross_entity_best.pt"
        break
      fi
    done
  fi
  if [ -n "$TARGET" ]; then
    WEIGHT_FILES+=("$TARGET")
    echo "  cross_entity resume: $TARGET"
  else
    echo "  warning: INCLUDE_CROSS_ENTITY=1 but no cross_entity_best.pt was found" >&2
  fi
fi

if [ "${#WEIGHT_FILES[@]}" -eq 0 ]; then
  echo "  no best checkpoints found; skipping weights.tgz" >&2
else
  tar czf "$OUT_DIR/weights.tgz" "${WEIGHT_FILES[@]}"
fi

echo
echo "[pack] sizes:"
du -h "$OUT_DIR"/*.tgz 2>/dev/null

echo
echo "[pack] upload commands:"
for tgz in "$OUT_DIR"/*.tgz; do
  echo "  gsutil cp $tgz $BUCKET/$(basename "$tgz")"
done
