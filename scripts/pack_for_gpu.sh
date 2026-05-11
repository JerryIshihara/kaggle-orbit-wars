#!/usr/bin/env bash
# Pack code, data, weights, and optional pair-score assets for upload to GCS.
# See docs/GPU_TRAINING.md for the full GPU workflow.
#
# Output:  /tmp/orbit-pack/{code,data,weights,pair_score_assets}.tgz
# Prints:  ready-to-paste gsutil cp commands unless UPLOAD=1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/orbit-pack}"
BUCKET="${BUCKET:-gs://orbit-wars-shipping}"
UPLOAD="${UPLOAD:-0}"
INCLUDE_PAIR_SCORE_ASSETS="${INCLUDE_PAIR_SCORE_ASSETS:-1}"
PAIR_SCORE_PLAYER="${PAIR_SCORE_PLAYER:-kovi}"
PAIR_SCORE_ACTION_CKPT="${PAIR_SCORE_ACTION_CKPT:-}"
OUTPUT_TGZS=()

mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"

require_csvs() {
  local dir="$1"
  local pattern="$2"
  local label="$3"
  local count
  count="$(find "$dir" -maxdepth 1 -name "$pattern" 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$count" = "0" ]; then
    echo "[pack] error: no $label CSVs found under $dir" >&2
    echo "       rebuild with scripts/build_encoder_dataset.py before packing." >&2
    exit 1
  fi
  echo "  $label CSVs: $count"
}

# ---- code: agents/, scripts/, utils/, requirements.txt, app/, run.py ----
# Excludes __pycache__, ckpts, datasets — these go in their own tar.
echo "[pack] code.tgz ..."
tar czf "$OUT_DIR/code.tgz" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  agents scripts utils app requirements.txt run.py
OUTPUT_TGZS+=("$OUT_DIR/code.tgz")

# ---- data: every CSV dataset the pretrain scripts read ----
echo "[pack] data.tgz ..."
require_csvs data/datasets/fleet 'fleet_*.csv' fleet
require_csvs data/datasets/planet 'planet_*.csv' planet
require_csvs data/datasets/entity 'entity_*.csv' entity
require_csvs data/datasets/cross_entity 'cross_entity_*.csv' cross_entity
require_csvs data/datasets/action 'action_*.csv' action
tar czf "$OUT_DIR/data.tgz" \
  --exclude='__pycache__' \
  data/datasets/fleet \
  data/datasets/planet \
  data/datasets/entity \
  data/datasets/cross_entity \
  data/datasets/action
OUTPUT_TGZS+=("$OUT_DIR/data.tgz")

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
  OUTPUT_TGZS+=("$OUT_DIR/weights.tgz")
fi

# ---- pair-score assets: action-stage encoder ckpt + replay owner map ----
# Pair-score filters action CSVs by replay owner (for example player=kovi),
# so it needs the player's replay subtree in addition to data.tgz. It also
# loads encoder/cross weights from an action-stage checkpoint.
if [ "$INCLUDE_PAIR_SCORE_ASSETS" = "1" ]; then
  echo "[pack] pair_score_assets.tgz ..."
  if [ -z "$PAIR_SCORE_ACTION_CKPT" ]; then
    for d in $(ls -td data/runs/action/*/ 2>/dev/null); do
      if [ -f "${d}action_best.pt" ]; then
        PAIR_SCORE_ACTION_CKPT="${d}action_best.pt"
        break
      fi
      if [ -f "${d}action_last.pt" ]; then
        PAIR_SCORE_ACTION_CKPT="${d}action_last.pt"
        break
      fi
    done
  fi
  if [ ! -f "$PAIR_SCORE_ACTION_CKPT" ]; then
    echo "  error: no action checkpoint found for pair_score_assets.tgz" >&2
    echo "         set PAIR_SCORE_ACTION_CKPT=data/runs/action/<run>/action_best.pt" >&2
    exit 1
  fi
  if [ ! -d "data/replays/$PAIR_SCORE_PLAYER" ]; then
    echo "  error: no replay directory for PAIR_SCORE_PLAYER=$PAIR_SCORE_PLAYER" >&2
    echo "         expected data/replays/$PAIR_SCORE_PLAYER" >&2
    exit 1
  fi
  player_replays="$(
    find "data/replays/$PAIR_SCORE_PLAYER" -maxdepth 1 -name '*.json.gz' \
      | wc -l | tr -d ' '
  )"
  player_action_matches="$(
    comm -12 \
      <(find "data/replays/$PAIR_SCORE_PLAYER" -maxdepth 1 -name '*.json.gz' \
          -exec basename {} .json.gz \; | sort) \
      <(find data/datasets/action -maxdepth 1 -name 'action_*.csv' \
          -exec basename {} .csv \; | sed 's/^action_//' | sort) \
      | wc -l | tr -d ' '
  )"
  if [ "$player_replays" = "0" ] || [ "$player_action_matches" = "0" ]; then
    echo "  error: pair-score player has no matching action CSVs" >&2
    echo "         player=$PAIR_SCORE_PLAYER replays=$player_replays action_matches=$player_action_matches" >&2
    echo "         rebuild/upload data.tgz after generating data/datasets/action." >&2
    exit 1
  fi
  echo "  action ckpt: $PAIR_SCORE_ACTION_CKPT"
  echo "  player replays: $PAIR_SCORE_PLAYER ($player_replays files, $player_action_matches action CSV matches)"
  tar czf "$OUT_DIR/pair_score_assets.tgz" \
    "$PAIR_SCORE_ACTION_CKPT" \
    "data/replays/$PAIR_SCORE_PLAYER"
  OUTPUT_TGZS+=("$OUT_DIR/pair_score_assets.tgz")
fi

echo
echo "[pack] sizes:"
du -h "${OUTPUT_TGZS[@]}" 2>/dev/null

echo
if [ "$UPLOAD" = "1" ]; then
  echo "[pack] uploading to $BUCKET ..."
  for tgz in "${OUTPUT_TGZS[@]}"; do
    gsutil cp "$tgz" "$BUCKET/$(basename "$tgz")"
  done
else
  echo "[pack] upload commands:"
  for tgz in "${OUTPUT_TGZS[@]}"; do
    echo "  gsutil cp $tgz $BUCKET/$(basename "$tgz")"
  done
  echo
  echo "[pack] or rerun with UPLOAD=1 to copy them now."
fi
