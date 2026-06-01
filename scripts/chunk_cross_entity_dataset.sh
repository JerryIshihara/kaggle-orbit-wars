#!/usr/bin/env bash
# Build independent cross-entity CSV dataset shards for faster Colab staging.
#
# Default object names match the L2 Colab notebook:
#   cross_entity_dataset_shard_00.tgz ... cross_entity_dataset_shard_07.tgz
#   cross_entity_dataset_shard.manifest.json
#
# Each shard is its own .tgz containing a subset of episode stems under:
#   cross_entity/cross_entity_<stem>.csv
#   entity/entity_<stem>.csv
#   fleet/fleet_<stem>.csv
#   planet/planet_<stem>.csv
#
# Extract every shard with:
#   tar -C data/datasets -xzf cross_entity_dataset_shard_XX.tgz
#
# The extracted directories merge naturally; no cat/byte-merge step is needed.
#
# Optional:
#   SOURCE_ROOT=/path/to/data/datasets
#   SPLIT_MANIFEST=/path/to/manifest.json
#
# When SPLIT_MANIFEST is provided, only the stems listed in its
# train/val/test arrays are packed, and the manifest can be uploaded as
# gs://.../manifest.json next to the shards.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/orbit-cross-dataset-shards}"
BUCKET="${BUCKET:-gs://orbit-wars-shipping/cross_entity}"
PREFIX="${PREFIX:-cross_entity_dataset_shard}"
SHARDS="${SHARDS:-8}"
UPLOAD="${UPLOAD:-0}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/data/datasets}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"
UPLOAD_SPLIT_MANIFEST="${UPLOAD_SPLIT_MANIFEST:-1}"

if [ "$SHARDS" -lt 1 ]; then
  echo "SHARDS must be >= 1" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/${PREFIX}_*.tgz "$OUT_DIR/${PREFIX}.manifest.json"

for dir in cross_entity entity fleet planet; do
  if [ ! -d "$SOURCE_ROOT/$dir" ]; then
    echo "missing $SOURCE_ROOT/$dir" >&2
    exit 1
  fi
done

STEMS_FILE="$OUT_DIR/stems.txt"
if [ -n "$SPLIT_MANIFEST" ]; then
  python3 - "$SPLIT_MANIFEST" > "$STEMS_FILE" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
seen = set()
for split in ("train", "val", "test"):
    for name in manifest.get(split, []):
        stem = name.removeprefix("cross_entity_").removesuffix(".csv")
        if stem not in seen:
            seen.add(stem)
            print(stem)
PY
else
  find "$SOURCE_ROOT/cross_entity" -maxdepth 1 -name 'cross_entity_*.csv' \
    -exec basename {} .csv \; \
    | sed 's/^cross_entity_//' \
    | sort > "$STEMS_FILE"
fi

N_STEMS="$(wc -l < "$STEMS_FILE" | tr -d ' ')"
if [ "$N_STEMS" = "0" ]; then
  echo "no cross_entity CSV stems found" >&2
  exit 1
fi

echo "[shard] source=$SOURCE_ROOT"
echo "[shard] stems=$N_STEMS shards=$SHARDS out=$OUT_DIR"
if [ -n "$SPLIT_MANIFEST" ]; then
  echo "[shard] split manifest=$SPLIT_MANIFEST"
fi

MANIFEST_LINES=()
for ((i = 0; i < SHARDS; i++)); do
  shard="$(printf '%02d' "$i")"
  list="$OUT_DIR/${PREFIX}_${shard}.files"
  : > "$list"
  awk -v shard="$i" -v shards="$SHARDS" 'NR % shards == shard {print}' "$STEMS_FILE" \
    | while read -r stem; do
        for rel in \
          "cross_entity/cross_entity_${stem}.csv" \
          "entity/entity_${stem}.csv" \
          "fleet/fleet_${stem}.csv" \
          "planet/planet_${stem}.csv"; do
          if [ ! -f "$SOURCE_ROOT/$rel" ]; then
            echo "missing $SOURCE_ROOT/$rel" >&2
            exit 1
          fi
          echo "$rel" >> "$list"
        done
      done
  n_files="$(wc -l < "$list" | tr -d ' ')"
  out="$OUT_DIR/${PREFIX}_${shard}.tgz"
  echo "  shard $shard: $n_files files -> $out"
  tar -C "$SOURCE_ROOT" -czf "$out" -T "$list"
  size="$(stat -f '%z' "$out" 2>/dev/null || stat -c '%s' "$out")"
  MANIFEST_LINES+=("{\"name\":\"${PREFIX}_${shard}.tgz\",\"bytes\":${size},\"files\":${n_files}}")
done

{
  echo "{"
  echo "  \"prefix\": \"$PREFIX\","
  echo "  \"shards\": $SHARDS,"
  echo "  \"stems\": $N_STEMS,"
  echo "  \"objects\": ["
  for ((i = 0; i < ${#MANIFEST_LINES[@]}; i++)); do
    comma=","
    if [ "$i" -eq "$((${#MANIFEST_LINES[@]} - 1))" ]; then
      comma=""
    fi
    echo "    ${MANIFEST_LINES[$i]}$comma"
  done
  echo "  ]"
  echo "}"
} > "$OUT_DIR/${PREFIX}.manifest.json"

echo
echo "[shard] outputs:"
du -h "$OUT_DIR"/${PREFIX}_*.tgz "$OUT_DIR/${PREFIX}.manifest.json"

if [ "$UPLOAD" = "1" ]; then
  echo
  echo "[upload] $BUCKET"
  gsutil -m cp "$OUT_DIR"/${PREFIX}_*.tgz "$BUCKET/"
  gsutil cp "$OUT_DIR/${PREFIX}.manifest.json" "$BUCKET/${PREFIX}.manifest.json"
  if [ -n "$SPLIT_MANIFEST" ] && [ "$UPLOAD_SPLIT_MANIFEST" = "1" ]; then
    gsutil cp "$SPLIT_MANIFEST" "$BUCKET/manifest.json"
  fi
else
  echo
  echo "[upload commands]"
  echo "  gsutil -m cp $OUT_DIR/${PREFIX}_*.tgz $BUCKET/"
  echo "  gsutil cp $OUT_DIR/${PREFIX}.manifest.json $BUCKET/${PREFIX}.manifest.json"
  if [ -n "$SPLIT_MANIFEST" ] && [ "$UPLOAD_SPLIT_MANIFEST" = "1" ]; then
    echo "  gsutil cp $SPLIT_MANIFEST $BUCKET/manifest.json"
  fi
fi
