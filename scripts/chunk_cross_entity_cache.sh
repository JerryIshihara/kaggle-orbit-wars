#!/usr/bin/env bash
# Byte-split a cross-entity .pt cache into chunks and optionally upload them.
#
# The V3 Colab notebook looks for either:
#   <CACHE_OBJECT>.manifest.json
#   <CACHE_OBJECT stem>.manifest.json
#
# Example:
#   SOURCE=data/datasets/cross_entity/cross_entity_cache_100k.pt \
#   PREFIX=cross_entity_cache_100k.pt \
#   CHUNKS=8 \
#   bash scripts/chunk_cross_entity_cache.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${SOURCE:-$REPO_ROOT/data/datasets/cross_entity/cross_entity_cache_100k.pt}"
PREFIX="${PREFIX:-$(basename "$SOURCE")}"
CHUNKS="${CHUNKS:-8}"
STAGE_DIR="${STAGE_DIR:-/tmp/orbit-cross-cache-chunks}"
BUCKET="${BUCKET:-gs://orbit-wars-shipping/cross_entity}"
UPLOAD="${UPLOAD:-1}"

if [ ! -f "$SOURCE" ]; then
  echo "[chunk] source not found: $SOURCE" >&2
  exit 1
fi

mkdir -p "$STAGE_DIR"
rm -f "$STAGE_DIR"/"${PREFIX}".part_* "$STAGE_DIR/${PREFIX}.manifest.json"

SOURCE_BYTES="$(stat -f%z "$SOURCE" 2>/dev/null || stat -c%s "$SOURCE")"
CHUNK_BYTES="$(( (SOURCE_BYTES + CHUNKS - 1) / CHUNKS ))"

echo "[chunk] source: $SOURCE"
echo "[chunk] bytes:  $SOURCE_BYTES"
echo "[chunk] chunks: $CHUNKS (~$CHUNK_BYTES bytes each)"
echo "[chunk] out:    $STAGE_DIR"

split -b "$CHUNK_BYTES" -d -a 2 "$SOURCE" "$STAGE_DIR/${PREFIX}.part_"

MANIFEST="$STAGE_DIR/${PREFIX}.manifest.json"
python3 - "$STAGE_DIR" "$PREFIX" "$SOURCE_BYTES" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

stage_dir = Path(sys.argv[1])
prefix = sys.argv[2]
source_bytes = int(sys.argv[3])
manifest_path = Path(sys.argv[4])

chunks = sorted(stage_dir.glob(f"{prefix}.part_*"))
payload = {"prefix": prefix, "total_bytes": source_bytes, "chunks": []}
for path in chunks:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            h.update(block)
    payload["chunks"].append({
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
    })

manifest_path.write_text(json.dumps(payload, indent=2))
print(f"[manifest] wrote {manifest_path} ({len(chunks)} chunks)")
PY

du -h "$STAGE_DIR"/"${PREFIX}".part_* "$MANIFEST"

if [ "$UPLOAD" = "1" ]; then
  echo "[upload] gsutil -m cp chunks + manifest to $BUCKET ..."
  gsutil -m cp "$STAGE_DIR"/"${PREFIX}".part_* "$MANIFEST" "$BUCKET/"
else
  echo "[upload] skipped. To upload:"
  echo "  gsutil -m cp $STAGE_DIR/${PREFIX}.part_* $MANIFEST $BUCKET/"
fi
