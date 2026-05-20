#!/usr/bin/env bash
# Byte-split a pair cache .pt into N chunks and upload them to GCS in
# parallel. The reassembled ``cat chunk_* > pair_cache.pt`` produces a
# byte-identical copy of the original — no torch round-trip needed.
#
# Why chunk?  A single ~33 GB upload over residential uplink throttles
# to ~5 MB/s (we measured this empirically — see prior single-stream
# attempt that took >2 hr for 76% of the file). Parallel uploads of
# many smaller objects each negotiate their own bandwidth budget; in
# practice 4-8 concurrent uploads saturate the available uplink.
#
# Pairs with the notebook's parallel pull cell:
#
#   gs://orbit-wars-shipping/entity/pair_cache_top4.part_00   ─┐
#   gs://orbit-wars-shipping/entity/pair_cache_top4.part_01   ─┤  pulled in
#   gs://orbit-wars-shipping/entity/pair_cache_top4.part_02   ─┤  parallel via
#   gs://orbit-wars-shipping/entity/pair_cache_top4.part_03   ─┘  ThreadPool +
#   gs://orbit-wars-shipping/entity/pair_cache_top4.manifest  ─►  cat → .pt
#   ...
#
# Run:
#   bash scripts/chunk_and_upload_pair_cache.sh                # default
#   CHUNKS=16 UPLOAD=0 bash scripts/chunk_and_upload_pair_cache.sh
#   SOURCE=.../bowEbi.pt PREFIX=pair_cache_top4 UPLOAD=1 bash ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${SOURCE:-$REPO_ROOT/data/datasets/_pair_cache/bowwowforeach_Ebi_ErfanEshratifar_Shun_PI_T6/bowwowforeach_Ebi_ErfanEshratifar_Shun_PI_T6_p64_f1024_all.pt}"
PREFIX="${PREFIX:-pair_cache_top4}"           # destination object stem on GCS
CHUNKS="${CHUNKS:-8}"                          # number of byte chunks
BUCKET="${BUCKET:-gs://orbit-wars-shipping/entity}"
UPLOAD="${UPLOAD:-1}"                          # 1 → also gsutil -m cp
STAGE_DIR="${STAGE_DIR:-/tmp/orbit-pair-chunks}"

if [ ! -f "$SOURCE" ]; then
  echo "[chunk] source not found: $SOURCE" >&2
  exit 1
fi
SOURCE_BYTES="$(stat -f%z "$SOURCE" 2>/dev/null || stat -c%s "$SOURCE")"
SOURCE_BYTES_GB="$(python3 -c "print(f'{$SOURCE_BYTES / 1024**3:.2f}')")"
# Chunk size = ceil(total / N). Use python for integer arithmetic since
# Bash $((..)) is fine on 64-bit Linux/macOS but easier to read this way.
CHUNK_BYTES="$(python3 -c "import math; print(math.ceil($SOURCE_BYTES / $CHUNKS))")"
CHUNK_GB="$(python3 -c "print(f'{$CHUNK_BYTES / 1024**3:.2f}')")"

echo "[chunk] source:     $SOURCE  ($SOURCE_BYTES_GB GB)"
echo "[chunk] N chunks:   $CHUNKS  (~$CHUNK_GB GB each)"
echo "[chunk] stage dir:  $STAGE_DIR"
echo "[chunk] prefix:     $PREFIX"
echo "[chunk] bucket:     $BUCKET"

# Clean prior staging output so re-runs don't half-overwrite.
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

# Zero-padded suffix width: ceil(log10(CHUNKS)) but at least 2 digits so
# we don't get `_0..._9, _10` lex-order surprises.
SUFFIX_WIDTH="$(python3 -c "print(max(2, len(str($CHUNKS - 1))))")"

echo "[chunk] splitting ..."
# Cross-platform split. macOS BSD `split` doesn't have --suffix-length
# (it has -a) and uses --numeric-suffixes differently; we use the POSIX
# subset (-a, -d, -b).
split -b "$CHUNK_BYTES" -a "$SUFFIX_WIDTH" -d \
  "$SOURCE" "$STAGE_DIR/${PREFIX}.part_"

# Verify chunk count matches expectation (split may produce fewer if
# CHUNK_BYTES * CHUNKS > SOURCE_BYTES, which can't happen with our ceil
# math, but cheap to guard).
ACTUAL_CHUNKS="$(find "$STAGE_DIR" -maxdepth 1 -name "${PREFIX}.part_*" | wc -l | tr -d ' ')"
if [ "$ACTUAL_CHUNKS" != "$CHUNKS" ]; then
  echo "[chunk] note: produced $ACTUAL_CHUNKS chunks (expected $CHUNKS)"
  CHUNKS="$ACTUAL_CHUNKS"
fi

# Write a small manifest so the notebook (or any consumer) knows exactly
# how many chunks to pull + their sha256s for end-to-end verification.
MANIFEST="$STAGE_DIR/${PREFIX}.manifest.json"
python3 - "$STAGE_DIR" "$PREFIX" "$SOURCE_BYTES" "$MANIFEST" <<'PY'
import json, hashlib, os, sys
stage_dir, prefix, source_bytes, manifest_path = sys.argv[1:5]
chunks = sorted(
    f for f in os.listdir(stage_dir)
    if f.startswith(f"{prefix}.part_")
)
out = {"prefix": prefix, "total_bytes": int(source_bytes), "chunks": []}
for name in chunks:
    p = os.path.join(stage_dir, name)
    size = os.path.getsize(p)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    out["chunks"].append({
        "name": name,
        "size_bytes": size,
        "sha256": h.hexdigest(),
    })
with open(manifest_path, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"[manifest] wrote {manifest_path}  ({len(chunks)} chunks)")
PY

echo
echo "[chunk] chunks staged:"
ls -lh "$STAGE_DIR"/${PREFIX}.part_* | awk '{print "  " $NF "  " $5}'
echo "  $(basename "$MANIFEST")  $(stat -f%z "$MANIFEST" 2>/dev/null || stat -c%s "$MANIFEST") bytes"

if [ "$UPLOAD" != "1" ]; then
  echo
  echo "[upload] skipped (UPLOAD=0). To upload:"
  echo "  gsutil -m cp \"$STAGE_DIR/${PREFIX}.part_*\" \"$BUCKET/\""
  echo "  gsutil cp \"$MANIFEST\" \"$BUCKET/$(basename "$MANIFEST")\""
  exit 0
fi

echo
echo "[upload] gsutil -m cp $CHUNKS chunks + manifest to $BUCKET ..."
# Glob expansion into multiple positional args lets gsutil -m parallelize
# across distinct objects.
gsutil -m cp "$STAGE_DIR"/${PREFIX}.part_* "$BUCKET/"
gsutil cp "$MANIFEST" "$BUCKET/$(basename "$MANIFEST")"

echo
echo "[upload] done. confirming on GCS:"
gsutil ls -lh "$BUCKET/${PREFIX}.part_*" "$BUCKET/$(basename "$MANIFEST")" | tail -n +1
