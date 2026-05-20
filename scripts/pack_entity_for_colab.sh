#!/usr/bin/env bash
# Pack a fresh entity-pretrain training bundle for ``notebooks/train_entity_encoder_colab.ipynb``.
#
# Produces (under $OUT_DIR, default /tmp/orbit-entity-pack):
#
#   code.tgz         ~250 KB   minimal ``agents`` shim + transformer_v2 +
#                              scripts/build_pair_dataset_orbital_occle.py
#   weights.tgz      ~16 MB    3 frozen L0 ckpts (planet + fleet + comet)
#
# The pair cache (``pair_cache.pt``, ~13 GB) is NOT re-packed by default — it
# lives at $BUCKET/pair_cache.pt already and only needs re-uploading when the
# upstream replay set changes. Run with INCLUDE_PAIR_CACHE=1 to force-include.
#
# Prints ready-to-paste ``gsutil cp`` commands unless UPLOAD=1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/orbit-entity-pack}"
BUCKET="${BUCKET:-gs://orbit-wars-shipping/entity}"
UPLOAD="${UPLOAD:-0}"
INCLUDE_PAIR_CACHE="${INCLUDE_PAIR_CACHE:-0}"

PAIR_CACHE_PATH="${PAIR_CACHE_PATH:-$REPO_ROOT/data/datasets/_pair_cache/bowwowforeach_Ebi_T6/bowwowforeach_Ebi_T6_p64_f1024_all.pt}"
PLANET_CKPT="${PLANET_CKPT:-$REPO_ROOT/data/runs/planet/specialist_planet_d256_no_traj_branch_40k_lr1e4_120ep/planet_encoder_best.pt}"
FLEET_CKPT="${FLEET_CKPT:-$REPO_ROOT/data/runs/fleet/specialist_fleet_d256_40k_lr1e4_120ep/fleet_encoder_best.pt}"
COMET_CKPT="${COMET_CKPT:-$REPO_ROOT/data/runs/comet/fullpath_scalar_multitask_d256_40k_lr1e4_120ep/comet_past_best.pt}"

mkdir -p "$OUT_DIR"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/orbit-entity-stage.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

# ---- code.tgz ----
# Layout inside the tarball (extracted at /content/orbit-wars/):
#   agents/{__init__.py, registry.py, physics_utils.py, transformer_v2/}
#   scripts/{__init__.py, build_pair_dataset_orbital_occle.py}
echo "[pack] building code.tgz ..."
STAGE_CODE="$STAGE/code"
mkdir -p "$STAGE_CODE/agents" "$STAGE_CODE/scripts"

# Bundle-only minimal ``agents/__init__.py``. The repo's real
# ``agents/__init__.py`` imports every sibling subtree (heuristic / archive /
# transformer_v2). The Colab notebook asserts ``'Minimal' in agents.__doc__``
# to confirm the shim — keep that token in the docstring.
cat > "$STAGE_CODE/agents/__init__.py" <<'PYEOF'
"""Minimal ``agents`` package for the Colab pair-pretrain bundle.

Local repo's ``agents/__init__.py`` imports every sibling subtree
(heuristic / archive / transformer_v2). The Colab bundle ships only
``transformer_v2``, so this shim wires up just what the entity
pretrain needs.
"""

from .registry import Agent, AgentSpec, list_agent_specs, list_agents, register  # noqa: F401
from . import transformer_v2  # noqa: F401
PYEOF

cp "$REPO_ROOT/agents/registry.py"      "$STAGE_CODE/agents/registry.py"
cp "$REPO_ROOT/agents/physics_utils.py" "$STAGE_CODE/agents/physics_utils.py"

# transformer_v2 tree, minus __pycache__ + .pyc.
(cd "$REPO_ROOT" && tar cf - \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    agents/transformer_v2) | (cd "$STAGE_CODE" && tar xf -)

# Pair-set dataset builder (the train cell does
# ``from scripts.build_pair_dataset_orbital_occle import CachedPairDataset``).
touch "$STAGE_CODE/scripts/__init__.py"
cp "$REPO_ROOT/scripts/build_pair_dataset_orbital_occle.py" \
   "$STAGE_CODE/scripts/build_pair_dataset_orbital_occle.py"

tar -C "$STAGE_CODE" -czf "$OUT_DIR/code.tgz" .
echo "  code.tgz: $(du -h "$OUT_DIR/code.tgz" | cut -f1)"

# ---- weights.tgz ----
# Layout inside the tarball (extracted at /content/orbit-wars/):
#   planet_encoder_best.pt
#   fleet_encoder_best.pt
#   comet_past_best.pt
# The notebook's ``ckpts`` cell does ``shutil.copy('.../planet_encoder_best.pt',
# PLANET_RUN_DIR / 'planet_encoder_best.pt')`` etc — flat layout in tarball, run
# dirs are created on the Colab side.
echo "[pack] building weights.tgz ..."
STAGE_WEIGHTS="$STAGE/weights"
mkdir -p "$STAGE_WEIGHTS"
for src_path in "$PLANET_CKPT" "$FLEET_CKPT" "$COMET_CKPT"; do
  if [ ! -f "$src_path" ]; then
    echo "  error: missing L0 ckpt: $src_path" >&2
    exit 1
  fi
  cp "$src_path" "$STAGE_WEIGHTS/$(basename "$src_path")"
  echo "  $(basename "$src_path"): $(du -h "$src_path" | cut -f1)"
done
tar -C "$STAGE_WEIGHTS" -czf "$OUT_DIR/weights.tgz" .
echo "  weights.tgz: $(du -h "$OUT_DIR/weights.tgz" | cut -f1)"

# ---- pair_cache.pt (optional, 13 GB) ----
OUTPUT_PAIR_CACHE=""
if [ "$INCLUDE_PAIR_CACHE" = "1" ]; then
  if [ ! -f "$PAIR_CACHE_PATH" ]; then
    echo "  error: pair cache not found: $PAIR_CACHE_PATH" >&2
    exit 1
  fi
  cp "$PAIR_CACHE_PATH" "$OUT_DIR/pair_cache.pt"
  OUTPUT_PAIR_CACHE="$OUT_DIR/pair_cache.pt"
  echo "  pair_cache.pt: $(du -h "$OUTPUT_PAIR_CACHE" | cut -f1)"
fi

echo
echo "[pack] outputs:"
du -h "$OUT_DIR"/* 2>/dev/null
echo
if [ "$UPLOAD" = "1" ]; then
  echo "[pack] uploading to $BUCKET ..."
  gsutil cp "$OUT_DIR/code.tgz"    "$BUCKET/code.tgz"
  gsutil cp "$OUT_DIR/weights.tgz" "$BUCKET/weights.tgz"
  if [ -n "$OUTPUT_PAIR_CACHE" ]; then
    gsutil cp "$OUTPUT_PAIR_CACHE" "$BUCKET/pair_cache.pt"
  fi
else
  echo "[pack] upload commands (run individually, or rerun this with UPLOAD=1):"
  echo "  gsutil cp $OUT_DIR/code.tgz    $BUCKET/code.tgz"
  echo "  gsutil cp $OUT_DIR/weights.tgz $BUCKET/weights.tgz"
  if [ -n "$OUTPUT_PAIR_CACHE" ]; then
    echo "  gsutil cp $OUTPUT_PAIR_CACHE $BUCKET/pair_cache.pt"
  fi
fi
