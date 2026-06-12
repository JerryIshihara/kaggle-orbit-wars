#!/usr/bin/env bash
# Build + upload the Colab code bundle (code.tgz) the pretrain notebooks pull.
#
#   bash scripts/build_code_tgz.sh            # build + upload
#   UPLOAD=0 bash scripts/build_code_tgz.sh   # build only
#
# CRITICAL: agents/__init__.py is REPLACED with a minimal shim. The repo's
# real __init__ wires the agent registry and imports kaggle_environments /
# heuristic agents, none of which exist on Colab pretrain VMs. The notebooks'
# wiring cells assert 'Minimal' in agents.__doc__ to catch a bundle built
# with the full registry init by mistake (this exact regression shipped on
# 2026-06-12 when an ad-hoc rsync rebuild dropped the shim).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE:-/tmp/ow_code_stage}"
BUCKET="${BUCKET:-gs://orbit-wars-shipping/entity}"
UPLOAD="${UPLOAD:-1}"

rm -rf "$STAGE"
mkdir -p "$STAGE/scripts"
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pt' \
  --exclude='.DS_Store' "$REPO_ROOT/agents" "$STAGE/"

# Swap in the minimal Colab shim (see header).
cat > "$STAGE/agents/__init__.py" <<'PY'
"""Minimal Colab shim for the ``agents`` package.

The repo's real ``__init__`` wires the agent registry and imports
kaggle_environments-backed heuristic agents — none of which exist on the
Colab pretrain VMs. Pretrain code only needs ``agents`` importable as a
namespace for ``agents.transformer_v2`` / ``agents.transformer_v3``
(their internals use relative imports). Notebook wiring cells assert
``'Minimal' in agents.__doc__`` to catch a code.tgz built with the full
registry init by mistake.
"""
PY

touch "$STAGE/scripts/__init__.py"
cp "$REPO_ROOT/scripts/build_pair_dataset_orbital_occle.py" \
   "$REPO_ROOT/scripts/show_multi_target_labels.py" \
   "$STAGE/scripts/"

tar czf /tmp/code.tgz -C "$STAGE" .
echo "[code.tgz] $(du -h /tmp/code.tgz | cut -f1)  ($(tar tzf /tmp/code.tgz | wc -l | tr -d ' ') entries, \
$(tar tzf /tmp/code.tgz | grep -c transformer_v3) transformer_v3 files)"
python3 - <<'PY'
import tarfile
with tarfile.open('/tmp/code.tgz') as t:
    init = t.extractfile('./agents/__init__.py').read().decode()
assert 'Minimal' in init, 'shim swap failed'
print('[code.tgz] agents/__init__.py shim verified (Minimal)')
PY

if [ "$UPLOAD" = "1" ]; then
  gcloud storage cp /tmp/code.tgz "$BUCKET/code.tgz"
  gcloud storage ls -l "$BUCKET/code.tgz"
fi
