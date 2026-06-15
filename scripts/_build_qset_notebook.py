"""One-shot builder for notebooks/pretrain_qset_colab.ipynb.

Stage 3: train Q_set (set-level value) on the episode OUTCOME (+1/-1), warm from
the set-token ckpt (Stage 2). Accuracy matters — it's the energy the
(source×target×size) diffusion search optimizes — so use the largest good/bad
cache available. No doomed label.

Run: .venv/bin/python scripts/_build_qset_notebook.py
"""
import json
from pathlib import Path

C = []

C.append(("markdown", """\
# Q_set pretrain — set-level value on the win/loss outcome (Stage 3)

Warm from the **set-token ckpt** (Stage 2). Forward → gather launches → set
encoder → `set_token` → SharedQTrunk → **`Q_set` regressed (Huber) to the
episode MC return (+1 win / -1 loss)**. The set token + frac/ships launch
encoding are fine-tuned (small LR); L2/L3/L4 tiny LR.

Good + bad replays are essential (the current topmeta300 cache is ~55/45);
**swap to the doubled all-players cache for the large dataset** — accuracy here
is what the `(source × target × fleet-size)` Q-guided diffusion search depends
on. No doomed recompute. Watch: **VAL `sign_acc`**, **`corr`**, and the
**Q[win] vs Q[loss]** gap widening.
"""))

C.append(("markdown", "## 1. Authenticate + pull bundle"))

C.append(("code", """\
from google.colab import auth
auth.authenticate_user()
BUCKET = 'gs://orbit-wars-shipping/entity'
PAIR_CACHE_PREFIX = 'pair_cache_topmeta300'           # ← swap to the doubled cache for MORE data
OUTCOME_SRC = f'{BUCKET}/runs/topmeta300_outcome.pt'  # ← matching outcome sidecar
# SET-TOKEN ckpt from Stage 2 (set the run tag you produced):
WARM_START_SRC = f'{BUCKET}/runs/set_token_topmeta300_d256_REPLACE_WITH_YOUR_TS/set_token_best.pt'
print('warm (set-token):', WARM_START_SRC, '\\noutcome:', OUTCOME_SRC)
"""))

C.append(("code", """\
import os, subprocess, time, hashlib, json, concurrent.futures
from pathlib import Path
WORK = Path('/content/orbit-wars'); WORK.mkdir(parents=True, exist_ok=True); os.chdir(WORK)

def _gcs_size(url):
    try:
        o = subprocess.run(['gcloud','storage','objects','describe',url,'--format=value(size)'],
                           check=True, capture_output=True, text=True); return int(o.stdout.strip())
    except Exception: return None
def _cp(src, dst):
    dst = Path(dst); dst.unlink(missing_ok=True)
    print(f'  pulling {src} -> {dst.name}', flush=True)
    subprocess.run(['gcloud','storage','cp',src,str(dst)], check=True); return dst.stat().st_size
def _sha256(p):
    h = hashlib.sha256()
    with open(p,'rb') as fh:
        for b in iter(lambda: fh.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def pull_cache(prefix, dst):
    dst = Path(dst); man_url = f'{BUCKET}/{prefix}.manifest.json'
    man = json.loads(subprocess.run(['gcloud','storage','cat',man_url], check=True, capture_output=True, text=True).stdout)
    total = int(man.get('total_bytes', 0))
    if dst.exists() and dst.stat().st_size == total: print(f'  {dst.name}: cached'); return
    cdir = WORK/f'{prefix}_chunks'; cdir.mkdir(exist_ok=True)
    def _p(s):
        cp = cdir/s['name']
        if not (cp.exists() and cp.stat().st_size == int(s.get('size_bytes',0))):
            subprocess.run(['gcloud','storage','cp',f"{BUCKET}/{s['name']}",str(cp)], check=True)
        if s.get('sha256') and _sha256(cp) != s['sha256']: raise RuntimeError('sha mismatch')
        return s['name']
    print(f'  {prefix}: {len(man["chunks"])} chunks {total/1024**3:.1f} GB')
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(man['chunks'])) as pool: list(pool.map(_p, man['chunks']))
    dst.unlink(missing_ok=True)
    with open(dst,'wb') as out:
        for c in man['chunks']:
            with open(cdir/c['name'],'rb') as fh:
                while True:
                    b = fh.read(1<<22)
                    if not b: break
                    out.write(b)
    import shutil; shutil.rmtree(cdir, ignore_errors=True); print(f'  assembled {dst.name}')

assert _gcs_size(WARM_START_SRC) is not None, 'set WARM_START_SRC to your Stage-2 set-token ckpt!'
_cp(f'{BUCKET}/code.tgz', WORK/'code.tgz'); _cp(f'{BUCKET}/weights.tgz', WORK/'weights.tgz')
pull_cache(PAIR_CACHE_PREFIX, WORK/'pair_cache.pt')
_cp(OUTCOME_SRC, WORK/'outcome.pt'); _cp(WARM_START_SRC, WORK/'warm_start.pt')
print('pull done')
"""))

C.append(("code", """\
!rm -rf agents scripts ckpts
!find . -maxdepth 1 -name '*.pt' ! -name 'pair_cache.pt' ! -name 'outcome.pt' ! -name 'warm_start.pt' -delete
!tar xzf code.tgz && tar xzf weights.tgz
import sys, importlib, gc
for m in list(sys.modules):
    if m.startswith(('agents','scripts')): del sys.modules[m]
importlib.invalidate_caches(); gc.collect()
import shutil
for d in ('ckpts/planet','ckpts/fleet','ckpts/comet'): os.makedirs(d, exist_ok=True)
shutil.copy('planet_encoder_best.pt','ckpts/planet/planet_encoder_best.pt')
shutil.copy('fleet_encoder_best.pt','ckpts/fleet/fleet_encoder_best.pt')
shutil.copy('comet_past_best.pt','ckpts/comet/comet_past_best.pt')
import torch
print('cuda:', torch.cuda.is_available())
oc = torch.load('outcome.pt', weights_only=False)
print('outcome labels:', len(oc['outcome']), '| win', oc.get('n_win'), 'loss', oc.get('n_loss'))
"""))

C.append(("markdown", "## 2. Train Q_set"))

C.append(("code", """\
import time
RUN_TAG = f'qset_topmeta300_d256_{time.strftime("%Y%m%d-%H%M%S")}'
OUT_DIR = f'data/runs/joint/{RUN_TAG}'; print('RUN_TAG =', RUN_TAG)
"""))

C.append(("code", """\
!python -u -m agents.transformer_v3.qset_pretrain \\
  --out-dir $OUT_DIR \\
  --pair-cache-path /content/orbit-wars/pair_cache.pt \\
  --outcome-sidecar /content/orbit-wars/outcome.pt \\
  --fleet-run-dir ckpts/fleet --planet-run-dir ckpts/planet --comet-run-dir ckpts/comet \\
  --warm-start /content/orbit-wars/warm_start.pt \\
  --epochs 25 --batch-size 32 \\
  --lr-q 1e-4 --lr-set 2e-5 --lr-l34 5e-6 --lr-l2 5e-6 \\
  --num-workers 2 --device cuda --progress-every 50
"""))

C.append(("markdown", "## 3. Push the run to GCS"))

C.append(("code", """\
import subprocess
from pathlib import Path
src = Path(OUT_DIR); assert src.is_dir()
subprocess.run(['gcloud','storage','cp','--recursive',str(src),f'{BUCKET}/runs/'], check=True)
print('uploaded to', f'{BUCKET}/runs/{src.name}/  (qset_best.pt = the diffusion energy)')
"""))


def cell(kind, src):
    base = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code": base.update({"execution_count": None, "outputs": []})
    return base


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "cells": [cell(k, s) for k, s in C]}
out = Path(__file__).resolve().parent.parent / "notebooks" / "pretrain_qset_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out} ({len(nb['cells'])} cells)")
