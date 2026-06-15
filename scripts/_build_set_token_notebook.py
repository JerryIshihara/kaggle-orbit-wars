"""One-shot builder for notebooks/pretrain_set_token_colab.ipynb.

Stage: pretrain the ACTION-SET TOKEN (self-supervised — autoencoder + masked
denoise, NO outcome). Shared Q-trunk + aux value/forecast heads are baked in as
forward-compatible stubs (trained in the next stage).

Run: .venv/bin/python scripts/_build_set_token_notebook.py
"""
import json
from pathlib import Path

C = []

C.append(("markdown", """\
# Action-set token pretrain (self-supervised: autoencoder + masked denoise)

Learns a permutation-invariant SET token over a turn's launches. Launch token =
`source_joint[s] ⊕ target_joint[t] ⊕ frac_embed(f) ⊕ ships_embed(log ships)`
(invariant + ABSOLUTE force). Two joint tasks, NO outcome:

* **AE** — full set → bottleneck `set_token` → reconstruct every owned source's
  target/HOLD.
* **MASK** — mask a random subset of fired launches → predict them from the rest
  (joint completion: coalitions / complementary targets).

Trunk L0/L1 frozen, **L2/L3/L4 small-LR (1e-5)**. Shared Q-trunk + aux heads
(q_set, 5 signals × 5 horizons forecast, survives, material) are saved as fresh
STUBS for the next stage. Watch: **AE `launch_acc`** and **MASK `launch_acc`**
climbing together.
"""))

C.append(("markdown", "## 1. Authenticate + pull bundle from GCS"))

C.append(("code", """\
from google.colab import auth
auth.authenticate_user()
BUCKET = 'gs://orbit-wars-shipping/entity'
PAIR_CACHE_PREFIX = 'pair_cache_topmeta300'
# Trunk warm start = the joint single-target ckpt (mature trunk; value/q heads
# are stripped, only L0..L4 + PairHead load).
WARM_START_SRC = (f'{BUCKET}/runs/jst_v5_topmeta300_d256_20260614-162516/'
                  f'jst_last.pt')
print(f'pulling from {BUCKET}\\n  pair={PAIR_CACHE_PREFIX}\\n  warm={WARM_START_SRC}')
"""))

C.append(("code", """\
import os, subprocess, time, hashlib, json, concurrent.futures
from pathlib import Path
WORK = Path('/content/orbit-wars'); WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

def _gcs_size(url):
    try:
        out = subprocess.run(['gcloud','storage','objects','describe',url,
                              '--format=value(size)'], check=True,
                             capture_output=True, text=True)
        return int(out.stdout.strip())
    except Exception:
        return None

def _cp(src, dst, force=True):
    dst = Path(dst)
    if dst.exists() and force: dst.unlink()
    print(f'  pulling {src} -> {dst.name} ...', flush=True)
    subprocess.run(['gcloud','storage','cp',src,str(dst)], check=True)
    return dst.stat().st_size

def _sha256(p):
    h = hashlib.sha256()
    with open(p,'rb') as fh:
        for blk in iter(lambda: fh.read(1<<20), b''): h.update(blk)
    return h.hexdigest()

def pull_cache(prefix, dst):
    dst = Path(dst)
    man_url = f'{BUCKET}/{prefix}.manifest.json'
    if _gcs_size(man_url) is None:
        raise RuntimeError(f'no cache manifest for {prefix}')
    man = json.loads(subprocess.run(['gcloud','storage','cat',man_url],
                     check=True, capture_output=True, text=True).stdout)
    total = int(man.get('total_bytes', 0))
    if dst.exists() and total and dst.stat().st_size == total:
        print(f'  {dst.name}: cached'); return
    cdir = WORK / f'{prefix}_chunks'; cdir.mkdir(exist_ok=True)
    def _pull(spec):
        cp = cdir / spec['name']
        if not (cp.exists() and cp.stat().st_size == int(spec.get('size_bytes',0))):
            subprocess.run(['gcloud','storage','cp',f"{BUCKET}/{spec['name']}",str(cp)], check=True)
        if spec.get('sha256') and _sha256(cp) != spec['sha256']:
            raise RuntimeError(f"sha256 mismatch {spec['name']}")
        return spec['name']
    print(f'  {prefix}: {len(man["chunks"])} chunks, {total/1024**3:.2f} GB')
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(man['chunks'])) as pool:
        list(pool.map(_pull, man['chunks']))
    if dst.exists(): dst.unlink()
    with open(dst,'wb') as out:
        for c in man['chunks']:
            with open(cdir/c['name'],'rb') as fh:
                while True:
                    b = fh.read(1<<22)
                    if not b: break
                    out.write(b)
    import shutil as _sh; _sh.rmtree(cdir, ignore_errors=True)
    print(f'  assembled {dst.name}: {dst.stat().st_size/1024**3:.2f} GB')

assert _gcs_size(WARM_START_SRC) is not None, f'{WARM_START_SRC} missing'
t0 = time.time()
_cp(f'{BUCKET}/code.tgz', WORK/'code.tgz')
_cp(f'{BUCKET}/weights.tgz', WORK/'weights.tgz')
pull_cache(PAIR_CACHE_PREFIX, WORK/'pair_cache.pt')
_cp(WARM_START_SRC, WORK/'warm_start.pt')
print(f'pull done in {time.time()-t0:.1f}s')
"""))

C.append(("code", """\
!rm -rf agents scripts ckpts
!find . -maxdepth 1 -name '*.pt' ! -name 'pair_cache.pt' ! -name 'warm_start.pt' -delete
!tar xzf code.tgz && tar xzf weights.tgz
import sys, importlib, gc
for m in list(sys.modules):
    if m.startswith('agents') or m.startswith('scripts'): del sys.modules[m]
importlib.invalidate_caches(); gc.collect()
!find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
print('extracted code.tgz + weights.tgz')
"""))

C.append(("markdown", "## 2. Wiring check — set-token modules + trunk warm-start"))

C.append(("code", """\
import torch, agents
from agents.transformer_v3 import EntityPretrainModelV3, adapt_v2_state_dict, N_UNION
from agents.transformer_v3.action_set_encoder import (
    ActionSetEncoder, SetReconHead, FracEmbed, SharedQTrunk, SetValueHeads,
    AUX_SIGNALS, AUX_HORIZONS)
assert 'Minimal' in (agents.__doc__ or ''), 'stale agents shim — restart kernel'
ck = torch.load('/content/orbit-wars/warm_start.pt', map_location='cpu', weights_only=False)
cfg = ck['config']
m = EntityPretrainModelV3(d_model=256,
                          conditioner_n_layers=int(cfg['conditioner_n_layers']),
                          head_n_layers=int(cfg['head_n_layers']),
                          with_consolidator=True, with_value_heads=False, with_short_aux=True)
sd = {k: v for k, v in ck['model'].items() if not k.startswith('value_heads.')}
res = m.load_state_dict(adapt_v2_state_dict(sd), strict=False)
bad = [k for k in res.missing_keys if not k.startswith(
       ('short_heads.','cross.fuse_player','cross.owner_proj','cross.long.player_tokens','cross.short.player_tokens'))]
assert not bad, f'backbone skew: {bad[:6]}'
# set-token modules
qt, vh = SharedQTrunk(256), SetValueHeads(256)
out = vh(qt(torch.randn(2,256), torch.randn(2,256), torch.randn(2,256)))
assert out['aux_fwd'].shape == (2, len(AUX_SIGNALS), len(AUX_HORIZONS))
print('OK: trunk loaded; set-token modules wire | aux', AUX_SIGNALS, AUX_HORIZONS)
del ck, m, sd
"""))

C.append(("markdown", "## 3. GPU + stage L0 encoders"))

C.append(("code", """\
import torch, shutil
from pathlib import Path
print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(cpu)')
PLANET_RUN_DIR = Path('/content/orbit-wars/ckpts/planet'); FLEET_RUN_DIR = Path('/content/orbit-wars/ckpts/fleet'); COMET_RUN_DIR = Path('/content/orbit-wars/ckpts/comet')
for d in (PLANET_RUN_DIR, FLEET_RUN_DIR, COMET_RUN_DIR): d.mkdir(parents=True, exist_ok=True)
shutil.copy('/content/orbit-wars/planet_encoder_best.pt', PLANET_RUN_DIR/'planet_encoder_best.pt')
shutil.copy('/content/orbit-wars/fleet_encoder_best.pt',  FLEET_RUN_DIR /'fleet_encoder_best.pt')
shutil.copy('/content/orbit-wars/comet_past_best.pt',     COMET_RUN_DIR /'comet_past_best.pt')
print('L0 staged')
"""))

C.append(("markdown", "## 4. Train (action-set token: autoencoder + masked denoise)"))

C.append(("code", """\
import time
TS = time.strftime('%Y%m%d-%H%M%S')
RUN_TAG = f'set_token_topmeta300_d256_{TS}'
OUT_DIR = f'data/runs/joint/{RUN_TAG}'
print('RUN_TAG =', RUN_TAG)
"""))

C.append(("code", """\
!python -u -m agents.transformer_v3.set_token_pretrain \\
  --out-dir $OUT_DIR \\
  --pair-cache-path /content/orbit-wars/pair_cache.pt \\
  --fleet-run-dir ckpts/fleet --planet-run-dir ckpts/planet --comet-run-dir ckpts/comet \\
  --warm-start /content/orbit-wars/warm_start.pt \\
  --epochs 20 --batch-size 32 \\
  --lr-set 1e-4 --lr-l34 1e-5 --lr-l2 1e-5 \\
  --d-frac 32 --d-ships 32 --mask-ratio 0.35 \\
  --num-workers 2 --device cuda --progress-every 50
"""))

C.append(("markdown", "## 5. Push the run to GCS"))

C.append(("code", """\
import subprocess
from pathlib import Path
src = Path(OUT_DIR); assert src.is_dir(), src
subprocess.run(['gcloud','storage','cp','--recursive',str(src),f'{BUCKET}/runs/'], check=True)
print('uploaded to', f'{BUCKET}/runs/{src.name}/')
"""))


def cell(kind, src):
    base = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "cells": [cell(k, s) for k, s in C]}
out = Path(__file__).resolve().parent.parent / "notebooks" / "pretrain_set_token_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out} ({len(nb['cells'])} cells)")
