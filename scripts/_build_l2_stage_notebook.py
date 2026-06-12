"""One-shot builder for notebooks/pretrain_l2only_dualfuse_colab.ipynb.

Stage-A pretrain: dual-rate L2 ONLY (v3.1 player tokens, no later parts).
Run: .venv/bin/python scripts/_build_l2_stage_notebook.py
"""

import json
from pathlib import Path

C = []  # (cell_type, source)

C.append(("markdown", """\
# Stage A — dual-L2-only pretrain (v3.1: player tokens, no later parts)

Trains **only the dual-rate L2** (long + short branches, the three zero-init
fusions, the pre-L2 owner projection, the 4 per-branch player CLS tokens) +
the stage's aux heads. L0/L1 frozen as always; **L3/L4, PairHead and the
value heads are neither built nor run** — they train in the later joint
stage, warm-started from this run's `l2_best.pt`.

Tasks (`agents/transformer_v3/l2_aux.py`) — all on the FUSED 512→256
outputs, each split short-term (t+5) / long-term (t+10):

| granularity | short (t+5) | long (t+10) |
|---|---|---|
| **planet** (fused ctx) | owner CE · ships Huber · arrivals/player Huber | same; + earliest-arrival CE |
| **player** (fused player tokens) | inbound · owned-frac · ships per slot | same |
| **global** (fused CLS) | ownership churn · board ship level | same |

Player/global targets are label-space aggregates derived on the fly from
the per-planet labels already in the pair cache — **no dataset rebuild**,
and NO cross cache in this stage (lighter pull). Uses **ALL ~50k
snapshots** (perception needs no acted filter; ~2.5× the joint stage's
rows) with an episode-level val split.

Warm start = the stopped `joint_v3dual` run (2 epochs): its fusion
short-half is already open and the short branch already forecasts
(`sh5/owner_acc 0.926`), so this stage refines rather than bootstraps.

**The output has no win head** — PPO cannot anchor on it directly; the
later joint stage is the bridge.
"""))

C.append(("markdown", "## 1. Authenticate + pull bundle from GCS"))

C.append(("code", """\
from google.colab import auth
auth.authenticate_user()
BUCKET = 'gs://orbit-wars-shipping/entity'
PAIR_CACHE_PREFIX = 'pair_cache_topmeta_jun10'
# Warm start: the stopped v3 dual-rate joint run (v3-shaped ckpt; the
# adapter passes it through 1:1, drops the removed consolidator, and the
# player-token machinery initializes fresh).
WARM_START_SRC = (f'{BUCKET}/runs/joint_v3dual_d256_T18u_head3_20260612-044208/'
                  f'joint_best.pt')
print(f'pulling from {BUCKET}\\n  action={PAIR_CACHE_PREFIX}'
      f'\\n  warm-start={WARM_START_SRC}\\n  (no cross cache in stage A)')
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
    if dst.exists() and not force:
        return dst.stat().st_size
    if dst.exists():
        dst.unlink()
    print(f'  pulling {src} -> {dst.name} ...', flush=True)
    subprocess.run(['gcloud','storage','cp',src,str(dst)], check=True)
    return dst.stat().st_size

def _sha256(p):
    h = hashlib.sha256()
    with open(p,'rb') as fh:
        for blk in iter(lambda: fh.read(1<<20), b''): h.update(blk)
    return h.hexdigest()

def pull_cache(prefix, dst):
    \"\"\"Chunked (manifest) or single-object cache pull + assemble.\"\"\"
    dst = Path(dst)
    man_url = f'{BUCKET}/{prefix}.manifest.json'
    if _gcs_size(man_url) is not None:
        man = json.loads(subprocess.run(['gcloud','storage','cat',man_url],
                         check=True, capture_output=True, text=True).stdout)
        total = int(man.get('total_bytes', 0))
        if dst.exists() and total and dst.stat().st_size == total:
            print(f'  {dst.name}: cached ({total/1024**3:.2f} GB)'); return
        cdir = WORK / f'{prefix}_chunks'; cdir.mkdir(exist_ok=True)
        def _pull(spec):
            cp = cdir / spec['name']
            if not (cp.exists() and cp.stat().st_size == int(spec.get('size_bytes',0))):
                subprocess.run(['gcloud','storage','cp',f"{BUCKET}/{spec['name']}",str(cp)], check=True)
            if spec.get('sha256') and _sha256(cp) != spec['sha256']:
                raise RuntimeError(f"sha256 mismatch {spec['name']}")
            return spec['name'], cp.stat().st_size
        print(f'  {prefix}: {len(man["chunks"])} chunks, {total/1024**3:.2f} GB')
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(man['chunks'])) as pool:
            for nm,sz in pool.map(_pull, man['chunks']):
                print(f'    {nm}  {sz/1024**2:.1f} MB')
        if dst.exists(): dst.unlink()
        with open(dst,'wb') as out:
            for c in man['chunks']:
                with open(cdir/c['name'],'rb') as fh:
                    while True:
                        b = fh.read(1<<22)
                        if not b: break
                        out.write(b)
        print(f'  assembled {dst.name}: {dst.stat().st_size/1024**3:.2f} GB')
        import shutil as _sh; _sh.rmtree(cdir, ignore_errors=True)
        print(f'  freed chunk dir {cdir.name}')
        return
    raise RuntimeError(f'no cache manifest for {prefix} on {BUCKET}')

assert _gcs_size(WARM_START_SRC) is not None, f'{WARM_START_SRC} missing'
t0 = time.time()
_cp(f'{BUCKET}/code.tgz', WORK/'code.tgz')
_cp(f'{BUCKET}/weights.tgz', WORK/'weights.tgz')
pull_cache(PAIR_CACHE_PREFIX, WORK/'pair_cache.pt')
_cp(WARM_START_SRC, WORK/'warm_start.pt')
print(f'pull done in {time.time()-t0:.1f}s')
"""))

C.append(("code", """\
# Wipe stale extracted code; leave the big cache alone.
!rm -rf agents scripts ckpts
!find . -maxdepth 1 -name '*.pt' ! -name 'pair_cache.pt' ! -name 'warm_start.pt' -delete
!tar xzf code.tgz
!tar xzf weights.tgz
import sys, importlib, gc
for m in list(sys.modules):
    if m.startswith('agents') or m.startswith('scripts'): del sys.modules[m]
importlib.invalidate_caches(); gc.collect()
!find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
print('extracted code.tgz + weights.tgz')
"""))

C.append(("markdown", """\
## 2. Verify wiring — v3.1 player tokens, adapter vs the real warm ckpt

Asserts: the warm checkpoint is v3-shaped and passes through the adapter
with only fresh keys missing (owner proj / player tokens / fuse_player);
fusions and owner projection are zero-init; player tokens are invisible to
the planet/global path (mask); the aux heads produce every task term.
"""))

C.append(("code", """\
import torch, agents
from agents.transformer_v3 import (
    EntityPretrainModelV3, adapt_v2_state_dict, N_UNION,
)
from agents.transformer_v3.l2_aux import DualL2AuxHeads, dual_l2_aux_loss
assert 'Minimal' in (agents.__doc__ or ''), 'stale agents shim — restart kernel'

ck = torch.load('/content/orbit-wars/warm_start.pt', map_location='cpu',
                weights_only=False)
cfg = ck['config']
assert cfg.get('arch') == 'dual_rate_l2_v3' and cfg.get('n_steps') == N_UNION, cfg
# Mirror the ckpt's head/conditioner depth (head3/cond3) — otherwise the
# (unused-in-stage-A) PairHead's FiLM keys mismatch and trip the skew check.
m = EntityPretrainModelV3(d_model=256, skip_l34=True,
                          conditioner_n_layers=int(cfg['conditioner_n_layers']),
                          head_n_layers=int(cfg['head_n_layers']),
                          with_consolidator=True, with_value_heads=False,
                          with_short_aux=False)
assert m.consolidator is None and m.cross.owner_proj.weight.abs().sum() == 0
W = m.cross.fuse_player.weight.detach()
assert torch.equal(W[:, :256], torch.eye(256)) and W[:, 256:].abs().sum() == 0
sd = {k: v for k, v in ck['model'].items()
      if not k.startswith(('value_heads.', 'short_heads.'))}
res = m.load_state_dict(adapt_v2_state_dict(sd), strict=False)
fresh_ok = ('cross.fuse_player', 'cross.owner_proj',
            'cross.long.player_tokens', 'cross.short.player_tokens')
bad = [k for k in res.missing_keys if not k.startswith(fresh_ok)]
assert not bad, f'backbone skew: {bad[:6]}'
assert torch.equal(m.cross.long.step_embed, ck['model']['cross.long.step_embed'])
fw = ck['model']['cross.fuse_tokens.weight']
print(f"warm ckpt OK: v3-shaped passthrough; fusion short-half |W|="
      f"{fw[:, 256:].abs().sum():.1f} (already open); fresh: player/owner/"
      f"fuse_player")

aux = DualL2AuxHeads(256)
B, P = 2, 8
preds = aux(torch.randn(B, P, 256), torch.randn(B, 4, 256), torch.randn(B, 256))
assert preds['player'].shape == (B, 4, 6) and preds['glob'].shape == (B, 4)
print('aux heads OK:', sorted(preds))
del ck, m, sd, aux
"""))

C.append(("markdown", "## 3. GPU check"))

C.append(("code", """\
import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(cpu)')
print('Stage A never runs PairHead (the P x P cell MLP) — per-step cost is '
      'dominated by L0/L1 over the 18-frame union. BATCH_SIZE=16 is safe; '
      '24 usually fits on A100/L4.')
"""))

C.append(("markdown", "## 4. Stage frozen L0 encoders into run-dir layout"))

C.append(("code", """\
import shutil
from pathlib import Path
PLANET_RUN_DIR = Path('/content/orbit-wars/ckpts/planet')
FLEET_RUN_DIR  = Path('/content/orbit-wars/ckpts/fleet')
COMET_RUN_DIR  = Path('/content/orbit-wars/ckpts/comet')
for d in (PLANET_RUN_DIR, FLEET_RUN_DIR, COMET_RUN_DIR): d.mkdir(parents=True, exist_ok=True)
shutil.copy('/content/orbit-wars/planet_encoder_best.pt', PLANET_RUN_DIR/'planet_encoder_best.pt')
shutil.copy('/content/orbit-wars/fleet_encoder_best.pt',  FLEET_RUN_DIR /'fleet_encoder_best.pt')
shutil.copy('/content/orbit-wars/comet_past_best.pt',     COMET_RUN_DIR /'comet_past_best.pt')
import torch
pc = torch.load(PLANET_RUN_DIR/'planet_encoder_best.pt', map_location='cpu', weights_only=False)
fc = torch.load(FLEET_RUN_DIR /'fleet_encoder_best.pt',  map_location='cpu', weights_only=False)
cc = torch.load(COMET_RUN_DIR /'comet_past_best.pt',     map_location='cpu', weights_only=False)
assert pc['config']['d_model']==fc['config']['d_model']==cc['config']['d_model']==256
print('L0 staged, all d=256')
"""))

C.append(("markdown", """\
## 5. Cache preflight — pair cache only

The driver restacks to the 18-frame union itself and asserts the t+5/t+10
label families; this cell just confirms the file + labels before the long
run starts.
"""))

C.append(("code", """\
PAIR_CACHE_PATH = '/content/orbit-wars/pair_cache.pt'
import os, torch
assert os.path.exists(PAIR_CACHE_PATH)
print(f'{PAIR_CACHE_PATH}  {os.path.getsize(PAIR_CACHE_PATH)/1024**3:.2f} GB')
payload = torch.load(PAIR_CACHE_PATH, map_location='cpu', weights_only=False, mmap=True)
snap = payload['snapshots'][0]
from agents.transformer_v3.l2_aux import L2_AUX_LABEL_KEYS
missing = [k for k in L2_AUX_LABEL_KEYS if k not in snap]
assert not missing, f'pair cache lacks {missing}'
print(f"pair cache OK: {len(payload['snapshots'])} snapshots (stage A trains "
      f"on ALL of them), t+5/t+10 label families present")
del payload, snap
"""))

C.append(("markdown", "## 6. Train (dual L2 + fused aux heads only)"))

C.append(("code", """\
import time
TS = time.strftime('%Y%m%d-%H%M%S')
RUN_TAG = f'l2only_dualfuse_d256_T18u_{TS}'
OUT_DIR = f'data/runs/joint/{RUN_TAG}'
BATCH_SIZE   = 16
EPOCHS       = 12
LR           = 1e-4   # L2-only: fresh player/owner parts want more than the
                      # joint stage's gentle 5e-5; warm branches tolerate it
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2
# Watch in the logs (printed per epoch, val side):
#   p/owner5_acc, p/owner10_acc  — the dual L2's forecast quality; the
#                                  5-vs-10 GAP shows short-vs-long branch
#                                  division of labor
#   p/earliest_acc               — who-strikes-first (should be high fast)
#   pl/*, g/*                    — player/global aggregates falling
#   total                        — l2_best.pt saves on val_total improvements
print('RUN_TAG =', RUN_TAG)
"""))

C.append(("code", """\
!python -u -m agents.transformer_v3.l2_pretrain \\
  --out-dir $OUT_DIR \\
  --pair-cache-path $PAIR_CACHE_PATH \\
  --fleet-run-dir $FLEET_RUN_DIR \\
  --planet-run-dir $PLANET_RUN_DIR \\
  --comet-run-dir $COMET_RUN_DIR \\
  --warm-start /content/orbit-wars/warm_start.pt \\
  --batch-size $BATCH_SIZE \\
  --epochs $EPOCHS \\
  --lr $LR \\
  --weight-decay $WEIGHT_DECAY \\
  --num-workers $NUM_WORKERS \\
  --device cuda \\
  --progress-every 50
"""))

C.append(("markdown", """\
## 7. Push the run to GCS

`l2_best.pt` = the stage-A artifact: a v3.1 model whose dual L2 (branches,
fusions, owner routing, player tokens) is trained on the forecast tasks.
The LATER joint stage warm-starts from it (adapter passes it through;
PairHead/L3/L4/value heads initialize there) and trains the action +
value superstructure. PPO needs that joint stage's win head.
"""))

C.append(("code", """\
import subprocess
from pathlib import Path
src = Path(OUT_DIR); assert src.is_dir(), src
subprocess.run(['gcloud','storage','cp','--recursive',str(src),f'{BUCKET}/runs/'], check=True)
print('uploaded to', f'{BUCKET}/runs/{src.name}/')
subprocess.run(['gcloud','storage','ls','--long','--readable-sizes',
                f'{BUCKET}/runs/{src.name}/'], check=False)
"""))


def cell(kind: str, src: str) -> dict:
    lines = src.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": [cell(k, s) for k, s in C],
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "pretrain_l2only_dualfuse_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out} ({len(nb['cells'])} cells)")
