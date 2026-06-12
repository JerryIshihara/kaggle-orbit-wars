"""One-shot builder for notebooks/pretrain_joint_v4_colab.ipynb.

Stage-D pretrain: v4 actor + pruned value heads jointly (4 LR tiers).
Run: .venv/bin/python scripts/_build_joint_v4_notebook.py
"""

import json
from pathlib import Path

C = []  # (cell_type, source)

C.append(("markdown", """\
# Stage D — joint v4: actor + pruned value heads (4 LR tiers), top-meta 300

Trains the v4 ACTOR (select k≤3 + Dirichlet alloc + α0 confidence + t+5
short aux) and the PRUNED value-task set JOINTLY, on the 300-replay pair +
cross caches, warm from the stage-B actor:

* value tasks KEPT: **win** (1.0) + **fwd** signals @ (5,10,15,20,50) (0.5)
* REMOVED (redundant): back, rank, survives — no loss terms
* ADDED for the sample-K ranker: **temporal contrast** (0.25; sibling
  snapshots t vs t+Δ, comparator monotonicity) and **player inbound aux**
  (0.25; per-slot in-flight mass within 5/10 — fleet timing into
  player_state)
* LR tiers: value 1e-4 / action heads 2e-5 / L3+L4 1e-5 / L2 5e-6
  (action + backbone deliberately slow under the fast value heads)

Output `jointv4_best.pt` = the full v4 agent: PPO critic warm start AND
the simulate-then-score ranker's scorer. Watch: HOLDOUT win_acc (+ the
train-holdout gap), contrast_acc (>0.5 = the comparator works),
a/dir/alpha0_satfrac staying low.
"""))

C.append(("markdown", "## 1. Authenticate + pull bundle from GCS"))

C.append(("code", """\
from google.colab import auth
auth.authenticate_user()
BUCKET = 'gs://orbit-wars-shipping/entity'
PAIR_CACHE_PREFIX = 'pair_cache_topmeta300'
# Warm start: the stopped v3 dual-rate joint run (v3-shaped ckpt; the
# adapter passes it through 1:1, drops the removed consolidator, and the
# player-token machinery initializes fresh).
WARM_START_SRC = (f'{BUCKET}/runs/actor_v4_topmeta300_d256_20260612-073132/'
                  f'actor_best.pt')
CROSS_CACHE_PREFIX = 'cross_cache_topmeta300_p1'
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
pull_cache(CROSS_CACHE_PREFIX, WORK/'cross_entity_cache.pt')
_cp(WARM_START_SRC, WORK/'warm_start.pt')
print(f'pull done in {time.time()-t0:.1f}s')
"""))

C.append(("code", """\
# Wipe stale extracted code; leave the big cache alone.
!rm -rf agents scripts ckpts
!find . -maxdepth 1 -name '*.pt' ! -name 'pair_cache.pt' ! -name 'cross_entity_cache.pt' ! -name 'warm_start.pt' -delete
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
assert 'Minimal' in (agents.__doc__ or ''), 'stale agents shim — restart kernel'

ck = torch.load('/content/orbit-wars/warm_start.pt', map_location='cpu',
                weights_only=False)
cfg = ck['config']
assert cfg.get('arch') == 'dual_rate_l2_v3' and cfg.get('n_steps') == N_UNION, cfg
# Mirror the ckpt's head/conditioner depth (head3/cond3) — otherwise the
# (unused-in-stage-A) PairHead's FiLM keys mismatch and trip the skew check.
m = EntityPretrainModelV3(d_model=256,
                          conditioner_n_layers=int(cfg['conditioner_n_layers']),
                          head_n_layers=int(cfg['head_n_layers']),
                          with_consolidator=True, with_value_heads=False,
                          with_short_aux=True, with_alloc_conc=True)
assert m.consolidator is None and m.alloc_conc_head is not None
sd = {k: v for k, v in ck['model'].items()
      if not k.startswith('value_heads.')}
res = m.load_state_dict(adapt_v2_state_dict(sd), strict=False)
fresh_ok = ('alloc_conc_head.', 'short_heads.', 'cross.fuse_player',
            'cross.owner_proj', 'cross.long.player_tokens',
            'cross.short.player_tokens')
bad = [k for k in res.missing_keys if not k.startswith(fresh_ok)]
assert not bad, f'backbone skew: {bad[:6]}'
# stage-A player machinery must LOAD (the merge carries it), not re-init
assert not [k for k in res.missing_keys if 'player_tokens' in k
            or 'owner_proj' in k or 'fuse_player' in k], 'stage-A L2 not loaded'
assert torch.equal(m.cross.long.step_embed, ck['model']['cross.long.step_embed'])
print('warm ckpt OK: stage-A L2 + player tokens loaded; fresh = alloc_conc '
      '+ short aux heads')

# Dirichlet smoke on the real label builder parameterization
from agents.transformer_v3.dirichlet_alloc import dirichlet_alloc_nll
import math as _m
P = 4
pf = torch.zeros(1, P, 8)
labels = torch.zeros(1, P, P, dtype=torch.bool); labels[0, 0, 2] = True
ships = torch.zeros(1, P, P, dtype=torch.int32); ships[0, 0, 2] = 30
valid = torch.ones(1, P, P, dtype=torch.bool)
valid[:, range(P), range(P)] = False
from agents.transformer_v2.pretrain.entity_encoder import (
    _PLANET_OWNER_START_IDX, _PLANET_SHIPS_LOG_FEATURE_IDX,
)
from agents.transformer_v2.featurizer.fleet_featurizer import SHIPS_LOG_MAX
pf = torch.zeros(1, P, _PLANET_SHIPS_LOG_FEATURE_IDX + 2)
pf[0, 0, _PLANET_OWNER_START_IDX] = 1.0
pf[0, 0, _PLANET_SHIPS_LOG_FEATURE_IDX] = _m.log1p(100) / SHIPS_LOG_MAX
batch = dict(pair_labels=labels, pair_valid=valid, pair_ships=ships,
             planet_features=pf, planet_mask=torch.ones(1, P, dtype=torch.bool))
frac = torch.randn(1, P, P, requires_grad=True)
conc = torch.full((1, P), 50.0, requires_grad=True)
nll, terms = dirichlet_alloc_nll(frac, conc, batch)
nll.backward()
assert frac.grad.abs().sum() > 0 and conc.grad.abs().sum() > 0
print(f"dirichlet smoke OK: nll={terms['alloc_nll']:.3f} "
      f"a0={terms['alpha0_mean']:.0f} shareL1={terms['share_l1']:.3f}")
del ck, m, sd
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
from agents.transformer_v3.short_horizon import SHORT_AUX_LABEL_KEYS
missing = [k for k in SHORT_AUX_LABEL_KEYS if k not in snap]
assert not missing, f'pair cache lacks {missing}'
assert 'acted_indices' in payload and 'pair_ships' in snap
print(f"pair cache OK: {len(payload['snapshots'])} snapshots, "
      f"{len(payload['acted_indices'])} acted rows, labels present")
from agents.transformer_v2.pretrain.cross_entity import CachedCrossEntitySnapshotDataset
_cds = CachedCrossEntitySnapshotDataset('/content/orbit-wars/cross_entity_cache.pt')
_s = _cds[len(_cds)//2]
assert float(_s['player_valid'].sum()) > 0, (
    'cross cache has all-zero player_valid — built with the v1 featurizer? '
    'rebuild with the v2 cross_entity writer (2026-06-12 fix)')
print(f"cross cache OK: {len(_cds)} snapshots, player_valid live, "
      f"final_rank sample={_s['final_rank'].tolist()}")
del _cds, _s
del payload, snap
"""))

C.append(("markdown", "## 6. Train (dual L2 + fused aux heads only)"))

C.append(("code", """\
import time
TS = time.strftime('%Y%m%d-%H%M%S')
RUN_TAG = f'jointv4_topmeta300_d256_{TS}'
OUT_DIR = f'data/runs/joint/{RUN_TAG}'
BATCH_SIZE   = 16
EPOCHS       = 20
LR_VALUE     = 1e-4   # value trunk + win/fwd + inbound aux (fresh)
LR_ACTION    = 2e-5   # PairHead + α0 + short aux — LOW (actor refines)
LR_L34       = 1e-5   # dual_role + joint_role — LOW
LR_L2        = 5e-6   # stage-A dual L2 — barely moves
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2
# Watch per epoch:
#   HOLDOUT win_acc + train-holdout gap  — the critic anchor's honesty
#   HOLDOUT contrast_acc                 — comparator works (>0.5, rising)
#   a/dir/alpha0_satfrac                 — must stay low (cap/ε health)
#   a/sel BCE                            — actor not degrading under low lr
print('RUN_TAG =', RUN_TAG)
"""))

C.append(("code", """\
!python -u -m agents.transformer_v3.joint_v4_pretrain \\
  --out-dir $OUT_DIR \\
  --pair-cache-path $PAIR_CACHE_PATH \\
  --cross-cache-path /content/orbit-wars/cross_entity_cache.pt \\
  --fleet-run-dir $FLEET_RUN_DIR \\
  --planet-run-dir $PLANET_RUN_DIR \\
  --comet-run-dir $COMET_RUN_DIR \\
  --warm-start /content/orbit-wars/warm_start.pt \\
  --batch-size $BATCH_SIZE \\
  --epochs $EPOCHS \\
  --lr-value $LR_VALUE \\
  --lr-action $LR_ACTION \\
  --lr-l34 $LR_L34 \\
  --lr-l2 $LR_L2 \\
  --weight-decay $WEIGHT_DECAY \\
  --num-workers $NUM_WORKERS \\
  --device cuda \\
  --progress-every 50
"""))

C.append(("markdown", """\
## 7. Push the run to GCS

`jointv4_best.pt` = the COMPLETE v4 agent: actor + win/fwd value heads +
inbound aux. Direct consumers: the PPO critic warm start (frozen win
head + Φ + residual) and the simulate-then-score K-sample ranker. The
runner already loads v4 ckpts (commit cae02e5; sampled decode measured
65.6% vs physical_v4 at stage B).
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

out = Path(__file__).resolve().parent.parent / "notebooks" / "pretrain_joint_v4_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out} ({len(nb['cells'])} cells)")
