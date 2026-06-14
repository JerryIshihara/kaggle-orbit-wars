"""One-shot builder for notebooks/pretrain_joint_single_target_colab.ipynb.

Stage-B JOINT single-target pretrain: the single-target actor (select CE +
LogitNormal alloc + frac_sigma confidence) + value heads (win/fwd/contrast/
inbound) + the v5 Q head (dense doomed label), all trained together so the
checkpoint deploys in BOTH det and qrank modes.

Run: .venv/bin/python scripts/_build_joint_single_target_notebook.py
"""

import json
from pathlib import Path

C = []  # (cell_type, source)

C.append(("markdown", """\
# Stage B — JOINT single-target: actor + value + Q + confidence (top-meta 300)

Fuses three pretrains so the produced checkpoint can deploy det AND qrank:

* **ACTOR** (single-target): per-source softmax SELECT over [targets ... HOLD]
  + LogitNormal sigmoid ALLOC with a learned per-source **confidence**
  (`frac_sigma`). The meta is 100% single-target/source — full commitment.
* **VALUE**: win + fwd signals + temporal contrast + per-slot inbound aux, on
  the t+Δ paired CROSS cache (warm-start strips value heads -> fresh -> trained).
* **Q** (v5, per-pair): the anti-doomed gate. DENSE label = every plan_launch-
  DOOMED legal pair -> `Q_DOOMED` (precomputed sidecar, aligned 1:1 to the pair
  cache); expert-fired pairs -> +1 (a constant MC win proxy). The strategic TD
  signal comes later from PPO rollout.

Warm start = the stage-A merged dual-L2 trunk (`joint_warm_merged.pt`); the
actor/value/Q/sigma heads all (re)train here.

Output `jst_best.pt` stamps `single_target_per_source_logitnormal_v5` +
`with_value_heads/with_q_head/with_frac_sigma = True`. Watch per epoch:
ACTOR launch_acc/launch_recall + sigma; **Q sep** (doomed below reach => NEG);
VALUE holdout win_acc.
"""))

C.append(("markdown", "## 1. Authenticate + pull bundle from GCS"))

C.append(("code", """\
from google.colab import auth
auth.authenticate_user()
BUCKET = 'gs://orbit-wars-shipping/entity'
PAIR_CACHE_PREFIX  = 'pair_cache_topmeta300'
CROSS_CACHE_PREFIX = 'cross_cache_topmeta300_p1'
# Stage-A merged trunk (dual L2 + player tokens, NO action/value/Q heads —
# they all train fresh here). Same warm start as the actor-only single-target.
WARM_START_SRC = (f'{BUCKET}/runs/l2only_dualfuse_d256_T18u_20260612-055715/'
                  f'joint_warm_merged.pt')
# v5 Q-head DENSE doomed label, precomputed over the replays (single object,
# ~100 MB). Aligned 1:1 to the pair cache's (episode, turn) + slot layout.
DOOMED_SIDECAR_SRC = f'{BUCKET}/runs/topmeta300_pair_doomed.pt'
print(f'pulling from {BUCKET}\\n  action={PAIR_CACHE_PREFIX}'
      f'\\n  value={CROSS_CACHE_PREFIX}\\n  warm-start={WARM_START_SRC}'
      f'\\n  doomed={DOOMED_SIDECAR_SRC}')
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
        return
    raise RuntimeError(f'no cache manifest for {prefix} on {BUCKET}')

assert _gcs_size(WARM_START_SRC) is not None, f'{WARM_START_SRC} missing'
assert _gcs_size(DOOMED_SIDECAR_SRC) is not None, f'{DOOMED_SIDECAR_SRC} missing'
t0 = time.time()
_cp(f'{BUCKET}/code.tgz', WORK/'code.tgz')
_cp(f'{BUCKET}/weights.tgz', WORK/'weights.tgz')
pull_cache(PAIR_CACHE_PREFIX, WORK/'pair_cache.pt')
pull_cache(CROSS_CACHE_PREFIX, WORK/'cross_entity_cache.pt')
_cp(WARM_START_SRC, WORK/'warm_start.pt')
_cp(DOOMED_SIDECAR_SRC, WORK/'pair_doomed.pt')
print(f'pull done in {time.time()-t0:.1f}s')
"""))

C.append(("code", """\
# Wipe stale extracted code; leave the big caches + sidecar alone.
!rm -rf agents scripts ckpts
!find . -maxdepth 1 -name '*.pt' ! -name 'pair_cache.pt' ! -name 'cross_entity_cache.pt' ! -name 'warm_start.pt' ! -name 'pair_doomed.pt' -delete
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
## 2. Verify wiring — all four heads build + warm ckpt loads

Asserts the stage-A trunk loads through the adapter with ONLY the fresh heads
missing (value / frac_sigma / q_head + player machinery), and the
single-target alloc NLL + Q dense loss are differentiable.
"""))

C.append(("code", """\
import torch, agents
from agents.transformer_v3 import EntityPretrainModelV3, adapt_v2_state_dict, N_UNION
assert 'Minimal' in (agents.__doc__ or ''), 'stale agents shim — restart kernel'

ck = torch.load('/content/orbit-wars/warm_start.pt', map_location='cpu',
                weights_only=False)
cfg = ck['config']
assert cfg.get('arch') == 'dual_rate_l2_v3' and cfg.get('n_steps') == N_UNION, cfg
m = EntityPretrainModelV3(d_model=256,
                          conditioner_n_layers=int(cfg['conditioner_n_layers']),
                          head_n_layers=int(cfg['head_n_layers']),
                          with_consolidator=True, with_value_heads=True,
                          value_dropout=0.1, with_short_aux=True,
                          with_frac_sigma=True, with_q_head=True)
assert m.frac_sigma_head is not None and m.pair_head.q_head is not None
assert m.value_heads is not None
sd = {k: v for k, v in ck['model'].items() if not k.startswith('value_heads.')}
res = m.load_state_dict(adapt_v2_state_dict(sd), strict=False)
fresh_ok = ('value_heads.', 'frac_sigma_head.', 'pair_head.q_head.',
            'short_heads.', 'cross.fuse_player', 'cross.owner_proj',
            'cross.long.player_tokens', 'cross.short.player_tokens')
bad = [k for k in res.missing_keys if not k.startswith(fresh_ok)]
assert not bad, f'backbone skew: {bad[:6]}'
for tag in ('value_heads.', 'frac_sigma_head.', 'pair_head.q_head.'):
    assert any(k.startswith(tag) for k in res.missing_keys), f'{tag} not fresh'
print('warm ckpt OK: stage-A trunk loaded; fresh =',
      sorted({k.split('.')[0] for k in res.missing_keys}))

# Q dense loss + single-target alloc smoke (differentiable)
from agents.transformer_v3.q_head import compute_q_loss, Q_DOOMED
P = 5
q = torch.randn(2, P, P, requires_grad=True)
fired = torch.zeros(2, P, P, dtype=torch.bool); fired[0, 1, 2] = True
doomed = torch.zeros(2, P, P, dtype=torch.bool); doomed[0, 1, 3] = True
legal = torch.ones(2, P, P, dtype=torch.bool)
ql, qd = compute_q_loss(q, fired_mask=fired, fired_return=torch.ones(2),
                        doomed_mask=doomed, legal_mask=legal)
ql.backward(); assert q.grad.abs().sum() > 0
print(f'Q smoke OK: loss={float(ql):.3f} doomed_mean={qd[\"q_doomed_mean\"]:.2f} '
      f'(target Q_DOOMED={Q_DOOMED})')
del ck, m, sd
"""))

C.append(("markdown", "## 3. GPU check"))

C.append(("code", """\
import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(cpu)')
print('PairHead (the P x P cell MLP incl. Q head) runs every actor step now; '
      'BATCH_SIZE=16 is safe, 24 usually fits on A100.')
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
## 5. Cache preflight — pair + cross + doomed sidecar

Confirms all three data sources load and the doomed sidecar overlaps the pair
cache's acted keys (slot/key alignment) before the long run.
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
acted_keys = {payload['keys'][i] for i in payload['acted_indices']}
print(f"pair cache OK: {len(payload['snapshots'])} snapshots, "
      f"{len(payload['acted_indices'])} acted rows")

# doomed sidecar overlap with acted keys (alignment sanity)
doomed = torch.load('/content/orbit-wars/pair_doomed.pt', weights_only=False)
dkeys = set(doomed['doomed'].keys())
overlap = len(dkeys & acted_keys)
print(f"doomed sidecar: {len(dkeys)} non-empty entries (P={doomed.get('P')}); "
      f"{overlap} overlap acted keys ({100*overlap/max(1,len(dkeys)):.0f}% of sidecar)")
assert overlap > 0, 'doomed sidecar keys do NOT match pair cache acted keys!'

from agents.transformer_v2.pretrain.cross_entity import CachedCrossEntitySnapshotDataset
_cds = CachedCrossEntitySnapshotDataset('/content/orbit-wars/cross_entity_cache.pt')
_s = _cds[len(_cds)//2]
assert float(_s['player_valid'].sum()) > 0, 'cross cache all-zero player_valid'
print(f"cross cache OK: {len(_cds)} snapshots, player_valid live")
del _cds, _s, payload, snap, doomed
"""))

C.append(("markdown", "## 6. Train (joint: single-target actor + value + Q + confidence)"))

C.append(("code", """\
import time
TS = time.strftime('%Y%m%d-%H%M%S')
RUN_TAG = f'jst_v5_topmeta300_d256_{TS}'
OUT_DIR = f'data/runs/joint/{RUN_TAG}'
BATCH_SIZE    = 16
EPOCHS        = 25
LR_VALUE      = 1e-4   # value trunk + win/fwd + inbound aux (fresh)
LR_ACTION     = 1e-4   # PairHead select/frac + short aux (single-target re-fit)
LR_Q          = 1e-4   # fresh Q head — own FAST group (clean dense label)
LR_SIGMA      = 5e-6   # fresh frac_sigma confidence head — own SLOW group
LR_L34        = 5e-5   # dual_role + joint_role
LR_L2         = 1e-5   # stage-A dual L2 — gentle
WEIGHT_DECAY  = 1e-4
LAUNCH_WEIGHT = 8.0    # up-weight launch rows so HOLD doesn't drown launch_acc
                       # (matches the actor-only fine-tune that competed well)
ALLOC_WEIGHT  = 1.0
Q_COEF        = 1.0    # weight of the Q dense-doomed loss
VALUE_COEF    = 1.0
NUM_WORKERS   = 2
# Watch per epoch (val side):
#   ACTOR launch_acc / launch_recall  — exact target & decide-to-launch
#   ACTOR sigma_mean                  — confidence (should FALL, sharpening)
#   Q sep  (doomed - reach)           — must go NEGATIVE (doomed ranks below)
#   VALUE win_acc                     — critic anchor honesty
print('RUN_TAG =', RUN_TAG)
"""))

C.append(("code", """\
!python -u -m agents.transformer_v3.joint_single_target_pretrain \\
  --out-dir $OUT_DIR \\
  --pair-cache-path $PAIR_CACHE_PATH \\
  --cross-cache-path /content/orbit-wars/cross_entity_cache.pt \\
  --doomed-sidecar /content/orbit-wars/pair_doomed.pt \\
  --fleet-run-dir $FLEET_RUN_DIR \\
  --planet-run-dir $PLANET_RUN_DIR \\
  --comet-run-dir $COMET_RUN_DIR \\
  --warm-start /content/orbit-wars/warm_start.pt \\
  --batch-size $BATCH_SIZE --epochs $EPOCHS \\
  --lr-value $LR_VALUE --lr-action $LR_ACTION --lr-q $LR_Q \\
  --lr-sigma $LR_SIGMA --lr-l34 $LR_L34 --lr-l2 $LR_L2 \\
  --weight-decay $WEIGHT_DECAY \\
  --launch-weight $LAUNCH_WEIGHT --alloc-weight $ALLOC_WEIGHT \\
  --q-coef $Q_COEF --value-coef $VALUE_COEF \\
  --num-workers $NUM_WORKERS --device cuda --progress-every 50
"""))

C.append(("markdown", """\
## 7. Push the run to GCS

`jst_best.pt` = the COMPLETE v5 single-target agent: actor (select + LogitNormal
alloc + frac_sigma confidence) + value heads + Q head. Deploys in BOTH det
(sigmoid loc) and qrank (sample_frac + Q-gate) modes; also the PPO critic +
Q warm start.
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

out = (Path(__file__).resolve().parent.parent / "notebooks"
       / "pretrain_joint_single_target_colab.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out} ({len(nb['cells'])} cells)")
