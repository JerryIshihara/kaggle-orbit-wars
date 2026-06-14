"""One-shot builder for notebooks/pretrain_single_target_colab.ipynb.

Stage-B pretrain: the SINGLE-TARGET actor (contract v5) on the stage-A dual L2.
Near-exact mirror of scripts/_build_actor_stage_notebook.py (same GCS pull,
same L0 staging, same cache preflight) — swaps the contract/loss + smoke.
Run: .venv/bin/python scripts/_build_single_target_notebook.py
"""

import json
from pathlib import Path

C = []  # (cell_type, source)

C.append(("markdown", """\
# Stage B — single-target pretrain, contract v5 (LogitNormal alloc), top-meta 300

Trains the ACTOR on the stage-A dual L2: **L3/L4 + PairHead + the new
per-source `frac_sigma` (confidence) head + the t+5 short aux**, with
**per-part learning rates** (heads / L3+L4 / L2 / frac_sigma) and **value
heads excluded** (a later stage).

NEW actor design (`single_target_per_source_logitnormal_v5`):

* SELECT = per-source **softmax over the target columns INCLUDING the
  diagonal HOLD** — one target per source, or hold. (The v4 bounded-k
  Dirichlet that let a source split across 2-3 targets is REVERTED — the
  top meta is 100% single-target per source; splitting dilutes commitment.)
* ALLOC = **LogitNormal-sigmoid** with a learned per-source **sigma**
  (confidence): `f = sigmoid(Normal(pair_frac, sigma))`. The `FracSigmaHead`
  predicts log-sigma off the L4 source tokens; deploy/PPO SAMPLE the fraction
  from it (sigma replaces the old hardcoded `--sigma`, exactly as alpha0 did
  for Dirichlet).
* pretrain tasks: single-target **select CE** (optional launch up-weight) +
  **LogitNormal NLL** of the expert's observed fraction-of-source ships
  (teaches loc AND sigma jointly) + t+5 short-branch aux (x0.5).

Data = **top-meta 300** (Jake Will 194 / Hober Malloc 68 / TonyK 25 /
M&J&M 13; Jun-10 + Jun-11 dailies, wins-first) — 100% single-target, so the
single-target labels are clean. Warm start = stage-A `joint_warm_merged.pt`
(trained dual L2 + player tokens; L3/L4 from the v3dual joint run); the new
`frac_sigma_head` is fresh-init on its own slow LR group.

Watch (printed each val pass): `SELECT launch_recall` (did the model decide to
launch at all — the over-holding diagnostic) and `launch_acc` (exact target);
`ALLOC sigma_mean / sigma_p10` should FALL as the contract sharpens.
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
WARM_START_SRC = (f'{BUCKET}/runs/l2only_dualfuse_d256_T18u_20260612-055715/'
                  f'joint_warm_merged.pt')
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
## 2. Verify wiring — v3.1 player tokens, frac_sigma head, single-target smoke

Asserts: the warm checkpoint is v3-shaped and passes through the adapter
with only fresh keys missing (frac_sigma head / owner proj / player tokens /
fuse_player); the new `frac_sigma_head` is fresh-init; player tokens are
invisible to the planet/global path. Then a tiny single-target loss smoke
(select CE + LogitNormal alloc NLL) confirms the contract pieces wire and
backward reaches the confidence head.
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
                          with_short_aux=True, with_frac_sigma=True)
assert m.consolidator is None and m.frac_sigma_head is not None
sd = {k: v for k, v in ck['model'].items()
      if not k.startswith('value_heads.')}
res = m.load_state_dict(adapt_v2_state_dict(sd), strict=False)
fresh_ok = ('frac_sigma_head.', 'short_heads.', 'cross.fuse_player',
            'cross.owner_proj', 'cross.long.player_tokens',
            'cross.short.player_tokens')
bad = [k for k in res.missing_keys if not k.startswith(fresh_ok)]
assert not bad, f'backbone skew: {bad[:6]}'
assert any(k.startswith('frac_sigma_head.') for k in res.missing_keys), \\
    'frac_sigma_head should be fresh-init (missing from warm ckpt)'
# stage-A player machinery must LOAD (the merge carries it), not re-init
assert not [k for k in res.missing_keys if 'player_tokens' in k
            or 'owner_proj' in k or 'fuse_player' in k], 'stage-A L2 not loaded'
assert torch.equal(m.cross.long.step_embed, ck['model']['cross.long.step_embed'])
print('warm ckpt OK: stage-A L2 + player tokens loaded; fresh = frac_sigma '
      '+ short aux heads')

# Single-target loss smoke on the real label-builder parameterization.
import math as _m
from agents.transformer_v2.pretrain.entity_encoder import (
    _pair_single_target_ce, _pair_frac_targets_from_batch,
    _PLANET_OWNER_START_IDX, _PLANET_SHIPS_LOG_FEATURE_IDX,
)
from agents.transformer_v2.featurizer.fleet_featurizer import SHIPS_LOG_MAX
from agents.transformer_v3.single_target_alloc import single_target_alloc_nll
from agents.transformer_v3.single_target_alloc import FracSigmaHead
P = 4
labels = torch.zeros(1, P, P, dtype=torch.bool); labels[0, 0, 2] = True
ships = torch.zeros(1, P, P, dtype=torch.int32); ships[0, 0, 2] = 30
valid = torch.ones(1, P, P, dtype=torch.bool); valid[:, range(P), range(P)] = False
pf = torch.zeros(1, P, _PLANET_SHIPS_LOG_FEATURE_IDX + 2)
pf[0, 0, _PLANET_OWNER_START_IDX] = 1.0
pf[0, 0, _PLANET_SHIPS_LOG_FEATURE_IDX] = _m.log1p(100) / SHIPS_LOG_MAX
batch = dict(pair_labels=labels, pair_valid=valid, pair_ships=ships,
             planet_features=pf, planet_mask=torch.ones(1, P, dtype=torch.bool))
logits = torch.randn(1, P, P, requires_grad=True)
frac = torch.randn(1, P, P, requires_grad=True)
sig_head = FracSigmaHead(8)
sigma_log = sig_head(torch.randn(1, P, 8))
ce, chosen, st = _pair_single_target_ce(logits, batch, labels.bool(), valid.bool())
ft = _pair_frac_targets_from_batch(batch, ships.float())
nll, at = single_target_alloc_nll(frac, frac_sigma_logs=sigma_log,
                                  chosen_mask=chosen, frac_targets=ft)
total = ce + nll
total.backward()
assert frac.grad.abs().sum() > 0 and logits.grad.abs().sum() > 0
assert sig_head.net[-1].weight.grad.abs().sum() > 0, 'no grad into frac_sigma'
print(f"single-target smoke OK: sel_ce={st['pair_ce']:.3f} "
      f"alloc_nll={at['alloc_nll']:.3f} sigma_mean={at['sigma_mean']:.3f} "
      f"frac_mae={at['frac_mae']:.3f} launch_recall={st['launch_recall']}")
del ck, m, sd
"""))

C.append(("markdown", "## 3. GPU check"))

C.append(("code", """\
import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(cpu)')
print('Per-step cost is dominated by L0/L1 over the 18-frame union; the '
      'single-target select CE + LogitNormal NLL are cheap. BATCH_SIZE=16 is '
      'safe; 24 usually fits on A100/L4.')
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

The driver restacks to the 18-frame union itself; this cell just confirms the
file + the t+5 short-aux labels + `pair_ships` (single-target frac labels)
before the long run starts.
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
      f"{len(payload['acted_indices'])} acted rows, pair_ships + t+5 labels present")
del payload, snap
"""))

C.append(("markdown", "## 6. Train (single-target select CE + LogitNormal alloc)"))

C.append(("code", """\
import time
TS = time.strftime('%Y%m%d-%H%M%S')
RUN_TAG = f'single_target_v5_topmeta300_d256_{TS}'
OUT_DIR = f'data/runs/joint/{RUN_TAG}'
BATCH_SIZE    = 16
EPOCHS        = 25
LR_HEADS      = 1e-4   # PairHead + short aux heads
LR_L34        = 5e-5   # dual_role + joint_role
LR_L2         = 1e-5   # stage-A-pretrained dual L2 — gentle
LR_SIGMA      = 5e-6   # fresh FracSigmaHead (confidence) — slow, own group
WEIGHT_DECAY  = 1e-4
LAUNCH_WEIGHT = 4.0    # up-weight launch rows so the dominant HOLD rows don't
                       # drown the launch signal in the select CE
ALLOC_WEIGHT  = 1.0
NUM_WORKERS   = 2
# Watch in the logs (printed per epoch, val side):
#   SELECT launch_recall  — decided to launch at all (over-holding diagnostic)
#   SELECT launch_acc     — exact target hit
#   SELECT hold_prec/recall
#   ALLOC  sigma_mean/p10 — confidence; should FALL as the contract sharpens
#   ALLOC  frac_mae       — |sigmoid(loc) - expert frac|
print('RUN_TAG =', RUN_TAG)
"""))

C.append(("code", """\
!python -u -m agents.transformer_v3.single_target_pretrain \\
  --out-dir $OUT_DIR \\
  --pair-cache-path $PAIR_CACHE_PATH \\
  --fleet-run-dir $FLEET_RUN_DIR \\
  --planet-run-dir $PLANET_RUN_DIR \\
  --comet-run-dir $COMET_RUN_DIR \\
  --warm-start /content/orbit-wars/warm_start.pt \\
  --batch-size $BATCH_SIZE \\
  --epochs $EPOCHS \\
  --lr-heads $LR_HEADS \\
  --lr-l34 $LR_L34 \\
  --lr-l2 $LR_L2 \\
  --lr-sigma $LR_SIGMA \\
  --launch-weight $LAUNCH_WEIGHT \\
  --alloc-weight $ALLOC_WEIGHT \\
  --weight-decay $WEIGHT_DECAY \\
  --num-workers $NUM_WORKERS \\
  --device cuda \\
  --progress-every 50
"""))

C.append(("markdown", """\
## 7. Push the run to GCS

`single_target_best.pt` = the stage-B artifact: the v5 single-target actor
(per-source softmax select + LogitNormal alloc with a learned `frac_sigma`
confidence) on the stage-A dual L2. It stamps
`action_contract = single_target_per_source_logitnormal_v5` +
`with_frac_sigma = True`. Before it can play: (a) the value stage (win head for
the PPO critic), (b) runner/PPO v5 decode support (mean / sample-from-sigma
arms, gate-A/B'd).
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
       / "pretrain_single_target_colab.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out} ({len(nb['cells'])} cells)")
