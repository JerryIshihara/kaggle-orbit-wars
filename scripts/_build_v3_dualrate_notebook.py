"""One-shot builder for notebooks/pretrain_v3_dualrate_topmeta_colab.ipynb.

transformer_v3 (dual-rate L2) joint pretrain on the TOP-META pair cache.
Run: .venv/bin/python scripts/_build_v3_dualrate_notebook.py
"""

import json
from pathlib import Path

C = []  # (cell_type, source)

C.append(("markdown", """\
# transformer_v3 pretrain — dual-rate L2 (long T=10@5 ∥ short T=10@2), top-meta data

Trains **`EntityPretrainModelV3`**: the v2 stack with L2 split into two
parallel `CrossEntityAttention` branches over ONE 18-frame **union** history
stack (offsets 45,40,35,30,25,20,18,16,15,14,12,10,8,6,5,4,2,0):

* **L2-LONG** — T=10 @ stride 5 (offsets 45…0), exact copy of the v2 window.
  Warm-started 1:1 from the v2 checkpoint.
* **L2-SHORT** — T=10 @ stride 2 (offsets 18…0), fine recency (launch
  cadence, dodge/intercept under the 1.30.x swept-collision physics).
* **Zero-init fusion** `[I|0] Linear(512→256)` (tokens + CLS separately):
  at init the fused output IS the long branch, so epoch 0 starts exactly at
  the warm-started v2 model. Downstream (consolidator/L3/L4/PairHead/value
  heads) is unchanged and keeps its warm start.
* **Short-horizon aux tasks** (`--short-aux-weight 0.5`) train the SHORT
  branch from step 0 past the fusion gate, reading its PRE-fusion tokens:
  `owner_t_plus_5` CE, `log_ships_t_plus_5` Huber, per-player
  `ships_arriving_within_5` Huber, `earliest_arrival_owner_slot` CE.
  Main select/alloc/value losses are unchanged (v2 multinomial-alloc recipe).

Data = **top-meta June experts** (Jake Will / Hober Malloc / M&J&M.ver2,
200 episodes): single-target per source 100%, ~98% commitment, HOLD≈0.
Union-T18 coverage was probed locally at **100%** — the restack is a
load-time metadata override, no cache rebuild.

**Output `joint_best.pt` is NOT deploy-ready yet**: the runner/PPO arch
adaptation for `arch=dual_rate_l2_v3` (union-offset deque walk + L2 swap at
load) is a separate pending step. Judge this run by pretrain metrics
(`act/*`, `val/win_acc` holdout gap, `sh5/*`).
"""))

C.append(("markdown", "## 1. Authenticate + pull bundle from GCS"))

C.append(("code", """\
from google.colab import auth
auth.authenticate_user()
BUCKET = 'gs://orbit-wars-shipping/entity'
# Action cache (pair): TOP-META June experts (Jake Will #1 / Hober Malloc #2 /
# M&J&M.ver2 #3; 200 episodes, 2026-06-10 daily), f512, EVERY turn stored
# (the union T=18 restack happens at load), per-cell pair_ships.
PAIR_CACHE_PREFIX = 'pair_cache_topmeta_jun10'
# Value cache: cap150 cross-entity cache with P1 future-level labels +
# split_stems (by-game holdout for the win-memorization gap log).
CROSS_CACHE_PREFIX = 'cross_cache_joint_cap150_p1'
# Warm start: the proven v2 multinomial-alloc base (82.5% vs physical_v4).
# The v3 driver key-maps cross.* onto BOTH branches automatically.
# >>> When the topmeta v2 pretrain finishes, SWAP to its joint_best: <<<
#   WARM_START_SRC = f'{BUCKET}/runs/joint_topmeta_d256_T10_head3_<TS>/joint_best.pt'
WARM_START_SRC = f'{BUCKET}/runs/joint_mt_alloc_d256_T10_head3_20260610-164152/joint_best.pt'
print(f'pulling from {BUCKET}\\n  action={PAIR_CACHE_PREFIX}  value={CROSS_CACHE_PREFIX}'
      f'\\n  warm-start={WARM_START_SRC}')
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
        import shutil as _sh; _sh.rmtree(cdir, ignore_errors=True)  # free chunk disk
        print(f'  freed chunk dir {cdir.name}')
        return
    for cand in (f'{prefix}.pt', f'{prefix}'):
        if _gcs_size(f'{BUCKET}/{cand}') is not None:
            sz = _cp(f'{BUCKET}/{cand}', dst, force=not dst.exists())
            print(f'  {dst.name}: {sz/1024**3:.2f} GB (single)'); return
    raise RuntimeError(f'no cache for prefix {prefix} on {BUCKET}')

assert _gcs_size(WARM_START_SRC) is not None, (
    f'{WARM_START_SRC} missing on GCS — check the runs/ folder')
t0 = time.time()
_cp(f'{BUCKET}/code.tgz', WORK/'code.tgz')
_cp(f'{BUCKET}/weights.tgz', WORK/'weights.tgz')
pull_cache(PAIR_CACHE_PREFIX, WORK/'pair_cache.pt')
pull_cache(CROSS_CACHE_PREFIX, WORK/'cross_entity_cache.pt')
_cp(WARM_START_SRC, WORK/'warm_start.pt')
print(f'pull done in {time.time()-t0:.1f}s')
"""))

C.append(("code", """\
# Wipe stale extracted code; leave the big caches alone.
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
## 2. Verify wiring — dual-rate L2, zero-init fusion, aux heads, contract loss

Builds the exact model `--arch v3` will build, then asserts the four design
properties: (a) v2→v3 warm-start key mapping is exact (missing = fusion +
fresh aux heads only), (b) fusion is zero-init `[I|0]`, (c) the alloc-CE
gradient reaches the HOLD diagonal (contract v2 unchanged), (d) the aux
heads exist and the union offsets are the expected 18 frames.
"""))

C.append(("code", """\
import math, torch, agents
from agents.transformer_v2.pretrain.entity_encoder import (
    EntityPretrainModel, compute_multi_loss,
    _PLANET_OWNER_START_IDX, _PLANET_SHIPS_LOG_FEATURE_IDX,
)
from agents.transformer_v2.featurizer.fleet_featurizer import SHIPS_LOG_MAX
from agents.transformer_v2.pretrain.value_heads import ValuePretrainHeads
from agents.transformer_v2.pretrain import joint_pretrain, alloc_labels
from agents.transformer_v3 import (
    EntityPretrainModelV3, adapt_v2_state_dict,
    UNION_HISTORY_OFFSETS, N_UNION,
)
assert 'Minimal' in (agents.__doc__ or ''), 'stale agents shim — restart kernel'
assert N_UNION == 18 and UNION_HISTORY_OFFSETS[-1] == 0, UNION_HISTORY_OFFSETS

m = EntityPretrainModelV3(d_model=256,
                          conditioner_n_layers=3, head_n_layers=3,
                          with_consolidator=True, with_value_heads=True)
for a in ('entity','cross','dual_role','joint_role','pair_head','consolidator',
          'value_heads','short_heads'):
    assert getattr(m, a) is not None, f'missing {a} (stale code.tgz)'
assert isinstance(m.value_heads, ValuePretrainHeads)
# (b) zero-init fusion: [I | 0], zero bias
W = m.cross.fuse_tokens.weight.detach()
assert torch.equal(W[:, :256], torch.eye(256)) and W[:, 256:].abs().sum() == 0
assert m.cross.fuse_tokens.bias.detach().abs().sum() == 0
rep = m.freeze_below_l2()
assert not any(p.requires_grad for p in m.entity.parameters()), 'L1 must be frozen'
assert all(p.requires_grad for p in m.cross.parameters()), 'dual L2 must be trainable'
assert all(p.requires_grad for p in m.short_heads.parameters()), 'aux heads must train'

# (a) warm-start mapping against the REAL ckpt
ck = torch.load('/content/orbit-wars/warm_start.pt', map_location='cpu', weights_only=False)
cfg = ck['config']
assert cfg['head_n_layers'] == 3 and cfg['conditioner_n_layers'] == 3, cfg
assert cfg['d_model'] == 256 and cfg['n_steps'] == 10, cfg
sd = {k: v for k, v in ck['model'].items() if not k.startswith('value_heads.')}
res = m.load_state_dict(adapt_v2_state_dict(sd), strict=False)
bad = [k for k in res.missing_keys
       if not k.startswith(('cross.fuse_', 'short_heads.', 'value_heads.'))]
assert not bad, f'backbone skew: {bad[:6]}'
assert not res.unexpected_keys, res.unexpected_keys[:6]
assert torch.equal(m.cross.long.step_embed, ck['model']['cross.step_embed'])
print(f"warm-start mapping OK: long branch exact; short branch copied w/ "
      f"step-embed offset remap; fusion zero-init; fresh: value+aux heads")

# (c) contract smoke: alloc CE reaches the HOLD diagonal (unchanged v2 loss)
P = 4
pf = torch.zeros(1, P, _PLANET_SHIPS_LOG_FEATURE_IDX + 2)
pf[0, 0, _PLANET_OWNER_START_IDX] = 1.0
pf[0, 0, _PLANET_SHIPS_LOG_FEATURE_IDX] = math.log1p(100) / SHIPS_LOG_MAX
mask = torch.ones(1, P, dtype=torch.bool)
labels = torch.zeros(1, P, P, dtype=torch.bool); labels[0, 0, 2] = True
ships = torch.zeros(1, P, P, dtype=torch.int32); ships[0, 0, 2] = 30
valid = mask.unsqueeze(2) & mask.unsqueeze(1)
valid[:, range(P), range(P)] = False
batch = dict(pair_labels=labels, pair_valid=valid, pair_ships=ships,
             planet_features=pf, planet_mask=mask)
preds = {'pair_logits': torch.randn(1, P, P, requires_grad=True),
         'pair_frac':  torch.randn(1, P, P, requires_grad=True)}
total, per = compute_multi_loss(preds, batch, multinomial_alloc=True)
total.backward()
assert abs(per['hold_share_label'] - 0.70) < 1e-4, per['hold_share_label']
assert preds['pair_frac'].grad[0, 0, 0].abs() > 0, 'HOLD (frac diagonal) got no gradient!'
assert preds['pair_logits'].grad.diagonal(dim1=-2, dim2=-1).abs().sum() == 0
print(f"contract v2 smoke OK: alloc_ce={per['pair_frac']:.4f} "
      f"hold_label={per['hold_share_label']:.3f} (expect 0.700)")
n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'trainable (L2+ incl. dual branches/fusion/aux): {n_tr:,}')
del ck, m, sd
"""))

C.append(("markdown", "## 3. GPU check"))

C.append(("code", """\
import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(cpu)')
print('NOTE: union T=18 costs ~1.8x L0/L1 + 2x L2 vs the v2 T=10 runs. '
      'BATCH_SIZE=16 fits A100/L4; on T4 drop to 8-12 if you see OOM.')
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
## 5. Cache paths + preflight

Pair cache must carry `pair_ships` + `acted_indices` + the t+5 label family
(the short-branch aux targets); the cross cache must carry the P1
future-level labels + by-game split. The union T=18 restack itself is
asserted inside `train_joint` (probe + ≥50% context-coverage gate; this
cache probed 100% locally).
"""))

C.append(("code", """\
PAIR_CACHE_PATH  = '/content/orbit-wars/pair_cache.pt'
CROSS_CACHE_PATH = '/content/orbit-wars/cross_entity_cache.pt'
import os, torch
for p in (PAIR_CACHE_PATH, CROSS_CACHE_PATH):
    assert os.path.exists(p), p
    print(f'{p}  {os.path.getsize(p)/1024**3:.2f} GB')

payload = torch.load(PAIR_CACHE_PATH, map_location='cpu', weights_only=False, mmap=True)
assert 'acted_indices' in payload, 'old-style pair cache (no acted_indices) — rebuild'
snap = payload['snapshots'][payload['acted_indices'][0]]
assert 'pair_ships' in snap, 'pair cache lacks pair_ships — rebuild with 3-tuple edges'
for k in ('owner_t_plus_5','log_ships_t_plus_5','valid_t_plus_5',
          'ships_arriving_within_5','earliest_arrival_owner_slot'):
    assert k in snap, f'pair cache lacks short-aux label {k}'
print(f"pair cache OK: {len(payload['snapshots'])} stored frames, "
      f"{len(payload['acted_indices'])} acted rows, t+5 aux labels present, "
      f"stored offsets={tuple(payload['config'].get('history_offsets') or ())} "
      f"(train_joint restacks to the T=18 union)")
del payload, snap

from agents.transformer_v2.pretrain.cross_entity import CachedCrossEntitySnapshotDataset
from agents.transformer_v2.pretrain.value_signals import P1_FWD_HORIZONS, N_P1_SIGNALS
_ds = CachedCrossEntitySnapshotDataset(CROSS_CACHE_PATH)
_need = ['p1_now','p1_future','p1_valid','p1_back','valid_back','survives_future']
_miss = [c for c in _need if c not in _ds.columns]
assert not _miss, f'cross cache missing P1 label cols {_miss}'
_s = _ds[0]
_fut = _s['p1_future']
assert tuple(_fut.shape[-2:]) == (N_P1_SIGNALS, len(P1_FWD_HORIZONS)), _fut.shape
print(f"cross cache schema OK: {_ds.columns['p1_future'].shape[0]} snapshots")
del _ds, _s  # release the mmap handle before training
"""))

C.append(("markdown", """\
## 6. Label inspection — REVIEW BEFORE TRAINING

Reference values from the 2026-06-12 local scan of THIS top-meta cache:

* **anomaly counters all 0** (`dropped_rows_ships0`, `dropped_rows_no_src`,
  `overflow_rows`);
* **HOLD share** mean ≈0.016, ==0 in ~96.8%, >0.5 in ~1.4% — top players
  commit ~98% of launchable ships when they fire;
* **fired targets/acting row: 100% single-target** (multi-launch = several
  SOURCES per step, never one source splitting);
* select sparsity: ~33% of owned rows act, ~0.011 positive-cell rate
  (`pair_pos_weight=600` compensates).
"""))

C.append(("code", """\
!python -u scripts/show_multi_target_labels.py \\
  --cache $PAIR_CACHE_PATH --n-snapshots 1500 --n-examples 3
"""))

C.append(("markdown", "## 7. Train (dual-rate L2, short-aux, multinomial-alloc contract)"))

C.append(("code", """\
import time
TS = time.strftime('%Y%m%d-%H%M%S')
RUN_TAG = f'joint_v3dual_d256_T18u_head3_{TS}'
OUT_DIR = f'data/runs/joint/{RUN_TAG}'
# hyperparams (v2 recipe + the two v3 flags in the command below)
D_MODEL              = 256
HEAD_N_LAYERS        = 3
CONDITIONER_N_LAYERS = 3
BATCH_SIZE, EPOCHS   = 16, 35   # T4: drop BATCH_SIZE to 8-12 on OOM
LR, WEIGHT_DECAY     = 5e-5, 1e-4
PAIR_POS_WEIGHT      = 600.0
ALLOC_WEIGHT         = 1.0
VALUE_COEF           = 1.0
VALUE_DROPOUT        = 0.1
SHORT_AUX_WEIGHT     = 0.5      # the 4 t+5 forecast tasks on the SHORT branch
NUM_WORKERS          = 2
WARM_START_PATH      = '/content/orbit-wars/warm_start.pt'
# Watch in the per-head logs:
#   sh5/owner_acc, sh5/earliest_acc = is the SHORT branch learning dynamics
#                                     (rises past majority-class = yes)
#   sh5/owner_t5, sh5/ships_t5      = fall fast then flatten
#   act/pair_logits                 = select BCE — stays near warm-start level
#   act/hold_mae                    = alloc calibration (<~0.05 good; this
#                                     cache's HOLD is near-zero everywhere)
#   val/win_acc + HOLDOUT gap       = same anti-memorization read as v2
print('RUN_TAG =', RUN_TAG)
"""))

C.append(("code", """\
# Run via the ! shell magic with `-u` (unbuffered) so the per-head logs
# STREAM LIVE into the cell. After "[joint] action cache:" verify the
# UNION restack lines: stored offsets -> OVERRIDING to (45,40,...,2,0)
# T=18, coverage ~100%, probe item planet_features=(18, 64, 138).
!python -u -m agents.transformer_v2.pretrain.joint_pretrain \\
  --arch v3 \\
  --short-aux-weight $SHORT_AUX_WEIGHT \\
  --out-dir $OUT_DIR \\
  --fleet-run-dir $FLEET_RUN_DIR \\
  --planet-run-dir $PLANET_RUN_DIR \\
  --comet-run-dir $COMET_RUN_DIR \\
  --pair-cache-path $PAIR_CACHE_PATH \\
  --cross-cache-path $CROSS_CACHE_PATH \\
  --d-model $D_MODEL \\
  --head-n-layers $HEAD_N_LAYERS \\
  --conditioner-n-layers $CONDITIONER_N_LAYERS \\
  --multinomial-alloc \\
  --pair-pos-weight $PAIR_POS_WEIGHT \\
  --alloc-weight $ALLOC_WEIGHT \\
  --batch-size $BATCH_SIZE \\
  --epochs $EPOCHS \\
  --lr $LR \\
  --weight-decay $WEIGHT_DECAY \\
  --value-coef $VALUE_COEF \\
  --value-dropout $VALUE_DROPOUT \\
  --warm-start $WARM_START_PATH \\
  --num-workers $NUM_WORKERS \\
  --device cuda \\
  --progress-every 50
"""))

C.append(("markdown", """\
## 8. Push the trained run back to GCS

`joint_best.pt` carries `config.arch = dual_rate_l2_v3` + the union/long/
short offset tuples, so loaders can reconstruct the architecture. Two
follow-ups before this model can play:

1. **Runner/PPO v3-arch adaptation** (pending, tracked): union-offset deque
   walk + `EntityPretrainModelV3` dispatch at load. Until then this ckpt is
   judged on pretrain metrics only.
2. **Fusion-gate read**: if `sh5/*` metrics are strong but the fusion weights'
   short half stays ≈0 (branch learned but unused), revisit with a late-phase
   fused-tap aux or a fusion nudge.
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

out = Path(__file__).resolve().parent.parent / "notebooks" / "pretrain_v3_dualrate_topmeta_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out} ({len(nb['cells'])} cells)")
