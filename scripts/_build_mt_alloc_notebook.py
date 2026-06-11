"""One-shot builder for notebooks/pretrain_multitarget_alloc_T10_colab.ipynb.

Kept in scripts/ so the notebook can be regenerated after cell-source edits;
run: .venv/bin/python scripts/_build_mt_alloc_notebook.py
"""

import json
from pathlib import Path

C = []  # (cell_type, source)

C.append(("markdown", """\
# Multi-target actor pretrain — Bernoulli select + multinomial alloc (T=10)

Pretrains the **`bernoulli_select_multinomial_alloc_v2`** actor contract — the
one PPO samples and the deploy runner thresholds — so PPO starts with *every*
input of its action distribution supervised:

* **Stage 1 (select):** whole-grid BCE on `pair_logits` vs expert launch cells
  (unchanged from previous multi-target pretrains, `pos_weight≈600`).
* **Stage 2 (alloc, NEW):** per acting source, cross-entropy between the
  expert's empirical ship split and `softmax([frac_loc[s, F_s],
  frac_loc[s, s]])` — the **exact softmax PPO's allocation Multinomial
  uses** (`sampler.py::sample_multi_target`, contract **v2**: the self/HOLD
  logit is the frac head's own diagonal; v1 borrowed `pair_logits[s, s]`).
  This is the only loss that reaches the **HOLD diagonal**, which was *never
  supervised* before and is the prime suspect for the PPO lr cliff
  (frozen ↔ detonation). Select (pair_head) and alloc (pair_frac_head)
  gradients stay decoupled at the head level.

Key facts baked into this run:

* **T=10 strided** (`HISTORY_OFFSETS = (45,40,…,5,0)`): the pair cache stores
  every turn as single frames; `train_joint` force-restacks it to the model's
  T=10 at load time (probe + assert), closing the old T=6/T=10 mismatch with
  the cross cache.
* **Architecture = agent A's** (`head_n_layers=3, conditioner_n_layers=3,
  d256, d_pair=256`), warm-started from **`entity_head3_value_merged.pt`**
  (A's 87.5% pretrain base). Value heads are dropped + retrained fresh on the
  cap150 cross cache with the by-game holdout (the validated anti-memorization
  recipe).
* Labels were reviewed on the same Ebi cache via
  `scripts/show_multi_target_labels.py` (§6 reruns it here): HOLD share mean
  ≈0.07 (p50 0.01, p90 0.19), 12% of acting rows multi-target, 0 anomalies.

**After training:** `joint_best.pt` is directly consumable as the PPO base
(same layout as `entity_head3_value_merged.pt`). NOTE the deploy runner's
threshold mode still sizes launches `sigmoid(frac)·ships` — with this
contract `frac_loc` are softmax-share logits, so deploy eval needs the
alloc-softmax sizing mode in `runner.py` first (see §8).
"""))

C.append(("markdown", "## 1. Authenticate + pull bundle from GCS"))

C.append(("code", """\
from google.colab import auth
auth.authenticate_user()
BUCKET = 'gs://orbit-wars-shipping/entity'
# Action cache (pair): Ebi-only, max_fleets=512, ~120k stored frames (EVERY
# turn — needed for the strided T=10 restack), ~24.5k acted launch turns
# (training rows), WITH per-cell pair_ships (3-tuple edges).
PAIR_CACHE_PREFIX = 'pair_cache_ebi_acted'
# Value cache: cap150 cross-entity cache with P1 future-level labels +
# split_stems (by-game holdout for the win-memorization gap log).
CROSS_CACHE_PREFIX = 'cross_cache_joint_cap150_p1'
# Warm start: agent A's pretrain base (87.5% vs physical_v4) — head3/cond3/
# d256/T10, 'model'-keyed, value heads dropped + retrained fresh.
# If this object is missing, upload it once from machine A:
#   gcloud storage cp data/runs/ppo/baseline_lr5e5_enthalf_iter5_20260604/entity_head3_value_merged.pt \\
#       gs://orbit-wars-shipping/entity/entity_head3_value_merged.pt
WARM_START_SRC = f'{BUCKET}/entity_head3_value_merged.pt'
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
    f'{WARM_START_SRC} missing — upload entity_head3_value_merged.pt from '
    f'machine A first (command in the cell above)')
t0 = time.time()
_cp(f'{BUCKET}/code.tgz', WORK/'code.tgz')
_cp(f'{BUCKET}/weights.tgz', WORK/'weights.tgz')
pull_cache(PAIR_CACHE_PREFIX, WORK/'pair_cache.pt')
pull_cache(CROSS_CACHE_PREFIX, WORK/'cross_entity_cache.pt')
_cp(WARM_START_SRC, WORK/'warm_start.pt')  # agent A's pretrain base (~26 MB)
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
## 2. Verify wiring — deep heads + the new contract loss

Builds the exact model `train_joint` will build (head3/cond3), runs a tiny
synthetic batch through `compute_multi_loss(multinomial_alloc=True)`, and
asserts the gradient reaches the HOLD diagonal — the whole point of this
contract. Also checks the warm-start ckpt's architecture matches.
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
assert 'Minimal' in (agents.__doc__ or ''), 'stale agents shim — restart kernel'

m = EntityPretrainModel(d_model=256, n_steps=10,
                        conditioner_n_layers=3, head_n_layers=3,
                        with_consolidator=True, with_value_heads=True)
for a in ('entity','cross','dual_role','joint_role','pair_head','consolidator','value_heads'):
    assert getattr(m, a) is not None, f'missing {a} (stale code.tgz)'
assert isinstance(m.value_heads, ValuePretrainHeads)
rep = m.freeze_below_l2()
assert not any(p.requires_grad for p in m.entity.parameters()), 'L1 must be frozen'
assert all(p.requires_grad for p in m.cross.parameters()), 'L2 must be trainable'
assert hasattr(joint_pretrain, 'train_joint')

# --- contract smoke: src 0 owns 100 ships, launches 30 to planet 2 ---
P = 4; D = _PLANET_SHIPS_LOG_FEATURE_IDX + 2
pf = torch.zeros(1, P, D)
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
# v2: HOLD lives on the FRAC head's diagonal; the select head's diagonal is
# dead and must stay gradient-free (select/alloc decoupling).
assert preds['pair_frac'].grad[0, 0, 0].abs() > 0, 'HOLD (frac diagonal) got no gradient!'
assert preds['pair_logits'].grad.diagonal(dim1=-2, dim2=-1).abs().sum() == 0, \\
    'select-head diagonal must be untouched under contract v2'
assert 'ppo_source' not in per, 'legacy source-CE alignment must be skipped'
print(f"contract v2 smoke OK: alloc_ce={per['pair_frac']:.4f} "
      f"hold_label={per['hold_share_label']:.3f} (expect 0.700) — "
      f"HOLD grad on frac diagonal, select head decoupled")

ck = torch.load('/content/orbit-wars/warm_start.pt', map_location='cpu', weights_only=False)
cfg = ck['config']
assert cfg['head_n_layers'] == 3 and cfg['conditioner_n_layers'] == 3, cfg
assert cfg['d_model'] == 256 and cfg['n_steps'] == 10, cfg
n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'warm-start architecture matches (head3/cond3/d256/T10); trainable (L2+): {n_tr:,}')
del ck, m
"""))

C.append(("markdown", "## 3. GPU check"))

C.append(("code", """\
import torch
print('cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(cpu)')
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

Pair cache must carry `pair_ships` (per-cell launch sizes — the alloc-CE
labels) and `acted_indices`; the cross cache must carry the P1 future-level
labels + by-game split. The T=10 restack itself is asserted inside
`train_joint` (probe + ≥50% context-coverage gate).
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
print(f"pair cache OK: {len(payload['snapshots'])} stored frames, "
      f"{len(payload['acted_indices'])} acted rows, stored offsets="
      f"{tuple(payload['config'].get('history_offsets') or ())} "
      f"(train_joint restacks to T=10 strided)")
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

Streams worked examples + aggregate stats of exactly what §7 trains on.
What to check (reference values from the 2026-06-10 local scan of this cache):

* **anomaly counters all ~0** (`dropped_rows_ships0`, `dropped_rows_no_src`,
  `overflow_rows`) — non-zero spikes mean label/feature drift;
* **HOLD share** mean ≈0.07, p50 ≈0.01, p90 ≈0.19, ~49% exactly 0 — Ebi
  launches nearly everything; the diagonal learns *mostly-fire-with-a-tail*;
* **fired targets/acting row**: ~88% single-target, ~12% two-target;
* select sparsity: ~13% of owned rows act, ~0.005 positive-cell rate
  (that's what `pair_pos_weight=600` compensates).
"""))

C.append(("code", """\
!python -u scripts/show_multi_target_labels.py \\
  --cache $PAIR_CACHE_PATH --n-snapshots 1500 --n-examples 3
"""))

C.append(("markdown", "## 7. Train (joint action + value, L2~ unfreeze, multinomial-alloc contract)"))

C.append(("code", """\
import time
TS = time.strftime('%Y%m%d-%H%M%S')
RUN_TAG = f'joint_mt_alloc_d256_T10_head3_{TS}'
OUT_DIR = f'data/runs/joint/{RUN_TAG}'
# hyperparams
D_MODEL, N_STEPS     = 256, 10
HEAD_N_LAYERS        = 3     # match agent A (deep heads)
CONDITIONER_N_LAYERS = 3     # match agent A
BATCH_SIZE, EPOCHS   = 16, 35  # fresh value heads need the epochs (v2 recipe)
LR, WEIGHT_DECAY     = 5e-5, 1e-4  # gentle: backbone + action head are warm
PAIR_POS_WEIGHT      = 600.0 # whole-grid select BCE positive weight (stage 1).
                             # Unchanged from every multi-target pretrain; the
                             # single-target LAUNCH_WEIGHT knob does not apply.
ALLOC_WEIGHT         = 1.0   # stage-2 allocation CE weight. Raise toward 2-4
                             # only if act/hold_mae plateaus high while
                             # act/pair_logits keeps falling (select drowning alloc).
VALUE_COEF           = 1.0
VALUE_DROPOUT        = 0.1   # value trunk/heads only — anti win-memorization
NUM_WORKERS          = 2
WARM_START_PATH      = '/content/orbit-wars/warm_start.pt'  # entity_head3_value_merged
# Watch in the per-head logs:
#   act/pair_frac        = alloc CE (NEW) — should fall fast then flatten
#   act/hold_share_pred  -> act/hold_share_label (~0.07 on this cache);
#   act/hold_mae         -> < ~0.05 = the diagonal is calibrated
#   act/pair_logits      = select BCE — should stay near its warm-start level
#                          (a big jump means the alloc CE is fighting select)
#   val/* + HOLDOUT      = same win/signal heads + memorization-gap log as v2
print('RUN_TAG =', RUN_TAG)
"""))

C.append(("code", """\
# Run via the ! shell magic with `-u` (unbuffered) so the per-head logs
# STREAM LIVE into the cell (subprocess.run block-buffers in Colab).
# train_joint will print the T=10 restack + context-coverage probe lines
# right after "[joint] action cache:" — verify them before the first epoch.
!python -u -m agents.transformer_v2.pretrain.joint_pretrain \\
  --out-dir $OUT_DIR \\
  --fleet-run-dir $FLEET_RUN_DIR \\
  --planet-run-dir $PLANET_RUN_DIR \\
  --comet-run-dir $COMET_RUN_DIR \\
  --pair-cache-path $PAIR_CACHE_PATH \\
  --cross-cache-path $CROSS_CACHE_PATH \\
  --d-model $D_MODEL \\
  --n-steps $N_STEPS \\
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
## 8. Push the trained run back to GCS — then the two follow-ups

`joint_best.pt` is a drop-in PPO base (same `model`+`config` layout as
`entity_head3_value_merged.pt`; `config.action_contract =
bernoulli_select_multinomial_alloc_v2`). Before judging it:

1. **Runner sizing mode** — threshold deploy still sizes
   `sigmoid(frac)·ships`; this contract's `frac_loc` are softmax-share
   logits. Add/flip the alloc-softmax sizing in `runner.py` (size fired
   cells by `softmax([frac_loc[fired], frac_loc[s, s]])·N` — v2: the HOLD
   share comes from the frac head's diagonal), THEN run the deploy eval
   panel. Sampled-contract PPO (`sample_multi_target`) needs no change.
2. **Deploy eval gate** — seat-balanced seed panel vs physical_v4 + vs A
   (`utils/eval_seeds.py`), not training metrics.
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
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": [cell(k, s) for k, s in C],
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "pretrain_multitarget_alloc_T10_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out} ({len(nb['cells'])} cells)")
