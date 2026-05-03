# Unfreeze OOM fixes + UI-runnable agent

Companion to [`GRADUAL_UNFREEZE.md`](./GRADUAL_UNFREEZE.md) and
[`GPU_TRAINING.md`](./GPU_TRAINING.md). Records the fixes for the unfreeze-train
OOM hit on Colab T4 and the recipe for plugging the freeze-train action `.pt`
into a UI-runnable agent.

---

## TL;DR

| Constraint | First action |
|---|---|
| OOM, can't change code | `PYTORCH_ALLOC_CONF=expandable_segments:True` + halve batch + bf16 autocast |
| OOM, can edit model | Replace additive attention in `entity_encoder.py:83` with `F.scaled_dot_product_attention` |
| Don't want to debug | Switch Colab runtime → **L4 GPU** (24 GB VRAM, $10/mo Colab Pro) |
| Training slow, not OOM | In-memory dataset + `num_workers=4`, `pin_memory=True`, `persistent_workers=True` |

---

## 1 · OOM during unfreeze (`entity_encoder.py:83`)

The error fires on:

```python
scores = self.v(torch.tanh(q + k)).squeeze(-1)
```

This is **Bahdanau-style additive attention**. `q + k` broadcasts to
`(B, N, M, H, D)` — quadratic in sequence length × heads, easy to blow past
14 GB on T4 once gradients are unfrozen.

### Fix priority

#### a. Cheapest, no code change (try first)

```python
# at the very top of the training notebook, BEFORE `import torch`
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

The error message itself recommends this — it can save 1–2 GB on T4 by killing
fragmentation.

#### b. Halve batch + grad-accum + mixed precision (combine)

Most of the saving comes from this combination:

```python
batch_size  = 16          # was 32
accum_steps = 2           # effective batch still 32
scaler      = None        # not needed for bf16

for step, batch in enumerate(loader):
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = model(**batch) / accum_steps
    loss.backward()
    if (step + 1) % accum_steps == 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
```

bf16 is preferred over fp16 on L4/A100 (no overflow risk, no GradScaler needed).
On T4 use fp16 + GradScaler instead.

#### c. Activation checkpointing on the encoder

```python
from torch.utils.checkpoint import checkpoint

# in EntityEncoder.forward():
out = checkpoint(self._inner_forward, x, use_reentrant=False)
```

Trades ~25% step time for ~40% activation memory. Worth it if (a)+(b) still OOM.

#### d. Real fix — swap additive → scaled-dot-product attention

Bahdanau attention is OOM-prone because it materializes the full `(B, N, M, H, D)`
tensor. Modern transformers use scaled-dot-product attention which never does:

```python
# agents/transformer_v1/encoder/entity_encoder.py
# REPLACE the additive block:
#   scores = self.v(torch.tanh(q + k)).squeeze(-1)
#   attn = F.softmax(scores, dim=-1)
#   out = attn @ v
# WITH:
import torch.nn.functional as F
out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

This automatically uses Flash Attention on L4/A100 — 5–10× memory cut for
typical sequences and faster than the additive variant on every backend.

#### e. GPU upgrade table (if you'd rather pay)

| GPU | VRAM | Where | Cost | Verdict |
|---|---|---|---|---|
| **L4** | 24 GB | Colab Pro | $10/mo | ✅ sweet spot — clears OOM at current batch size |
| A100 40 GB | 40 GB | Colab Pro+ / GCP | $1–3/hr | overkill for this model |
| A100 80 GB | 80 GB | RunPod / Lambda | $2/hr | only if scaling up sequences ≫ current |

**Pick L4** unless you specifically need >24 GB.

---

## 2 · L4 + in-memory loader (the "training is slow" track)

Once you're on L4 with 51 GB RAM, the data loader almost always becomes the
bottleneck before the GPU does. Standard fix:

```python
# agents/transformer_v1/pretrain/loader.py  (or wherever your DataLoader lives)
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class InMemoryDataset(Dataset):
    """Eager-load + tensorize once. Caches the parsed list to disk so re-runs skip parsing."""
    def __init__(self, sources, parse_fn, cache_path="/content/_inmem_cache.pt"):
        try:
            self.samples = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"  loaded {len(self.samples):,} samples from {cache_path}")
        except FileNotFoundError:
            print(f"  parsing {len(sources):,} sources into RAM…")
            self.samples = [parse_fn(s) for s in tqdm(sources)]
            torch.save(self.samples, cache_path)

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i):  return self.samples[i]


def make_loader(ds, batch_size, shuffle=True):
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,            # L4 + 2-4 vCPUs sweet spot
        pin_memory=True,          # async CPU→GPU transfer
        persistent_workers=True,  # don't respawn between epochs
        prefetch_factor=4,
        shuffle=shuffle,
    )

# free 20-30% on L4 matmul
torch.set_float32_matmul_precision("high")
```

Expected wall-time effect: `data_load_time / step` drops from 15–30% → <1%.
If it doesn't, your `parse_fn` is the bottleneck — move tensor conversion into
`__init__`, not `__getitem__`.

---

## 3 · UI-runnable agent from action `.pt`

After freeze-train (encoders frozen, action decoder fitted) you have a
checkpoint at `data/runs/action/<run>/best.pt`. Wire it through the physical
helper into a registered agent so the kaggle-environments UI / runner can use
it the same way it uses `physical_v4`.

The helper conversion is the same one `physical_v4` uses — given a target
planet pick, compute `(source_planet, angle, ships)` via lead-aim + surplus
check. We import those primitives from `agents.physical_v4.agent` so we don't
re-implement physics.

### File: `agents/transformer_v1/runner.py`

A drop-in skeleton is included at `agents/transformer_v1/runner.py`. Key
ideas:

1. **`Agent.load(ckpt_path)`** — loads `best.pt`, restores model + cfg, moves
   to device, calls `.eval()` and warms up cuDNN with one fake forward.
2. **`Agent.act(obs)`** — converts obs → tensors, runs the policy, picks the
   target planet (argmax or sampled), then hands off to the physical helper to
   pick the launcher and compute angle + ships.
3. **`@register("transformer_v1")`** — exposes the agent through the existing
   registry so `kaggle_environments.make("orbit_wars").run(["transformer_v1", "physical_v4"])`
   just works.

### Run it

```bash
# Smoke test (100 fake ticks, prints Hz)
cd /content/orbit-wars
python -m agents.transformer_v1.runner --ckpt data/runs/action/<run>/best.pt --smoke-test

# UI / replay mode
python run.py --p1 transformer_v1 --p2 physical_v4 --episodes 5 --render
```

Expected throughput on L4 at batch=1: **120–250 Hz**, plenty for the 60 Hz UI.

### Action masking (do this once basic agent is running, before unfreeze)

Transformers love picking "clever" but illegal moves. Mask before argmax:

```python
mask = torch.tensor(
    [is_legal_target(i, obs) for i in range(target_logits.size(-1))],
    device=self.device, dtype=torch.bool,
)
target_logits = target_logits.masked_fill(~mask, float("-inf"))
target_idx = target_logits.argmax(dim=-1).item()
```

Doing this *before* unfreeze-train improves data-collection signal-to-noise
without retraining anything.

---

## 4 · Push results to GCS

```bash
# parallel checkpoint dir upload
gsutil -m cp -r data/runs/action/<run> gs://<bucket>/orbit-wars/runs/action-<run>/

# single big checkpoint
gsutil cp data/runs/action/<run>/best.pt gs://<bucket>/orbit-wars/best.pt

# resume-friendly (only copies changed files)
gsutil -m rsync -r data/runs gs://<bucket>/orbit-wars/runs

# parallel composite for >5 GB single files
gsutil -o "GSUtil:parallel_composite_upload_threshold=150M" cp big.pt gs://<bucket>/big.pt
```

Colab auth: `from google.colab import auth; auth.authenticate_user()` then
`!gcloud config set project <PROJECT>`.

---

## Status checklist

- [x] freeze-train (encoders frozen, action decoder fit) — `best.pt` produced
- [ ] runner skeleton — `agents/transformer_v1/runner.py`
- [ ] action masking in `Agent.act()`
- [ ] unfreeze-train without OOM — pick fix track from §1
- [ ] push final ckpts to GCS — §4
