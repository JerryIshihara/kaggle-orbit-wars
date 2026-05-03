# GPU training on GCP

End-to-end recipe for moving a pretrain run to a Compute Engine GPU
VM: package code + data + existing checkpoints, train, monitor, pull
the new checkpoint back. The `cross_entity` pretrain (transformer +
multi-task heads) is the first target, but the same workflow applies
to any future run.

## Sizing

For the current dataset (~600 MB CSVs) and model size (~100 k–1 M
params), pick an inexpensive Compute Engine GPU VM:

| Spec | Choice | Why |
|---|---|---|
| Machine type | `n1-standard-8` (8 vCPU, 30 GB) | CPU is mostly idle — bandwidth + RAM matter more than cores |
| GPU | 1 × **T4** | $0.35/hr on-demand; ~10× faster than CPU on this size; bigger card unnecessary |
| Disk | 100 GB SSD | dataset + ckpts + scratch comfortably |
| Image | `pytorch-latest-gpu` (Deep Learning VM) | PyTorch + CUDA preinstalled |
| Region | nearest with T4 stock | latency on `gsutil cp` matters when iterating |

A T4-equipped VM costs roughly **$0.5/hr** all-in (machine + GPU +
disk). Keep it stopped between runs; ~$10 for a long pretrain
campaign of multiple iterations.

## One-time setup

### 1. Create the VM

```bash
# Adjust ZONE / PROJECT to your account.
ZONE="us-central1-b"
PROJECT="your-project-id"
VM_NAME="orbit-wars-gpu"

gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

The Deep Learning image installs the NVIDIA driver on first boot
(`install-nvidia-driver=True`); SSH in once and let it complete:

```bash
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="nvidia-smi"
```

### 2. Create a GCS bucket for transfer

```bash
gsutil mb -l us-central1 gs://orbit-wars-shipping
```

We use the bucket as a transfer staging area for code, data, and
checkpoints — `gsutil cp` is faster and more robust than `scp` for
multi-GB tars.

## Per-training-run workflow

### 3. Pack on local

```bash
bash scripts/pack_for_gpu.sh
```

Produces three tarballs in `/tmp/orbit-pack/`:

* `code.tgz` — agents/, scripts/, utils/, requirements.txt
  (~few MB; everything you'd need to re-run training)
* `data.tgz` — only the dataset dirs the cross-entity training
  reads: `data/datasets/{fleet,planet,entity,cross_entity}/`
  (~600 MB)
* `weights.tgz` — frozen encoder checkpoints to load at training
  start (`data/runs/fleet/<best>/`, `data/runs/planet/<best>/`,
  `data/runs/entity/<best>/`)

The script prints `gsutil cp` commands to copy each tarball to the
bucket — paste them or pipe through.

### 4. Push and unpack on the VM

```bash
gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- \
  bash -lc 'gsutil cp gs://orbit-wars-shipping/{code,data,weights}.tgz . \
            && tar xzf code.tgz \
            && tar xzf data.tgz \
            && tar xzf weights.tgz'
```

### 5. Launch training in tmux

```bash
gcloud compute ssh "$VM_NAME" --zone="$ZONE"
# inside the VM:
tmux new -s train
cd ~/orbit-wars
pip install -r requirements.txt   # first run only
python -m agents.transformer_v1.pretrain.cross_entity \
    --epochs 30 --batch-size 64 --num-workers 4 \
    --device cuda
# Detach: Ctrl-b d
```

Tmux means SSH disconnects don't kill the run; reattach later via
`tmux attach -t train`.

### 6. Monitor progress

While training, the run writes `data/runs/cross_entity/<ts>/log.json`
on each epoch. From local, poll the latest log without holding an
open SSH session:

```bash
bash scripts/sync_from_gpu.sh logs        # log.json + test_summary if present
```

The script `scp`'s the latest run dir's *small* artifacts (json
logs, ~kB) so a one-line cron can keep your local copy fresh.

For per-step monitoring (during a long epoch) ssh in and `tail -f
data/runs/cross_entity/<ts>/log.json` — JSON Lines or pretty-print
either way.

### 7. Pull the checkpoint back

When the run finishes, sync everything in the run dir, including the
heavy `.pt` files:

```bash
bash scripts/sync_from_gpu.sh full
```

Then optionally delete the bucket copies and stop the VM:

```bash
gsutil rm -r gs://orbit-wars-shipping/{code,data,weights}.tgz
gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
```

The instance keeps its disk while stopped — restart with
`instances start` to reuse the unpacked tree without re-uploading
code or data.

## What gets uploaded vs not

| Path | Uploaded | Why |
|---|---|---|
| `agents/transformer_v1/` | ✓ | training code |
| `scripts/`, `utils/` | ✓ | data loaders, runners |
| `requirements.txt` | ✓ | reproducible env |
| `data/datasets/{fleet,planet,entity,cross_entity}/*.csv` | ✓ | training data |
| `data/datasets/*/manifest.json` | ✓ | split definitions |
| `data/runs/{fleet,planet,entity}/<best>/*.pt` | ✓ | frozen encoder weights |
| `data/replays/` | ✗ | not needed once datasets exist |
| `notebooks/`, `data/runs/.../log.json` from prior local runs | ✗ | not used by the GPU run |
| `.venv/`, `__pycache__/` | ✗ | rebuilt on the VM |

## Why this layout

* **Bucket as relay** instead of direct scp — robust to connection
  drops on multi-GB transfers and lets you re-pull on a fresh VM
  without re-tarring locally.
* **Three tarballs** (code/data/weights) instead of one — code is
  tiny and changes often, data is big and changes rarely, weights
  are medium and change per pretrain stage. Independent uploads
  minimize re-transfer when iterating.
* **tmux + `gsutil` JSON pulls** beats W&B / TensorBoard for the
  size of run we have here. If the project grows to needing real-
  time dashboards, switching to W&B is a one-line addition to the
  pretrain scripts (`wandb.init` + `wandb.log` per epoch).
* **Stop, don't delete the VM** between runs — the disk persists,
  so re-runs only re-tar+upload code, not data.

## Future tweaks

* For multi-GPU runs, swap `--device cuda` for `torch.distributed`
  launching (`torchrun`) — the dataset/model are already
  device-agnostic, so the change is one wrapper around `train()`.
* For mixed-precision, add `--amp` flag wrapping the forward in
  `torch.amp.autocast('cuda', dtype=torch.bfloat16)` — T4 doesn't
  support bf16 but A10/A100 do, and FP16 is fine on T4.
* For TPU runs, Deep Learning VM has a TPU image; the encoder
  modules are vanilla PyTorch so torch_xla works without code changes.
