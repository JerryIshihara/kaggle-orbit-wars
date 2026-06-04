# PPO RunPod Infserver Protocol

Last updated: 2026-06-04.

This document records the current single-RunPod PPO trial design, the launch
protocol, the logging contract, and the issues found while bringing up the
2-iteration trial.

## Current Status

The current intended run is not the old CPU pool rollout. It is:

```text
rollout = infserver
device = cuda
model forward = one parent-owned GPU model instance
env stepping = many CPU worker processes
T = 10
max_planets = 64
max_fleets = 512
post_pack_workers = 0 for the current trial
temporary episode spool = /dev/shm/ow_infserver_spool
durable output = /workspace/ow/ppo_<run_id>
```

The pod checked for this run:

```text
pod id: 2fufyrlvbytqfb
ssh: root@213.192.2.72 -p 40060
key: /Users/agent/.runpod_key.txt
GPU: RTX 3090, 24 GB
CPU: 256 vCPU class
```

Stopped 128-fleet run superseded by the 512-fleet requirement:

```text
run_id: runpod_2fufyrlvbytqfb_infserver_gpu_p200_f128_w20_post0_cpupack_20260603-184440
pid: 13191
out_dir: /workspace/ow/ppo_runpod_2fufyrlvbytqfb_infserver_gpu_p200_f128_w20_post0_cpupack_20260603-184440
stream: ssh tail of out_dir/nohup.out
rollout: infserver, device=cuda, PROCS=200, POST_PROCS=0
shape: T=10, max_planets=64, max_fleets=128
update: rollout pack_device=cpu, PPO minibatch size=256
status: stopped before completion because the target cap was changed back to max_fleets=512
```

Failed 128-fleet run before the CPU-pack fix:

```text
run_id: runpod_2fufyrlvbytqfb_infserver_gpu_p200_f128_w20_post0_20260603-183322
out_dir: /workspace/ow/ppo_runpod_2fufyrlvbytqfb_infserver_gpu_p200_f128_w20_post0_20260603-183322
result: rollout 0 completed, then PPO packing failed before update
rollout 0: 64/64 games, forwards=2348, batch_mean=41.7, batch_max=196
failure: torch.OutOfMemoryError while concatenating full PPO rollout tensors on CUDA
```

Failed 512-fleet run after the CPU-pack fix:

```text
run_id: runpod_2fufyrlvbytqfb_infserver_gpu_p200_f512_w20_mb64_cpupack_20260603-185401
out_dir: /workspace/ow/ppo_runpod_2fufyrlvbytqfb_infserver_gpu_p200_f512_w20_mb64_cpupack_20260603-185401
rollout: infserver, device=cuda, PROCS=200, POST_PROCS=0
shape: T=10, max_planets=64, max_fleets=512
update: PPO minibatch size=64, rollout pack_device=cpu
last observed progress: rollout 0 at 30/64 games, gpu_forwards=1344, batch_mean=50.0, batch_max=133
failure: trainer disappeared mid-rollout without Python traceback, CUDA OOM, dmesg OOM, or RUN DONE
```

Wrong-cap run stopped after that failure:

```text
run_id: runpod_2fufyrlvbytqfb_infserver_gpu_p200_f128_w20_post0_compactclean_20260603-190151
pid: 15382
shape: T=10, max_planets=64, max_fleets=128
status: killed because it was still running the wrong 128-fleet cap
```

Next intended run:

```text
shape: T=10, max_planets=64, max_fleets=512
games_per_iter: 64
PROCS: 48
actual_env_workers: 48
PPO minibatch size: 64
infserver_max_batch: 128
reason: keep the full 512 fleet cap while reducing rollout and update VRAM pressure
```

Both local and remote code currently contain the infserver safety patches:

```text
--infserver-spool-dir
--delete-infserver-spool-after-post
--packed-rollout-artifacts-only
NumPy worker->parent IPC for step requests
infserver stall guard
pool CPU-forward guard
rollout-to-PPO pack_device=cpu
```

## Model And PPO Objective

The PPO trial uses the pretrained joint actor/value checkpoint staged as:

```text
/root/ow/joint_best.pt
```

Before rollout, `train_local_trial` verifies that the supervised checkpoint was
actually loaded into the entity model. The expected successful line is:

```text
pretrained load check: ckpt_tensors=172 exact_match=172 value_heads=56 OK
```

The current PPO critic is Design A reward decomposition:

```text
V(s) = win_weight * sigmoid(win_logit_self) - Phi(s) + residual(glob)
```

Current code uses `sigmoid(win_logit_self)`, not `2 * sigmoid(win_logit_self)`.
Any older note saying `2sigma(win)` is not the current implementation.

`Phi(s)` is computed analytically from five P1 signals:

```text
ship_adv
production_adv
planet_adv
safety
fleet_speed_adv
```

Each signal is an advantage share in `[0, 1]`, generally:

```text
own / (own + strongest_or_relevant_enemy_reference)
```

PBRS shaping rewrites rollout rewards as:

```text
r_t = gamma * Phi(s_{t+1}) - Phi(s_t)
terminal_reward += win_weight * z
```

where `z = 1` for a learner-seat win and `0` otherwise. The stored step value is
also shifted by `-Phi(s)` so GAE uses the same shaped-return convention as the
critic.

Current default reward weights in the trainer:

```text
win_weight = 0.5
signal_weight = 0.1 for each of the 5 signals
value_gamma = 0.997
```

The pretrained win/value heads are loaded from `joint_best.pt`. The residual
`value_head` is zero-initialized but trainable during PPO. For `--unfreeze head`,
the intended trainable set is action heads plus the reward-decomp residual; lower
encoder layers stay frozen.

## Infserver Rollout Design

The purpose of `--rollout infserver --device cuda` is to keep model inference on
the GPU while using CPU cores for env stepping.

Process layout:

```text
parent trainer process
  owns PPOActorCritic on CUDA
  owns planet/fleet/comet encoders on CUDA
  batches ready one-step seat observations
  runs _batched_forward under torch.inference_mode()

CPU worker processes
  own orbit_wars env instances
  own FleetTracker state
  build T=10 features on CPU
  send current-seat feature stores to parent
  receive logits/value/sigma
  decode actions and step envs
```

Workers do not load the model and must not run model forward. The old `pool`
rollout is CPU-forward only and is rejected unless
`--allow-cpu-forward-rollout` is explicitly passed.

Hot-path worker requests are converted from torch tensors to NumPy arrays before
queueing. This avoids torch multiprocessing shared-storage reducers filling
`/dev/shm`.

Finished episodes are written by workers as files and passed by path:

```text
/dev/shm/ow_infserver_spool/<run_id>/<vtag>/episode_000000.pt
```

Post-pack workers consume completed episode files while rollout continues. The
spool must be local/tmpfs (`/dev/shm`), not the network filesystem under
`/workspace`.

## Current Run Configuration

Current bootstrap defaults in `scripts/ppo_runpod_bootstrap.sh`:

```text
ROLLOUT=infserver
DEVICE=cuda
POST_PROCS=4
MAX_PLANETS=64
MAX_FLEETS=512
MINIBATCH_SIZE=64 when MAX_FLEETS>=512
INFSERVER_STALL_TIMEOUT_S=180
INFSERVER_SPOOL_DIR=/dev/shm/ow_infserver_spool
PACKED_ROLLOUT_ARTIFACTS_ONLY=1
DELETE_INFSERVER_SPOOL_AFTER_POST=1
DELETE_LOCAL_ROLLOUTS_AFTER_UPLOAD=1
```

Worker-count rule:

```text
if PROCS is set:
  use PROCS
elif MAX_FLEETS >= 512 and nproc >= 128:
  use 48
elif nproc >= 128:
  use 200
else:
  use nproc - POST_PROCS
```

Important limitation in the current code:

```text
actual_env_workers = min(PROCS, games)
```

So with `--games 64`, even `PROCS=200` creates only 64 active env workers. To
use 150-200 env workers without changing the scheduler, `--games` must be at
least the desired worker count. Otherwise the GPU batch size is limited by the
64 concurrent games, although each game can contribute multiple active seats.

Recommended 512-fleet 2-iteration trial command shape:

```bash
cd /root/ow

RUN_ID="runpod_2fufyrlvbytqfb_infserver_gpu_p48_f512_w20_mb64_cpupack_$(date -u +%Y%m%d-%H%M%S)"
OUT="/workspace/ow/ppo_${RUN_ID}"
mkdir -p "$OUT"
touch "$OUT/nohup.out"

nohup bash -lc "
  cd /root/ow
  env \
    OW=/root/ow \
    OUT_DIR=\"$OUT\" \
    RUN_ID=\"$RUN_ID\" \
    STREAM_URL=\"$OUT/STATE.json\" \
    ROLLOUT=infserver \
    DEVICE=cuda \
    PROCS=48 \
    POST_PROCS=0 \
    MAX_FLEETS=512 \
    MAX_PLANETS=64 \
    MINIBATCH_SIZE=64 \
    SKIP_DEPS=1 \
    INFSERVER_SPOOL_DIR=\"/dev/shm/ow_infserver_spool_${RUN_ID}\" \
    DELETE_INFSERVER_SPOOL_AFTER_POST=1 \
    PACKED_ROLLOUT_ARTIFACTS_ONLY=1 \
    DELETE_LOCAL_ROLLOUTS_AFTER_UPLOAD=0 \
    INFSERVER_STALL_TIMEOUT_S=300 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/ppo_runpod_bootstrap.sh \
    --iters 2 \
    --games 64 \
    --unfreeze head \
    --progress-step-every 25 \
    --infserver-max-batch 128 \
    --infserver-batch-window-ms 20.0
  rc=\$?
  echo \"[launcher] EXIT rc=\$rc at \$(date -u)\"
  exit \$rc
" > "$OUT/nohup.out" 2>&1 &

echo "PID=$! RUN_ID=$RUN_ID OUT=$OUT"
```

Why `PROCS=48`: with `--games 64`, the trainer creates
`min(PROCS, games)` workers, so this launches 48 actual env workers. The
previous 512-fleet attempt with effective 64 workers disappeared mid-rollout
without a traceback. Reducing workers and capping inference batches at 128 is a
conservative 3090-memory setting while keeping the requested 512 fleet slots.

Why `MINIBATCH_SIZE=64`: 512 fleet slots make each PPO minibatch about 4x
larger than the stopped 128-fleet debug run. `64 * 512` has roughly the same
fleet-token footprint as `256 * 128`, while the CPU-pack fix prevents
full-rollout tensors from being materialized on CUDA.

If the goal is to actually use around 200 env cores, increase games:

```text
--games 200
```

or change the rollout scheduler so many workers pull game tasks dynamically from
a queue. The current implementation partitions a fixed game list and clamps
worker count to the number of games.

## Log Streaming Protocol

The complete stream must start before rollout, dependency install, artifact
extraction, or model load. The authoritative stream is `nohup.out`.

The safe sequence is:

1. Create the output directory.
2. Create `nohup.out`.
3. Start a local SSH tail from line 1.
4. Launch bootstrap with stdout/stderr redirected to the same file.

Stream command:

```bash
ssh root@213.192.2.72 -p 40060 -i /Users/agent/.runpod_key.txt \
  -o IdentitiesOnly=yes \
  'tail -n +1 -F /workspace/ow/ppo_<run_id>/nohup.out'
```

`tail -n +1 -F` prints the file from the first line and then follows future
writes. This captures:

```text
bootstrap start line
python/torch/CUDA check
dependency install start/end
code.tgz and weights.tgz extraction
reward-decomp preflight
train_local_trial args
checkpoint load verification
rollout progress
Phi shaping stats
RL/GAE/PPO metrics
checkpoint save
final RUN DONE or traceback
```

`STREAM_URL="$OUT/STATE.json"` is useful for structured progress, but it starts
only after `train_local_trial` initializes. It does not replace `nohup.out` for
beginning-of-run logging.

Current structured progress contains:

```text
stage
workers
workers_alive
games_done / games_total
gpu_forwards
batch_mean / batch_max
fwd_ms_mean
worker_progress rows
worker_exits
wall_s
```

Current worker progress rows contain:

```text
wid
game_idx
seed
num_players
learner_seat
step
status
```

Current gap: per-core timing fields (`feat_s`, `gpu_wait_s`, `decode_s`,
`env_s`, active seats, and env step) are recorded in final game stats but are
not yet streamed in each worker progress row. The interrupted patch was intended
to add those fields to live row logging.

PPO training logs include:

```text
RL signals: reward[...] ep_return[...]
GAE/value: value[...] adv[...] return[...]
PPO batches: n_mb=... size[min/mean/max]=...
PPO epoch K: mb=... kl=... ratio=... policy=... value=... entropy=... clip=...
update: ... kl=... policy_loss=... value_loss=... entropy=... clip_frac=...
```

## Artifact And Versioning Protocol

Run outputs should stay under:

```text
/workspace/ow/ppo_<run_id>
```

Expected versioned artifacts for a complete 2-iteration trial:

```text
checkpoints/policy_v0000.pt
checkpoints/policy_v0001.pt
checkpoints/policy_v0002.pt
checkpoints/policy_latest.pt
rollouts/v0000/...
rollouts/v0001/...
train.log
train_log.jsonl
STATE.json
nohup.out
```

`policy_v0000.pt` is the loaded initial policy snapshot. After two PPO updates,
`policy_v0002.pt` must exist.

With `PACKED_ROLLOUT_ARTIFACTS_ONLY=1`, raw tensor-heavy episode shards are not
duplicated in durable rollout artifacts. PPO-packed rollout inputs and manifests
are kept instead.

No local Google ADC or reusable Google credential should be copied to the pod.
If GCS output is needed, use a pod-native service account/auth path. Without
pod-side auth, keep `/workspace` as the durable RunPod artifact location and
sync it manually after the run.

## Current Issues And Fixes

### 1. CPU Forward Was Accidentally Used

Problem:

```text
--rollout pool --device cpu
```

This loaded the checkpoint but ran model forward on CPU, wasting GPU pod time
and violating the intended design.

Current fix:

```text
--rollout infserver --device cuda
```

and `--rollout pool` is blocked unless `--allow-cpu-forward-rollout` is passed.

### 2. Stale code.tgz Can Overwrite Patched Code

Bootstrap extracts `/root/ow/code.tgz` at startup. Uploading a new archive under
another name is not enough. The intended code bundle must replace:

```text
/root/ow/code.tgz
```

Preflight checks catch some stale-code cases:

```text
PPOActorCritic has reward_decomp
train_local_trial has apply_phi_shaping
joint_best.pt has value_heads.*
```

They do not prove every infserver patch is present, so grep for the expected
flags before launch when in doubt.

### 3. Torch Tensor IPC Filled /dev/shm

Problem:

Workers originally sent tensor-heavy request/episode objects through
multiprocessing queues. Torch used shared-memory storage reducers and `/dev/shm`
filled up.

Current fix:

```text
step requests: torch tensors -> NumPy arrays before queueing
finished episodes: torch.save() to a local spool file, queue only the path
```

### 4. /workspace Spool Caused End-Of-Rollout Stalls

Observed failed run:

```text
runpod_2fufyrlvbytqfb_infserver_gpu_p200_f256_w20_20260603-181639
games_done = 60/64
final workers stuck in D-state request_wait_answer
rollout stalled after about 200s of no progress
```

Likely cause:

```text
episode spool files were written under /workspace, a network filesystem
```

Current fix:

```text
INFSERVER_SPOOL_DIR=/dev/shm/ow_infserver_spool
--delete-infserver-spool-after-post
```

Durable artifacts still go to `/workspace`; only transient episode spools go to
`/dev/shm`.

### 5. PROCS=200 Does Not Mean 200 Active Workers With 64 Games

The current rollout code clamps worker count:

```text
n_workers = min(PROCS, len(specs))
```

For `--games 64`, at most 64 workers are active. This is why increasing
`PROCS` alone may not fill the GPU to batch 256. Use more games or implement a
dynamic game-task scheduler.

### 6. GCS/Firestore Streaming Is Optional And Not The Source Of Truth

`STATE.json` works as a structured local path. GCS/Firestore requires pod-side
auth and installed clients. Do not copy local ADC credentials to the pod.

For this pod, the reliable live stream is SSH tailing `nohup.out` from
`/workspace`.

### 7. Live Step Logging Is Present But Not Yet Detailed Enough

Current logs show rollout-level progress and worker rows at
`--progress-step-every`, plus final per-game timing. They do not yet stream all
per-core timing buckets during the game. Add the following fields to the worker
progress rows if deeper live diagnosis is needed:

```text
env_step
active_seats
wall_s
feat_s
gpu_wait_s
decode_s
env_s
```

### 8. Incremental T=10 Encoding Is Still Not Implemented

Every step currently re-encodes the full T=10 window. Since 9 frames overlap
with the previous step, caching per-frame L0/L1 encodings and only encoding the
new frame should cut a major part of rollout forward cost. L2-L4 should still
attend over the full T-window.

### 9. Full-Rollout CUDA Packing OOM

Observed failed run:

```text
runpod_2fufyrlvbytqfb_infserver_gpu_p200_f128_w20_post0_20260603-183322
rollout 0 completed: 64 ok, 0 failed, total_steps=31936
failure point: episodes_to_ppo -> _packed_episodes_to_ppo
error: torch.OutOfMemoryError while concatenating pair_type_ids on CUDA
attempted allocation: 9.75 GiB
GPU already held: about 14.55 GiB
```

Cause:

```text
train_local_trial passed device=cuda into rollout-to-PPO packing, so the full
rollout tensor set was materialized on GPU before PPO update. On a 24 GB 3090,
the CUDA model/encoder memory plus the full rollout tensor concat does not fit.
```

Current local fix:

```text
rollout-to-PPO packing always uses device=cpu
ppo_update_local moves only the current PPOMinibatch to the policy device
PPO forward/backward still runs on CUDA because the policy lives on CUDA
process log now records pack_device=cpu
```

Expected successful process log after the fix:

```text
process[serial pack_device=cpu]: ...
```

or, with post-pack workers:

```text
process[live post_pack_workers=4 pack_device=cpu]: ...
```

## Completion Checks For The 2-Iteration Trial

Do not consider the goal complete until all checks pass:

```bash
ssh root@213.192.2.72 -p 40060 -i /Users/agent/.runpod_key.txt \
  -o IdentitiesOnly=yes \
  'RUN=/workspace/ow/ppo_<run_id>;
   grep -E "RUN DONE|Traceback|RuntimeError|stalled" "$RUN/nohup.out" | tail -n 50;
   find "$RUN/checkpoints" -maxdepth 1 -type f -printf "%f %s\n" | sort;
   tail -n 5 "$RUN/train_log.jsonl";
   find "$RUN/rollouts" -maxdepth 3 -type f | head -n 50;
   ps -eo pid,ppid,etime,pcpu,pmem,state,wchan:20,args |
     grep -E "train_local_trial|ppo_runpod_bootstrap" | grep -v grep || true;
   nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
     --format=csv,noheader;
   df -h /dev/shm /workspace'
```

Required evidence:

```text
nohup.out says RUN DONE
train_log.jsonl has rows for iter 0 and iter 1
policy_v0000.pt exists
policy_v0001.pt exists
policy_v0002.pt exists
policy_latest.pt exists
rollouts/v0000 exists with packed manifests/shards
rollouts/v0001 exists with packed manifests/shards
no active failed/stuck trainer process remains
```
