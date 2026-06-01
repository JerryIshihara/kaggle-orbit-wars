# PPO actor↔learner coordination — central STATE protocol

One or more **GCP CPU rollout VMs** (`c4-*` or `n2-*`) and **Colab** (learner, GPU)
run the PPO loop together with **no held SSH and no central scheduler**. They
coordinate through one authoritative STATE object, normally a Firestore doc
(`fs://ppo_runs/<run_id>`) with GCS used for big files. Whoever owns the current
phase acts, uploads its artifacts, then flips the phase to hand off.

```
   ┌──────────────── learner flips (Colab) ───────────────────────────────────────┐
   ▼                                                                               │
await_rollout ──actors roll + update actors[id].progress──▶ await_train ──Colab claims──▶ training ──┐
 model=vK                                              shards=vK/      +train progress          │
 iter=K                                                iter=K                                   │
   ▲                                                                          iter=K+1, model=v{K+1}
   └───────────────────────────────────────────────────────────────────────────────────────┘
```

This generalizes `docs/PPO_TWO_CPU_PROTOCOL.md` (implicit file-presence triggers) into an **explicit,
named state** both sides listen to — matching the user's mental model: *"rollout can start with model
pt, training can start with shards at …, rollout in progress with …"*.

---

## The state object — `fs://ppo_runs/<run_id>`

The control channel may also be `gs://.../STATE.json` for local/GCS polling, but
the current GCP-VM protocol uses Firestore for push-style updates. Big files are
always under `gcs_prefix`, for example `gs://orbit-wars-shipping/ppo/<run_id>`.

```json
{
  "run_id": "ppo_gcp_20260531-120000",
  "phase": "await_rollout",
  "iter": 0,
  "iters_total": 20,
  "model":  { "heads": "gs://.../heads/policy_v0.heads.pt",
              "full":  "gs://.../checkpoints/policy_v0.pt",
              "base_version": "<entity_sha12>", "policy_version": 0 },
  "shards": null,
  "config": { "history_window": 10, "episodes": 16, "num_players": "mix",
              "seed_base": 100000, "device": "cpu", "max_fleets": 512, "sigma": 0.35 },
  "actor":   { "id": "gcp-actor", "claimed_iter": -1, "heartbeat_ts": null, "progress": null },
  "expected_actors": ["c4-0"],
  "actors": {},
  "learner": { "id": "colab",    "claimed_iter": -1, "heartbeat_ts": null, "progress": null },
  "updated_by": "colab",
  "updated_ts": "2026-05-31T12:00:00Z",
  "message": ""
}
```

`learner.progress` carries live training telemetry. Rollout telemetry is either
`actor.progress` for the legacy single-actor path, or `actors[actor_id].progress`
for the current `expected_actors` path. The pool progress includes per-core rows
under `workers:[...]` with seed/player-count/step/score/thread count.

---

## Phases

| phase | meaning | who acts | sets these on entry | flips to (on finish) |
|---|---|---|---|---|
| `await_rollout` | model `vK` ready | **GCP actor VM(s)** | `model`, `iter=K` | `rolling_out` for legacy single actor, or `await_train` after the multi-actor barrier |
| `rolling_out`   | legacy single actor rolling, live | actor bumps `actor.progress` | — | `await_train`, `shards=rollouts/vK/<id>/` |
| `await_train`   | shards `vK` ready | **learner (Colab)** | `shards`, `iter=K` | `training` (claim) |
| `training`      | learner training, live | learner bumps `learner.progress` | — | `await_rollout`, `iter=K+1`, `model=v{K+1}` |
| `done`          | `iter >= iters_total` | both exit | — | — |
| `error`         | a side hit a fatal error | both halt | `message` | (manual) |

---

## Invariants (why it's safe without a lock)

1. **Phase-gated single writer.** Exactly one side owns each phase and is the *only* writer during it;
   the other side only **reads**. No concurrent writes → no lost updates, no CAS needed. (Hardening
   option: GCS `x-goog-if-generation-match` gives true compare-and-swap if a 2nd actor is ever added.)
2. **Artifacts before STATE (atomic handoff).** The owner uploads *all* its outputs (shards + `_DONE`,
   or `policy_v{K+1}.pt` + heads) **before** flipping the phase. GCS object replace is atomic, so when
   the peer sees the new phase, the referenced artifacts are guaranteed complete. (The `_DONE`-last rule,
   generalized to STATE-last.)
3. **Idempotent / resumable.** Before working, each side checks if its output already exists and
   short-circuits:
   - actor on `await_rollout K`: if `rollouts/vK/<id>/_DONE` exists → skip rollout, flip to `await_train`.
   - learner on `await_train K`: if `checkpoints/policy_v{K+1}.pt` exists → skip train, flip to `await_rollout K+1`.
   A crashed side re-reads STATE and resumes or skips. **Output files are ground truth; STATE is the hint.**
4. **Claim guard.** `await_* → -ing` writes `claimed_iter=K` + `heartbeat_ts` under the actor/learner id.
   With one actor VM + one Colab this is trivially safe; the id/lease matters only if a duplicate is launched.
5. **Liveness.** The active side bumps `heartbeat_ts` every `progress_every` s. No bump for
   `> stall_timeout` ⇒ peer surfaces `stalled` (no auto-reclaim with a single actor/learner).

---

## `ppo_state.py` API (the shared contract — both daemons import this)

```python
PHASE_AWAIT_ROLLOUT, PHASE_ROLLING_OUT, PHASE_AWAIT_TRAIN, PHASE_TRAINING, PHASE_DONE, PHASE_ERROR

state_url(prefix)                  -> "<prefix>/STATE.json"           # gs:// or local
read_state(url)                    -> dict | None                      # fs://, gs://, or local
write_state(url, state)            -> None                             # Firestore set or atomic full replace; stamps updated_ts
init_state(url, *, iters_total, model, config, run_id,
           gcs_prefix, expected_actors) -> dict                        # phase=await_rollout, iter=0
transition(url, *, expect_phase, new_phase, who, **fields) -> dict     # read→assert phase→merge→write
set_progress(url, side, progress)  -> None                             # bump actor|learner .progress + heartbeat
wait_for(url, phases, *, poll_s, on_tick) -> dict                      # block until phase in {phases}; on_tick(state) each poll
done_marker_exists(shards_prefix)  -> bool                             # rollouts/vK/<id>/_DONE
model_for_iter(state, K)           -> dict                             # the model block for iter K
```

`transition(expect_phase=…)` raises if the live phase ≠ `expect_phase` (someone else moved it / stale
read) so a confused side fails loud instead of double-acting. `set_progress` is the cheap, frequent
write the owner makes while working; `transition` is the rare handoff write.

---

## The two listener loops

**Actor — `ppo_actor_daemon.py` (runs on each GCP VM, `nohup`):**
```
while True:
    st = read_state(URL)
    if st.phase == DONE: break
    if st.phase == AWAIT_ROLLOUT:
        K = st.iter
        if done_marker_exists(rollouts/vK/<actor_id>/):           # idempotent
            skip my rollout slice
        else:
            download model.heads
            subprocess(rollout_worker --policy-heads … --out rollouts/vK/<actor_id>/)
            # while it runs: read worker progress.json -> update_actor(URL, actor_id, p)
        if every expected actor has rollouts/vK/<id>/_DONE:
            transition(expect=AWAIT_ROLLOUT, new=AWAIT_TRAIN,
                       who=actor_id, shards=rollouts/vK/, iter=K)
        on error : transition(new=ERROR, who=actor_id, message=…)
    else: sleep(poll_s)
```

**Learner — `ppo_learner_daemon.py` (runs on Colab, or driven by a notebook cell):**
```
while True:
    st = read_state(URL)
    if st.phase == DONE or st.iter >= st.iters_total: write_state(phase=DONE); break
    if st.phase == AWAIT_TRAIN:
        K = st.iter
        if exists(checkpoints/policy_v{K+1}.pt):                  # idempotent
            transition(expect=AWAIT_TRAIN, new=AWAIT_ROLLOUT, who=colab, iter=K+1, model=v{K+1}); continue
        transition(expect=AWAIT_TRAIN, new=TRAINING, who=colab, claimed_iter=K)
        subprocess(learner_step --shards st.shards --out-ckpt policy_v{K+1} …)
        # mirror learner_step progress -> set_progress(URL,'learner',p)
        next = DONE if K+1 >= iters_total else AWAIT_ROLLOUT
        transition(expect=TRAINING, new=next, who=colab, iter=K+1, model=v{K+1})
    else:
        on_tick: print actors[*].progress    # watch GCP rollout live
        sleep(poll_s)
```

When it is **not** a side's turn, it prints the *other* side's `progress`, so the user watching **either**
console sees the whole loop advance.

---

## Bootstrap
1. Colab init cell: push `policy_v0` (full + heads), then
   `init_state(URL, iters_total=N, model=v0, config={…})` → `phase=await_rollout, iter=0`.
2. Actor daemon(s) on the GCP VM(s) see `await_rollout 0` → roll → `await_train 0`.
3. Colab learner loop sees `await_train 0` → trains → `await_rollout 1`. … until `done`.

## GCP VM access

Each rollout VM needs permission to read Firestore state, pull head deltas, and
write rollout shards to GCS. On GCP this should be the instance service account
with Storage read/write plus Firestore access. The VM should also have:

- staged repo code and `ckpts/{entity,planet,fleet,comet}`
- `google-cloud-firestore` for `fs://` control
- `google-cloud-storage` or `gcloud` for GCS transport
- high open-file limit / torch `file_system` sharing for the COW pool path

## Monitor
`ppo_monitor.py <state>` (or a Colab cell) prints `STATE.phase/iter` plus learner
and per-actor progress every few seconds — a read-only dashboard runnable from
Colab or any rollout VM, since STATE is the single shared truth.
