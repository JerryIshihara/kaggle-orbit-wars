# Colab distributed-PPO bring-up — issues & fixes

Every failure hit bringing up `notebooks/ppo_distributed_colab.ipynb` (Colab learner +
GCP `e2-medium` rollout VM + GCS transport), with root cause and fix. Most were **environment /
packaging** problems, not logic bugs — and several were **silently swallowed** (`check=False`,
`strict=False`, default args) so they surfaced one cell later than where they originated.

If you're setting this up fresh, read **§0 Working configuration** + **§6 Pre-flight checklist**
first; the rest is the post-mortem.

---

## 0. Working configuration (the values that must be right)

| Knob | Correct value | Why |
|---|---|---|
| `PROJECT` | `analog-receiver-489214-e9` + `gcloud config set project` | Colab auth sets **no** default project; `gcloud compute` needs one |
| `SSH_USER` | `ppo` (any **non-root**) | Colab runs as root; GCE doesn't provision root SSH |
| `code.tgz` / `weights.tgz` | from `gs://orbit-wars-shipping/**entity/**` | `entity/weights.tgz` is the **flat** bundle *with comet*; the root one is nested + comet-less |
| `kaggle_environments` | `pip install` on **both** Colab and the VM | importing the `agents` package imports the env at module load |
| VM image / deps | `debian-12`, pip with `--break-system-packages` | Debian 12 is PEP 668 "externally managed" |
| VM scopes | `--scopes storage-rw` | so the VM's default SA can `gcloud storage cp` |
| Critic | none (`allow_debug_glob_critic=True`) | the L3L4 actor has no `PlayerConsolidator` → glob value head |

---

## 1. Staging — code & dependencies

### 1.1 `ImportError: cannot import name 'shards' from '...ppo' (.../ppo.py)`
- **Cause:** the `code.tgz` on GCS was **stale** — it contained the old single-file `ppo.py` stub
  instead of the `ppo/` **package** (where `shards.py` / `rollout_worker.py` / `learner_step.py` live).
- **Fix:** rebuild `code.tgz` from the current repo and upload it:
  ```bash
  tar czf code.tgz --exclude='__pycache__' --exclude='*.pyc' agents app requirements.txt run.py scripts utils
  gcloud storage cp code.tgz gs://orbit-wars-shipping/entity/code.tgz
  ```
- **Lesson:** `code.tgz` is a **snapshot** of the repo. Any change to `agents/` or `scripts/` means
  rebuild + re-upload. The notebook pulls `entity/code.tgz`; the VM bootstrap pulls the same.

### 1.2 `ModuleNotFoundError: No module named 'kaggle_environments'`
- **Cause:** `import agents` pulls in agent modules (e.g. `agents/mcts_v1/agent.py`) that
  `from kaggle_environments... import Fleet, Planet` **at module load** — but the notebook never
  installed it. (Importing the package needs it even though the learner doesn't run the env.)
- **Fix:** `pip install -q kaggle_environments` **before** `import agents` (notebook cell §2). The VM
  needs it too (it actually runs the env) — handled in `ppo_vm_bootstrap.sh`.

### 1.3 L0 encoders never staged / `comet_past_best.pt` missing
- **Symptom:** `load_supervised` can't find encoders (or silently uses default dirs).
- **Cause:** the **root** `weights.tgz` is the `pack_for_gpu.sh` layout — nested
  `data/runs/<kind>/<ts>/...` and **no comet encoder** — but cell §2 expects **flat**
  `planet/fleet/comet_*_best.pt` at the cwd. The flat copy silently no-ops (`if (WORK/src).exists()`).
- **Fix:** pull from `gs://orbit-wars-shipping/**entity/**weights.tgz` (the `pack_entity_for_colab.sh`
  flat bundle, which includes comet). **Two `weights.tgz` conventions exist** — use the `entity/` one for Colab.

### 1.4 `CalledProcessError` on `gcloud storage cp .../SET_ME_.../cross_entity_best.pt`
- **Cause:** `CRITIC_RUN` was a `SET_ME_...` placeholder → the path didn't exist.
- **Fix:** this pipeline dropped the separate critic entirely (see §2.1), so there's no `CRITIC_RUN` to set.

---

## 2. Model / API mismatches

### 2.1 `ValueError: L1-branch critic_model is no longer supported.`
- **Cause:** `PPOActorCritic.__init__` was refactored — it **rejects** `critic_model`. The critic is
  now the value head **inside** `PPOActorCritic` reading the actor's post-L2 `player_state`
  (`actor_critic.py:194`). The notebook (and a stale `train.py` path) still built a
  `CrossEntityCriticModel` and passed `critic_model=`.
- **Fix:** `policy = PPOActorCritic(entity_model, sigma=SIGMA, allow_debug_glob_critic=True)` — no
  `critic_model`, no separate critic ckpt. Matches `smoke.py`.
- **Consequence:** the **L3L4 actor has no `PlayerConsolidator`**, so `player_state` is `None` and the
  critic falls back to the **glob `value_head`** (trained from scratch in PPO). The strong
  player_state critic would need a consolidator-equipped actor (none exists yet with L3/L4).

### 2.2 PPO trains the value head on a *random* consolidator
- **Cause:** `load_supervised` always built `EntityPretrainModel(...)` with the **default
  `with_consolidator=True`**, then `load_state_dict(strict=False)`. For a no-consolidator actor ckpt,
  the missing module stayed at **fresh random init** — and `actor_critic.py:194` then used that garbage
  `player_state` instead of the glob fallback.
- **Fix:** respect the ckpt — detect `with_consolidator` / `skip_l34` from the saved config **or** the
  state-dict keys (mirror `TransformerAgent.load`, `runner.py:268-286`):
  ```python
  with_consolidator = bool(cfg["with_consolidator"]) if "with_consolidator" in cfg \
      else any(k.startswith("consolidator.") for k in ckpt["model"])
  ```
- **Lesson:** `strict=False` + a default-on submodule = silent random-init. Always gate optional
  submodules on the ckpt.

---

## 3. GCP / gcloud on Colab

### 3.1 `ERROR: The required property [project] is not currently set`
- **Cause:** Colab's `auth.authenticate_user()` authenticates but sets **no default project**.
  `gcloud storage` infers it from the bucket (so staging worked), but `gcloud compute`
  (create/ssh/describe) **requires** an explicit project. This was the real root cause of the create
  failing every time.
- **Fix:** set it once after auth:
  ```python
  PROJECT = 'analog-receiver-489214-e9'
  subprocess.run(['gcloud','config','set','project',PROJECT], check=True)
  ```

### 3.2 VM create failed *silently* → every later SSH failed
- **Cause:** the create cell used `subprocess.run(..., check=False)`, so the §3.1 project error was
  **swallowed**. No VM was created, yet the notebook proceeded to SSH (which then failed with the
  confusing "instance not found" / "Permission denied").
- **Fix:** capture output + **retry** + verify `RUNNING` before SSH; raise with the real error if create
  truly fails (quota / billing / org policy / zone capacity):
  ```python
  cr = subprocess.run([...create...], capture_output=True, text=True); print(cr.stdout+cr.stderr)
  ```
- **Lesson:** never `check=False` on a step whose failure invalidates everything after it.

### 3.3 SSH `root@VM: Permission denied (publickey)`
- **Cause:** **Colab runs as root**, so `gcloud compute ssh <instance>` defaults to the local
  username `root`. GCE's guest agent provisions metadata SSH keys for **non-root** users only — root
  login isn't set up — so the propagated key is rejected. (It worked from a dev Mac because that runs
  as a non-root user.)
- **Fix:** SSH as an explicit non-root user: `gcloud compute ssh ppo@<instance> ...`. The guest agent
  provisions `ppo` from the key; home dir `/home/ppo` is what the bootstrap/daemon use (`$HOME`-relative).

### 3.4 First-SSH `Connection refused` / `key has not propagated`
- **Cause:** `instances create` returns before the VM finishes booting and before the SSH key
  propagates (~30-90s on a fresh VM).
- **Fix:** retry SSH with backoff (the §5 `ssh()` helper retries 9× / 20s). `Connection refused` →
  booting; `Permission denied` after propagation → almost always the **username** issue (§3.3), not timing.
- **Firewall (verified OK here):** the VPC had both `default-allow-ssh` (0.0.0.0/0 tcp:22) and
  `allow-iap-ssh` (35.235.240.0/20). If yours lacks an SSH rule, that's a separate persistent failure.

---

## 4. VM bootstrap (Debian 12)

### 4.1 `error: externally-managed-environment`
- **Cause:** Debian 12's system Python is **PEP 668 "externally managed"** — `pip install` system-wide
  is blocked.
- **Fix:** `--break-system-packages` (or a venv). In `ppo_vm_bootstrap.sh`:
  ```bash
  export PIP_BREAK_SYSTEM_PACKAGES=1
  python3 -m pip install -q --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
  python3 -m pip install -q --break-system-packages kaggle_environments numpy psutil
  ```

---

## 5. Orchestration / daemon

### 5.1 Re-running §5 spawns a *second* poll-daemon
- **Cause:** the daemon launch is `nohup ... &` with no guard, so re-running §5 leaves the old daemon
  running too — two daemons race on the same `rollouts/vK/` (double rollouts / clobbered shards).
- **Fix:** `pkill -f 'agents.transformer_v2.ppo.vm_daemon'` before `nohup` in `ppo_vm_daemon.sh`.
- **Related:** keep `RUN_ID` stable — the daemon watches `ppo/<RUN_ID>/heads/`; re-running cell §0
  changes the timestamped `RUN_ID` and orphans a running daemon.

---

## 6. Pre-flight checklist

Before running the notebook end-to-end:

- [ ] `PROJECT` set + `gcloud config set project` in the auth cell (§3.1).
- [ ] `SSH_USER` is non-root (§3.3).
- [ ] `code.tgz` rebuilt from the current repo if any `agents/`/`scripts/` changed (§1.1).
- [ ] code+weights pulled from the **`entity/`** prefix; `weights.tgz` is flat **with comet** (§1.3).
- [ ] `pip install kaggle_environments` runs before `import agents` (§1.2).
- [ ] bootstrap uses `--break-system-packages` (§4.1).
- [ ] daemon launch `pkill`s any prior daemon; `RUN_ID` kept stable (§5.1).
- [ ] critic path: `allow_debug_glob_critic=True`, no `critic_model`; `load_supervised` respects the
      ckpt's `with_consolidator`/`skip_l34` (§2).
- [ ] **stop the VM when done:** `gcloud compute instances stop orbit-wars-ppo-actor --zone us-central1-b`.

---

## 7. Meta-lessons

1. **Silent swallowing hid most of these.** `check=False`, `strict=False`, and default-on args turned
   a clear error into a confusing one a cell or two later. Fail loud at the source.
2. **`*.tgz` bundles are snapshots, not live code.** Stale/`code.tgz` and wrong-layout `weights.tgz`
   caused two separate "it ran fine locally" surprises.
3. **Colab's `gcloud` ≠ a dev box's.** No default project, and it runs as **root** — the two GCP issues
   both stem from this and don't reproduce locally.
4. **A real end-to-end dry-run beats static checks.** Syntax/flag/symbol checks all passed while the
   bundle, project, and SSH-user issues remained — only actually *running* each cell surfaced them.
