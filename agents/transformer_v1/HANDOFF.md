# transformer_v1 — Handoff (PairScoreHead smoke test)

**Date written:** 2026-05-10
**Previous owner:** ppo-transformer-v1 / pair-score-head redesign session
**Branch / commit you should pull from:** `main` at the time you read this.

---

## Why this exists

The last PPO run on the multi-head `ActionDecoder` (`data/runs/ppo_lo_lr_20260507-023932`) finished with `eval_winrate = 0.0`, a degenerate fraction policy (`frac_log_std` pinned at the σ-floor, `frac_sample_mean ≈ 0.52`), and KL-early-stopping firing on every iteration. That symptom set is consistent with a head-side problem (multi-head joint distribution + premature optimisation) — but it could also be that the **encoder representation itself doesn't carry enough source-target signal** to support an expert action.

Before any further PPO work, we want a clean answer to one question:

> Can the current frozen encoder representation support direct expert `(source, target)` pair prediction from kovi's replays?

If yes → put NOOP / frac / value / PPO back on top. If the head can't even overfit a tiny subset → **the encoder/labels are the problem**, not PPO.

## Architecture (what's live now)

```
RAW OBS
  ▼
L-FEAT  Featurizer (CPU, no params)              [UNCHANGED]
L0      PlanetEncoder + FleetEncoder              [FROZEN]
L1      PlanetEntityEncoder (K=4 owner cells)     [FROZEN]
L2      CrossEntityAttention (3 layers, 4 heads)  [FROZEN]
                       ▼
L3      PairScoreHead   ← ONLY trainable thing
        h_ij = [glob ‖ ctx_i ‖ ctx_j ‖ ctx_i ⊙ ctx_j]
        s_ij = MLP(4·d → hidden → 1)              # (B, P, P)
        masked to -inf where ¬(src_valid_i ∧ tgt_valid_j)
                       ▼
LOSS    joint pair CE on flatten(P×P), y = src·P + tgt
        (acted rows only, expert = kovi)
```

`ActionDecoder`, `ContextualActionDecoder`, `PairActionDecoder`, `GlobalStateDecoder`, `LaunchOutcomeDecoder` and all their training/PPO/inference glue were **removed**. Recover them from git history (commit `f4056d3` and earlier) if needed.

`runner.py` and `ppo.py` raise `NotImplementedError("decoder removed; PairScoreStack is for representation validation only — runtime sampling not supported")` if you try to use the agent in a game. **Don't paper over that error** — fix it deliberately by adding a runtime decoder back (not the goal of this stage).

## Where things live

| Thing | Path |
|---|---|
| Pair-score code | `agents/transformer_v1/pretrain/pair_score.py` |
| Encoder modules (frozen) | `agents/transformer_v1/encoder/{fleet,planet,entity}_encoder.py`, `agents/transformer_v1/aggregator/cross_entity.py` |
| Dataset glue | `agents/transformer_v1/pretrain/expert_action.py` (only `ActionSnapshotDataset` + `FRAC_Z_MIN` survive) |
| Encoder reuse helper | `_entity_tokens_per_step` in `agents/transformer_v1/pretrain/cross_entity.py` |
| Original full design | `agents/transformer_v1/DESIGN.md` (describes the *spec*; code differs — see "Notable gap" near the bottom of that file) |
| Critique of 2026-05-04 | `agents/transformer_v1/CRITIQUE_2026_05_04.md` (untracked workspace note from earlier session — read for prior debugging context) |
| Encoder checkpoint | `data/runs/action/20260505-143435/action_best.pt` ← committed via `git add -f`, normally gitignored |
| Replays (per-player) | `data/replays/<player>/<replay_id>_<num_players>_<seat>.json.gz` — tracked in git |
| Dataset CSVs | `data/datasets/{action,planet,fleet,entity,cross_entity}/*.csv` — **gitignored**; regenerate via `scripts/build_encoder_dataset.py --num-episodes N` |
| Colab notebook | `notebooks/train_pair_score_colab.ipynb` |
| GCS bucket | `gs://orbit-wars-shipping/` — `code.tgz`, `data.tgz`, `weights.tgz`, `pair_score_assets.tgz` |

## Expert choice (locked-in — don't churn this without reason)

`kovi` — highest win-rate among the 5 sampled players. Numbers from the action CSV `winner_seat` field at session end (2026-05-10):

| player | replays | wins | win-rate |
|---|---:|---:|---:|
| **kovi** | **119** | **58** | **48.7%** |
| Shun_PI | 117 | 51 | 43.6% |
| bowwowforeach | 99 | 39 | 39.4% |
| Erfan Eshratifar | 73 | 26 | 35.6% |
| Orbital Occle | 238 | 78 | 32.8% |

Filter is implemented in `pair_score.discover_action_csvs(...)` via `--player kovi` (composes with `--filter winner|all`).

## How to run it

### Local CPU smoke test (5 min)

```bash
.venv/bin/python -m agents.transformer_v1.pretrain.pair_score \
    --encoder-ckpt data/runs/action/20260505-143435/action_best.pt \
    --player kovi --filter all \
    --max-rows 30 --overfit \
    --batch-size 16 --lr 1e-3 --epochs 3 \
    --device cpu \
    --out-dir /tmp/pair_score_smoke
```

This exercises the full pipeline (CSV → tensors → encoders → PairScoreHead → loss) on 30 acted rows. Loss should drop after a few epochs. Last verified working at session end (3 epochs, loss 4.94 → 4.81; gradient flowing through the head).

### Colab (Experiment 1 + 2, ~15 min on T4)

Open `notebooks/train_pair_score_colab.ipynb`. Prereq: `pair_score_assets.tgz` exists at `gs://orbit-wars-shipping/pair_score_assets.tgz`. If missing, build it locally:

```bash
tar czf /tmp/pair_score_assets.tgz \
    data/runs/action/20260505-143435/action_best.pt data/replays/kovi
gsutil cp /tmp/pair_score_assets.tgz gs://orbit-wars-shipping/
```

The notebook runs:
- **§5 Experiment 1** — 50 acted rows, train==val, 150 epochs (BLOCKING gate). Expected: `train_loss < 0.1`, `top1 > 0.9`.
- **§6 Experiment 2** — 5000 acted rows, 80/20 split, 10 epochs. Track `val_top1` vs `random_valid_top1`. Minimal success: ≥ 3× random. Strong success: top1 ≥ 0.30, top3 ≥ 0.55.

## Decision tree (what to do based on results)

| Outcome | Diagnosis | Next step |
|---|---|---|
| Experiment 1 fails (no overfit) | label/mask/flatten/encoder bug | **STOP and debug.** Run the unit tests in `pair_score.py` first (mask shape, label round-trip). Inspect `src_valid` / `tgt_valid` for the failing rows — the dataset has a known patch (`force-include expert source` in `ActionSnapshotDataset._build_snapshot`) that can mask other bugs. |
| Experiment 2 ≤ 1× random | encoder representation insufficient | Revisit **L1** (PlanetEntityEncoder — increase capacity, fix mask coverage, change pair feature) or **L2** (CrossEntityAttention — add layers/heads). Do NOT just retrain pair-head — it can't fix what isn't there. |
| Experiment 2 ≥ 3× random, top1 < 0.30 | weak signal | Try `--filter winner` on kovi (kovi-wins-only — cleaner labels). Try a bigger `--max-rows`. Try a wider `--hidden`. |
| Experiment 2 ≥ 3× random, top1 ≥ 0.30 | representation OK | Layer NOOP head + frac head + value head back on top. THEN retry PPO with the head-side fixes the prior PPO run flagged: lower policy-phase LR, reactivate BC anchor, raise σ-floor or remove it, gentler KL cap. |

## Constraints — things to **not** churn without thought

1. **Don't unfreeze the encoders.** The whole point is to test the existing representation. If you unfreeze, you're testing a different thing.
2. **Don't add more heads.** One head, one loss, one dataset filter. Adding NOOP/value/frac before the pair head shows signal makes failure modes ambiguous.
3. **Don't change the expert** unless Experiment 2 fails and you've already tried `--filter winner` on kovi. The win-rate ranking above is the fixed reference.
4. **Don't rewrite `runner.py` / `ppo.py`** to "fix" the NotImplementedError — they're stubbed deliberately. Re-introduce a runtime decoder only when you have a head that's worth deploying.
5. **`data/runs/` is gitignored** — the action ckpt at `data/runs/action/20260505-143435/action_best.pt` is the one exception, force-added so this handoff is self-contained. Future ckpts go via `gs://orbit-wars-shipping/` or a new force-add (don't lift the gitignore globally).

## Open threads / TODOs

1. **`featurizer/action_featurizer.py`** has a 155-line uncommitted modification on disk from before this session. Not mine; left untouched. Look at the diff (`git diff agents/transformer_v1/featurizer/action_featurizer.py`) before doing anything in that file — there's in-progress work there.
2. **Replays for `local`** in `data/replays/local/` are 8 episodes including a sub-directory `phy_phy_rnd_rnd_4p` that broke a one-shot Python iteration earlier. Skip it or guard for sub-directories if you walk that tree.
3. **Stale daemons** from the earlier PPO run may still be running on the user's machine: `progress_writer` (`pid 60327`) and `ppo_discord_watcher` (`pid 59113`), pointing at the dead PPO process pid 35542. Kill via `tmp/stop_ppo_training.sh` if you don't need them.
4. **`scripts/pack_for_gpu.sh`** doesn't include action ckpts or `data/replays/`. The Colab notebook now compensates by fetching `pair_score_assets.tgz` separately — but if you want a single tarball workflow, extend the pack script with `INCLUDE_ACTION=1` and `REPLAYS_PLAYER=<name>` flags.

## Trail of artifacts to look at before doing anything

| File | Why |
|---|---|
| `agents/transformer_v1/HANDOFF.md` | this memo |
| `agents/transformer_v1/CRITIQUE_2026_05_04.md` | most recent prior-session critique of the full stack |
| `tmp/ppo_training_progress.md` | last PPO run's iter-30 snapshot (eval_winrate=0.0) — explains why we're here |
| `agents/transformer_v1/DESIGN.md` | the original spec (note: code diverged into the v2-path; see `pair_score.py` for what's actually live) |
| `tmp/pair_actor_design.md` | older PairActionDecoder design, useful as a sketch of what a future runtime decoder could look like |
| `data/runs/ppo_lo_lr_20260507-023932/ppo_log.jsonl` | per-iter metrics from the failed PPO run |

## End-state at handoff

- `main` at commit (run `git log --oneline -5` to see). Pushed to `https://github.com/JerryIshihara/kaggle-orbit-wars`.
- Encoder ckpt + 120 kovi replays in repo. Datasets + run outputs on GCS.
- Pair-score smoke test runs end-to-end locally on CPU. Colab path verified up to the GCS-pull cell; full Colab run pending the user's `gsutil cp /tmp/pair_score_assets.tgz gs://orbit-wars-shipping/` step.
- No PPO process running. No live training jobs.

Good luck.
