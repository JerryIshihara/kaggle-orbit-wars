# transformer_v2 — Design

Encoding the Orbit Wars observation for a transformer-based agent.

**Core decision: fold fleet info into per-planet tokens** rather than treating fleets as a separate token stream. Decisions in this game are per-planet ("where do I launch from / attack / hold?"), and the fleet-level info that matters for those decisions is *aggregate* ("how threatened is this planet", "how soon does help arrive"). Individual fleet trajectories don't drive policy at the planet level, and pairing fleets to their target planets is something we can compute analytically rather than asking attention to learn implicitly.

This gives a clean **49-token sequence** (P + 1 global) instead of the 145 you'd get with a separate fleet stream — ~9× less attention compute per layer, no fleet-truncation policy needed, and a strong inductive bias toward the way the policy actually works.

## Per-planet token (raw → projected to d_model)

Planets carry their own state plus pre-aggregated fleet pressure into and out of them. ~26 dims:

```
planet_features (per planet, ~26 dims):
  # --- identity / kinematics ---
  x / 100, y / 100,                          # position
  radius / 3,                                # 1 + ln(production), max ~2.6
  log1p(ships) / log(5000),                  # current ship count
  production / 5,                            # 1..5
  is_orbiting,                               # 0/1
  is_comet,                                  # 0/1

  # --- ownership (relative slot, see below) ---
  is_self, is_enemy_a, is_enemy_b, is_enemy_c, is_neutral,

  # --- inbound fleet pressure (per-owner aggregates) ---
  inbound_my_ships / 100,
  inbound_enemy_a_ships / 100,
  inbound_enemy_b_ships / 100,
  inbound_enemy_c_ships / 100,
  nearest_my_inbound_eta / 50,               # soonest friendly arrival (clipped to 1.0 if none)
  nearest_enemy_inbound_eta / 50,            # soonest hostile arrival
  n_inbound_total / 30,                      # how many fleets aimed here at all

  # --- outbound (mostly meaningful for our own planets) ---
  outbound_ships / 100,                      # ships in flight FROM this planet right now
  n_outbound_total / 10,

  # --- predicted state at a fixed horizon (analytical, NOT learned) ---
  predicted_garrison_at_h20 / 100,           # via predict_garrison_at_arrival(planet, eta=20, ...)
  predicted_owner_self_at_h20,               # 0/1 — will I own this in 20 turns?
  predicted_owner_enemy_at_h20,              # 0/1

  # --- temporal deltas (Option A: prev obs cached in agent) ---
  ships_delta_last_turn / 100,
  owner_changed_last_turn,                   # 0/1
  turns_since_owner_change / 50,             # state-tracked
```

The owner is encoded **relative to the learner's slot**: `rel = (owner − learner_slot) % num_players`, then one-hot `(is_self, is_enemy_a, is_enemy_b, is_enemy_c, is_neutral)`. In 2-player only `is_self`, `is_enemy_a`, and `is_neutral` ever fire — the policy stays valid for both modes without retraining.

### Computing the fleet aggregates

For each planet `p`, walk every in-flight fleet `f` once:

1. Compute `eta = _fleet_eta_to_planet(f, p)` (already in `physics_utils.py`).
2. If `eta is None`, the fleet's trajectory misses `p` — skip.
3. Find the planet with the *smallest* such ETA across all candidates → that's `f`'s target.
4. Increment `inbound_<owner>_ships` and `n_inbound_total` on the target planet.
5. Increment `outbound_ships` and `n_outbound_total` on the planet whose `id == f.from_planet_id`.

Cost: `O(F · P)` per step. With F ≤ 500 and P ≤ 40, that's ~20k ops — negligible compared to one Python frame of `featurize`.

`predict_garrison_at_h20` and `predicted_owner_*_at_h20` come from `predict_garrison_at_arrival(planet, eta=20, player, fleets)` already implemented in `physics_utils.py`. Two minor changes there: (1) also return the predicted owner, not just the count; (2) clamp eta to 20 turns so the per-planet feature has a consistent horizon.

### Pre-spawn comet caveat

`is_comet=1` fires only for comets that have *spawned* (so they appear in `obs["planets"]` with an entry in `obs["comet_planet_ids"]`). Pre-spawn comet positions are unknowable to any agent — heuristic or learned — so we don't try to encode them. Once spawned, a comet's full path is visible in `obs["comets"]`; the model's `(x, y)` features carry current position and the inbound aggregates flow naturally. Looking up future comet positions via the path index is an optional refinement (skipped in v1).

## Token assembly

```
B = batch size, P = MAX_PLANETS = 48

planet_tok: (B, P, d_model)         # planet_proj(features); planet_mask: (B, P)
scalar_tok: (B, 1, d_model)         # global features projected to one token
                                    # — step/500, remaining/500, total ships,
                                    # angular_velocity, comet_count/20,
                                    # my_total_ships, enemy_total_ships,
                                    # n_fleets_total / 100,
                                    # n_my_fleets_in_flight / 100,
                                    # n_enemy_fleets_in_flight / 100

tokens = concat([planet_tok, scalar_tok], dim=1)   # (B, P+1, d_model)
mask   = concat([planet_mask, ones], dim=1)        # (B, P+1)
```

## Self-attention block

```
encoded = TransformerEncoder(
  tokens,
  src_key_padding_mask = ~mask,    # bool: True where padded
  num_layers           = 4,
  nhead                = 4,
  dim_feedforward      = 4 * d_model,
)
```

What attention learns naturally without further supervision:

- **Coalition pairing**: planet tokens with high `inbound_my_ships` attend to other planets I might support, learning evacuation / consolidation patterns. Multi-source swarm signatures emerge from `outbound_ships` on several of my planets pointing toward the same target whose `nearest_enemy_inbound_eta` is short.
- **Frontier defense**: planets with low `nearest_enemy_inbound_eta` (under threat) get attended to by safer rear planets with surplus `outbound_ships`, learning the reinforce-from-rear pattern.
- **Phase awareness**: the global scalar token carries `step/500` and `remaining/500` and feeds back into every planet via cross-attention; the model can switch doctrines (expand → consolidate → race) without per-phase tuning.

## Self-attention block

```
encoded = TransformerEncoder(
  tokens,
  src_key_padding_mask = ~mask,    # bool: True where padded
  num_layers           = 4,
  nhead                = 4,
  dim_feedforward      = 4 * d_model,
)
```

What attention learns naturally without further supervision:

- **Threat detection**: enemy fleet token attends to your planet tokens → "this fleet → that planet" relationship.
- **Coordination**: multi-head heads can specialize — one head for fleet/planet pairs, one for fleet/fleet (incoming clusters), one for own/enemy proximity.
- **Sun avoidance**: positions encode this; attention to the (engineered) sun-distance feature on each fleet biases away.

## Action head per owned planet

For each owned planet `i` (we already know which tokens are ours via `is_self`):

```
my_tokens    = encoded[is_self_planets]      # (B, M, d_model)
ctx          = mean_pool(encoded, mask)      # (B, d_model)
h            = MLP(concat(my_tokens, ctx))   # (B, M, d_model)

launch_logit = Linear(h, 1)                  # Bernoulli per planet
target_query = Linear(h, d_model)            # query
target_keys  = Linear(encoded_planets, d_model)  # keys, all planets
target_logit = einsum('bmd,bpd->bmp', target_query, target_keys)  # M×P scores
ship_logit   = Linear(h, 1)                  # → sigmoid → frac mean
```

Target uses **dot-product attention over planets** instead of a 2500-cell cell-grid — way smaller action space (40 vs 2500) and semantically meaningful (you target a planet, not empty space).

## Practical points

- **Sequence length**: P + 1 = 49 tokens. About 9× cheaper attention per layer than the alternative fleet-stream design (145 tokens).
- **No fleet truncation needed**: with fleets folded into per-planet aggregates, 4-player end-game blowouts (max 446 fleets in our public replays) cost the same in attention as a quiet opening — the aggregation is `O(F·P)` Python work outside the model, then it's just 49 tokens.
- **Padding**: planets are zero-padded to `P=48` (real games have 16–40); `src_key_padding_mask` skips padded slots in attention.
- **Compute cost**: `O((P+1)² · d_model)` per layer. With P+1 = 49, d_model = 64, 4 layers ≈ ~600K FLOPs per forward. Trivial on GPU, fast on CPU; per-forward stays well under the 1s `actTimeout` even with comfortable margin for the rest of `act()`.
- **Sample efficiency vs CNN**: a transformer with this entity encoding gets the **4-fold mirror symmetry of orbit_wars for free** (set processing is permutation-invariant; attention treats `(x, y)` and `(100−x, 100−y)` symmetrically because position is a feature, not a tensor index). The CNN needs explicit data augmentation to match.

## v2 path: re-introduce fleet tokens if needed

If PPO eventually plateaus on tasks that genuinely need *fleet-level* reasoning the v1 aggregation collapsed — e.g., predicting a specific multi-fleet interception window, modeling what an opponent's specific fleet will do mid-flight, decoding swarms by their pairwise composition — promote fleets back to tokens with the original 13-dim feature vector and concat them after the planet stream. Keep the planet aggregates regardless: they're cheap and don't hurt. v2 sequence length: `P + F + 1 ≈ 145`. The truncation policy (rank by `eta_to_nearest_my_planet`, keep top-96) becomes relevant again at that point.

## Where the per-planet fleet aggregates pay off

The `inbound_*_ships`, `nearest_*_inbound_eta`, and `predicted_garrison_at_h20` features are the highest-leverage non-obvious adds. Without them, the model would have to:

1. Recover per-fleet identities from a separate token stream (extra sequence length).
2. Cross-attend fleets ↔ planets to figure out which fleet is heading where.
3. Sum ships per target in latent space.
4. Compute production-during-travel + combat resolution arithmetic in latent space.

That's a lot of capacity burned on perception the policy can't reason about until it converges. With them, "is this planet under threat" / "who will own this planet in 20 turns" / "how much help do I have arriving" is one feature lookup — the model spends its capacity on the *strategic* decisions ("given this much threat, do I retreat or counter?") instead of the perceptual extraction.

This is the same trick the heuristic agents already exploit (Frizzer's threat model, Pascal's wave-strike timing, ykhnkf's frontier scoring) — just baked into observations instead of rules. Single pass over `obs["fleets"]` per turn, `O(F·P)` Python ops, computed once in `featurize`.

## Temporal information

The v1 encoder is **single-frame** (one turn's obs in, action distribution out). No recurrence, no stacked frames. That's a deliberate simplification, justified by Orbit Wars' deterministic dynamics:

- **Fleet age** is recoverable from `distance(fleet, from_planet) / speed` — no history needed.
- **Planet trajectories** are fully determined by `(initial_position, angular_velocity)`; future positions are projectable from any single frame.
- **Future fleet positions** are linear extrapolation; the encoder doesn't need previous frames to predict where it's going.

What single-frame *can't* capture: opponent strategy patterns ("has this opponent been hoarding?"), your own multi-turn intent ("which of my fleets are coordinated?"), and recent ownership changes ("did this planet flip last turn?"). For each, three options ordered by complexity:

### A. Engineered delta features (recommended for v1)

Stash the previous obs in the agent instance, diff against current at featurize time, append per-entity delta scalars:

```
fleet (+2 dims):
  spawned_this_turn   ∈ {0, 1}
  age_turns / 50      = distance(fleet, from_planet) / (speed · 50)

planet (+3 dims):
  ships_delta_last_turn / 100                    ← reveals reinforcement / capture
  owner_changed_last_turn ∈ {0, 1}
  turns_since_owner_change / 50                  ← also state-tracked
```

Cost: zero training-loop plumbing (the agent just remembers last obs), small fixed bump to token feature dim. Captures the "what just happened" signal that drives most short-horizon decisions (intercepting a freshly-launched threat, defending a recently-flipped planet, exploiting an opponent's depleted production).

### B. Stacked frames (Atari-style)

Concatenate the last `K=4` frames of features per token. Token raw dim multiplies by K; encoder size unchanged because projection still maps to `d_model`. No state plumbing needed in training (pure feed-forward), but token-feature dim grows linearly and the encoder must learn the temporal correlations from scratch.

Better than A when arbitrary-horizon recurrences matter; worse when most info is local. For Orbit Wars, A captures ~95% of what B would.

### C. Recurrent layer (real temporal memory, v2 territory)

Pool encoder output to one `d_model` vector per turn → feed to a `GRU` with small hidden state (`h_rnn = 64`) → condition the action head on the recurrent state:

```
encoded      = TransformerEncoder(tokens, mask)       # (B, 145, d_model)
ctx_t        = masked_mean(encoded, mask)             # (B, d_model)
h_t, state   = GRU(ctx_t, prev_state)                 # (B, h_rnn), → state for next turn
my_h         = MLP(concat(my_planet_tokens, h_t))     # (B, M, d_model)
... action heads on my_h ...
```

Required training-loop changes:
- Rollout buffer carries `state` per turn alongside (obs, action, reward, value).
- Episode boundary resets `state` to zero.
- BPTT over truncated rollout segments; or detach state and treat each step as iid (cheaper, less expressive).

Worth the complexity only if behavior actually depends on long history. For orbit_wars (Markov-ish given current obs + delta features), probably not for v1.

### Recommendation

**v1: Option A.** Stash previous obs in the agent's `__call__` closure, compute deltas inline, append to per-entity feature vectors. No model architecture change beyond the input dim bump. v2 can grow into B or C if PPO plateaus on opponent-modeling tasks.
