# Orbit Wars Competition: Competitor Research Methodologies

## Heuristic / Rule-Based

### ☀️ Orbit Wars: Sun-Dodging Baseline
- **Author**: Natsu Yamaguchi (debugendless)
- **Votes**: 28
- **Approach**: Pure greedy heuristic. Identifies nearest unowned planets, predicts arrival, accounts for sun collision. Targets weak neutrals early; broadcasts consolidation fleets mid-game. No prediction beyond 1–2 turns.
- **Link**: https://www.kaggle.com/code/debugendless/orbit-wars-sun-dodging-baseline

---

## Search / Monte Carlo Tree Search

### 🧠 KRONOS MCTS — Monte Carlo Tree Search + Deep RL Architecture
- **Author**: Dr/ameen Fayed (aminmahmoudalifayed)
- **Votes**: 12
- **Approach**: Full MCTS with ~500 simulation rollouts per decision. Deep RL value network trains on simulated futures to guide tree expansion. Hybrid deterministic + stochastic node exploration. Handles planet rotation and comet dynamics within simulation.
- **Link**: https://www.kaggle.com/code/aminmahmoudalifayed/orbital-strategist-the-revolutionary-orbit-wars

---

## Neural Networks / Deep RL

### [Score > 1,000] Orbit Wars: Structured Baseline
- **Author**: Pilkwang Kim (pilkwang)
- **Votes**: 85
- **LB Score**: 1000+
- **Approach**: Physics-accurate simulator (110-turn horizon) + dynamic programming for target allocation. Multi-layer heuristic: early-game expand neutrals, mid-game opportunistic captures, late-game finish. Sun-dodge via waypoint routing. Handles planet rotation, fleet speed scaling, comet collision. Allocation scoring balances production value, travel time, and game-time remaining.
- **Link**: https://www.kaggle.com/code/pilkwang/score-1-000-orbit-wars-structured-baseline

### Orbit (Star) Wars | LB: MAX 1224
- **Author**: Roman Tamrazov (romantamrazov)
- **Votes**: 42
- **LB Score**: 1224 (highest found)
- **Approach**: Highest-scoring entry. Advanced simulator with 12+ candidate strategies, adaptive lookahead (up to 9 turns). Lead-aim prediction with 6 iterative refinements. Comet prioritization, inner-planet dominance tactics. Multi-agent threat modeling + dynamic urgency weighting. Real-time strategy switching based on domination metrics.
- **Link**: https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224

### [LB - 958.1] Orbit Wars 2026 - Reinforce
- **Author**: sigmaborov
- **Votes**: 57
- **LB Score**: 958.1
- **Approach**: Multi-turn lookahead with incoming fleet simulation. Policy gradient training on defense-needed heuristics. Greedy allocation optimized by experience. State-space simplification: encode planet ownership, ships, incoming threats into reward signal for fleet dispatch decisions.
- **Link**: https://www.kaggle.com/code/sigmaborov/lb-958-1-orbit-wars-2026-reinforce

### [LB - 928.7] Physics-Accurate Planner
- **Author**: sigmaborov
- **Votes**: 25
- **LB Score**: 928.7
- **Approach**: Deterministic physics simulation (exact arrival times, fleet speed scaling, planet rotation). Candidate generation for all feasible (source, target, ship_count) tuples. Score each by: production × remaining_turns / (cost + travel_time). Dual-phase: allocation phase captures high-value targets; reinforcement phase consolidates rear lines toward front.
- **Link**: https://www.kaggle.com/code/sigmaborov/lb-928-7-physics-accurate-planner

### [LB MAX SCORE 1000] - AGI IS HERE
- **Author**: bogdan janson (johnjanson)
- **Votes**: 21
- **LB Score**: 1000
- **Approach**: Adaptive heuristic with turn-based strategy pivot. Early: greedy neutral expansion. Mid: multichannel attack on enemies with fleet consolidation. Late: finish-mode aggression (all-in on weak opponents). Incorporates advance travel-time prediction + dynamic urgency scaling as game time depletes.
- **Link**: https://www.kaggle.com/code/johnjanson/lb-max-score-1000-agi-is-here

### Orbit Wars - Agent (100-Step Forward Oracle)
- **Author**: Lakhindar Pal (lakhindarpal)
- **Votes**: 15
- **Approach**: 100-turn forward oracle + tempo-based micro. Rejects 1v1 heuristics; emphasizes opportunism and economic mapping. Multi-turn fleet routing with comet avoidance. Threat scoring by distance and urgency. Real-time adaptation to opponent fleets.
- **Link**: https://www.kaggle.com/code/lakhindarpal/orbit-wars-agent

### OrbitWars: Fusion Strats Timeline Defense
- **Author**: Artem Nazemtsev (artemnazemtsev)
- **Votes**: 12
- **Approach**: 12+ candidate strategies with adaptive simulator (up to 9 turns). Advanced lead-aim prediction (6 iterations). Comet-first priority, inner-planet domination tactics. Predictor with sun-dodging waypoints. Heuristic conflict resolution when multiple threats overlap.
- **Link**: https://www.kaggle.com/code/artemnazemtsev/orbitwars-fusion-strats-timeline-defense

---

## Hybrid / Multi-Method

### Orbit Wars 2026 - Tactical Heuristic
- **Author**: sigmaborov
- **Votes**: 36
- **Approach**: Core heuristic + learning loop. Tracks enemy patterns (opening, reaction times, preferred targets). Adapts fleet allocation weights based on opponent type. Maintains planet defense reserves; uses reserve surplus for opportunistic attacks. Combines rule-based tactics with experience-driven weighting.
- **Link**: https://www.kaggle.com/code/sigmaborov/orbit-wars-2026-tactical-heuristic

---

## Starter / Educational

### Getting Started (Tutorial)
- **Author**: Bovard Doerschuk-Tiberi (bovard)
- **Votes**: 169 (most popular)
- **Approach**: Educational baseline. Nearest-planet greedy sniper: find nearest unowned planet from each owned planet, send minimum ships to capture. Ignores travel time, does not account for production during transit. Demonstrates fundamental API use and angle computation.
- **Link**: https://www.kaggle.com/code/bovard/getting-started

### Orbit Wars 2026 - Starter
- **Author**: sigmaborov
- **Votes**: 64
- **Approach**: Clean starter baseline. Physics-aware fleet speed scaling. Simple greedy targeting (nearest unowned planet). Sun-collision detection. Minimal feature engineering; builds intuition for competition mechanics.
- **Link**: https://www.kaggle.com/code/sigmaborov/orbit-wars-2026-starter

---

## Summary Statistics

- **Total Kernels Reviewed**: 12
- **Most Common Approach**: Heuristic / Rule-Based with Physics Simulation (9/12)
- **Highest LB Score**: Roman Tamrazov @ **1224**
- **Strongest Tactic Across Entries**: Multi-turn lookahead with arrival-time prediction and sun-dodging waypoint routing
- **Least Common**: Pure MCTS (1 entry); Pure RL (0 entries)
