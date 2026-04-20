# Orbit Wars Competition: Kernel Sources

| Rank | Title | Author | Votes | Last Run | URL | Summary |
|------|-------|--------|-------|----------|-----|---------|
| 1 | Getting Started | Bovard Doerschuk-Tiberi | 169 | 2026-04-18 | [Link](https://www.kaggle.com/code/bovard/getting-started) | Tutorial demonstrating nearest-planet greedy sniper heuristic; ignores travel-time and production; teaches API fundamentals. |
| 2 | [Score > 1,000] Orbit Wars: Structured Baseline | Pilkwang Kim | 85 | 2026-04-19 | [Link](https://www.kaggle.com/code/pilkwang/score-1-000-orbit-wars-structured-baseline) | 110-turn physics simulator + DP allocation; multi-phase strategy (expand → attack → finish); handles rotation, comets, sun-dodge. |
| 3 | Orbit Wars 2026 - Starter | sigmaborov | 64 | 2026-04-16 | [Link](https://www.kaggle.com/code/sigmaborov/orbit-wars-2026-starter) | Clean baseline with fleet-speed scaling, greedy targeting, sun detection; builds intuition for mechanics. |
| 4 | [LB - 958.1] Orbit Wars 2026 - Reinforce | sigmaborov | 57 | 2026-04-18 | [Link](https://www.kaggle.com/code/sigmaborov/lb-958-1-orbit-wars-2026-reinforce) | Policy gradient training on defense heuristics; multi-turn lookahead; incoming-threat simulation. |
| 5 | Orbit (Star) Wars - LB: MAX 1224 | Roman Tamrazov | 42 | 2026-04-20 | [Link](https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224) | **Highest-scoring entry (1224)**; 12+ strategies, adaptive 9-turn lookahead, 6-iteration lead-aim, threat modeling, dynamic urgency. |
| 6 | Orbit Wars 2026 - Tactical Heuristic | sigmaborov | 36 | 2026-04-16 | [Link](https://www.kaggle.com/code/sigmaborov/orbit-wars-2026-tactical-heuristic) | Core heuristic + learning loop; tracks enemy patterns, adapts allocation weights, maintains reserves. |
| 7 | ☀️ Orbit Wars: Sun-Dodging Baseline | Natsu Yamaguchi | 28 | 2026-04-17 | [Link](https://www.kaggle.com/code/debugendless/orbit-wars-sun-dodging-baseline) | Pure greedy with nearest-planet targeting, 1–2 turn prediction, sun-collision evasion, consolidation phase. |
| 8 | [LB - 928.7] Physics-Accurate Planner | sigmaborov | 25 | 2026-04-18 | [Link](https://www.kaggle.com/code/sigmaborov/lb-928-7-physics-accurate-planner) | Deterministic physics sim (exact arrivals, fleet scaling), dual-phase allocation (capture → reinforce). |
| 9 | [LB MAX SCORE 1000] - AGI IS HERE | bogdan janson | 21 | 2026-04-19 | [Link](https://www.kaggle.com/code/johnjanson/lb-max-score-1000-agi-is-here) | Adaptive turn-based strategy pivot (expand → attack → finish); dynamic urgency scaling. |
| 10 | Orbit Wars - Agent | Lakhindar Pal | 15 | 2026-04-17 | [Link](https://www.kaggle.com/code/lakhindarpal/orbit-wars-agent) | 100-turn forward oracle + tempo micro; rejects 1v1 heuristics; economic mapping, threat scoring. |
| 11 | Orbital Strategist — The Revolutionary Orbit Wars (KRONOS MCTS) | Dr/ameen Fayed | 12 | 2026-04-20 | [Link](https://www.kaggle.com/code/aminmahmoudalifayed/orbital-strategist-the-revolutionary-orbit-wars) | Full MCTS (~500 rollouts/decision) + deep RL value network; hybrid deterministic + stochastic expansion. |
| 12 | OrbitWars: Fusion Strats Timeline Defense | Artem Nazemtsev | 12 | 2026-04-18 | [Link](https://www.kaggle.com/code/artemnazemtsev/orbitwars-fusion-strats-timeline-defense) | 12+ strategies, adaptive 9-turn lookahead, 6-iteration lead-aim, comet-first, sun-dodge waypoints. |

---

## Notes

- **Votes** reflect community engagement (upvotes on Kaggle); higher vote count often correlates with clarity and reproducibility, not necessarily final LB score.
- **LB (Leaderboard) Scores** are reported within notebook titles; highest confirmed: **1224** (Tamrazov).
- **Last Run** dates indicate recent kernel activity (as of 2026-04-20 snapshot).
- **Sigma Borov** appears 5× with varied approaches, suggesting iterative refinement and exploration of the solution space.
