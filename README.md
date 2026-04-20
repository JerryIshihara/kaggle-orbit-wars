# kaggle-orbit-wars

Agent development for the Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) competition.

## The game

Orbit Wars is a real-time strategy game on a 100×100 continuous 2D board with a sun at the center, played by 2 or 4 agents over 500 turns. Each player starts with one home planet and sends fleets to capture neutral and enemy planets.

- **Planets** produce 1–5 ships/turn. Inner planets orbit the sun; outer ones are static.
- **Fleets** travel in straight lines. Speed scales with size (1 ship = 1.0/turn, ~1000 ships = 6.0/turn). Fleets crossing the sun are destroyed.
- **Comets** spawn in groups of 4 at steps 50/150/250/350/450 on elliptical trajectories.
- **Win condition**: most total ships (on planets + in fleets) at step 500, or last player standing.

Full rules: see Kaggle `README.md` and `agents.md` (fetched via `kaggle competitions download orbit-wars -f ...`).

## Competition details

- **Prize pool**: $50,000 USD (Featured competition)
- **Deadline**: 2026-06-23
- **Teams**: 521+
- **Submission**: a `main.py` with an `agent(obs)` function returning `[[from_planet_id, angle, num_ships], ...]`

## Repo layout (planned)

- `agents/` — agent implementations
- `sim/` — local simulation / evaluation harness on top of `kaggle-environments`
- `notebooks/` — exploration and replay analysis

## Getting started

```bash
pip install kaggle kaggle-environments

# Credentials are read from ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY env vars.

# Run the baseline agent locally
python -c "from kaggle_environments import make; env = make('orbit_wars', debug=True); env.run(['main.py', 'random']); print([(i, s.reward) for i, s in enumerate(env.steps[-1])])"

# Submit
kaggle competitions submit orbit-wars -f main.py -m "baseline"
```
