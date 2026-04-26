from kaggle_environments import make
import pandas as pd

# encoder class

# decoder class: actor
# decoder class: critic


def _summarize_field_value(v):
    shape = None
    length = None
    dtype = type(v).__name__

    # numpy / torch / array-likes often expose shape
    if hasattr(v, "shape"):
        try:
            shape = tuple(v.shape)
        except Exception:
            shape = None

    if hasattr(v, "__len__"):
        try:
            length = len(v)
        except Exception:
            length = None

    if length is None and shape is not None and len(shape) > 0:
        length = shape[0]

    if length is None:
        length = 1

    return dtype, shape, length


def _get_player0_observation(obs):
    """
    Orbit Wars observations sometimes come as:
    - per-agent dicts directly, or
    - a dict with a 'players' array/list of per-player observations.
    This tries both.
    """
    if isinstance(obs, dict) and "players" in obs and isinstance(obs["players"], (list, tuple)) and obs["players"]:
        return obs["players"][0]
    if isinstance(obs, (list, tuple)) and obs:
        return obs[0]
    return obs


if __name__ == "__main__":
    env = make("orbit_wars", debug=True)
    print(f"Environment: {env.name} v{env.version}")
    print(f"Players: {env.specification.agents}")
    print(f"Max steps: {env.configuration.episodeSteps}")
    env.run(["random", "random", "random", "random"])


    header = ["step", "player", "angular_velocity", "len_comet_planet_ids", "len_planets", "len_fleets", "len_comets"]
    df = pd.DataFrame(columns=header)
    agent_idx = 0
    rows = []

    for step_idx in range(len(env.steps)):
        obs = env.steps[step_idx][agent_idx].observation
        rows.append({
            "step": step_idx,
            "player": agent_idx,
            "angular_velocity": obs["angular_velocity"],
            "len_comet_planet_ids": len(obs["comet_planet_ids"]),
            "len_planets": len(obs["planets"]),
            "len_fleets": len(obs["fleets"]),
            "len_comets": len(obs["comets"]),
        })

    df = pd.DataFrame(rows)

    print(df.head(500))
