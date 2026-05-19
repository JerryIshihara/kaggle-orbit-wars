import math
import random

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from ...registry import register

LAUNCH_PROB = 0.1


@register(
    "random_v1",
    "Random valid moves — each owned planet has a 10% chance to launch a "
    "random-sized fleet at a uniformly random angle.",
)
def random_valid_agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    for p in planets:
        if p.owner != player or p.ships < 1:
            continue
        if random.random() < LAUNCH_PROB:
            angle = random.uniform(0, 2 * math.pi)
            num = random.randint(1, int(p.ships))
            moves.append([p.id, angle, num])

    return moves
