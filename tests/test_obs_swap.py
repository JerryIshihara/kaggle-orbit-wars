import math

from agents.transformer_v2.featurizer.swap import (
    ClockwiseSeatSwap,
    clockwise_owner_map,
    clockwise_owner_map_from_initial_obs,
    infer_num_players_from_obs,
    spatial_rotation_from_initial_obs,
    swap_observation_seats,
)


def _obs(step=0, player=0):
    return {
        "step": step,
        "player": player,
        "planets": [
            [10, 0, 10.0, 10.0, 3.0, 50, 1],
            [11, 1, 90.0, 10.0, 3.0, 50, 1],
            [12, 2, 90.0, 90.0, 3.0, 50, 1],
            [13, 3, 10.0, 90.0, 3.0, 50, 1],
            [14, -1, 50.0, 50.0, 3.0, 20, 0],
        ],
        "initial_planets": [
            [10, 0, 10.0, 10.0, 3.0, 50, 1],
            [11, 1, 90.0, 10.0, 3.0, 50, 1],
        ],
        "fleets": [
            [100, 0, 20.0, 20.0, 0.0, 10, 12],
            [101, 3, 80.0, 80.0, 3.14, 13, 7],
        ],
        "comets": [
            {
                "path_index": 0,
                "planet_ids": [14],
                "paths": [[[10.0, 50.0], [50.0, 10.0]]],
            },
        ],
    }


def test_clockwise_owner_map():
    assert clockwise_owner_map(2) == (1, 0)
    assert clockwise_owner_map(4) == (1, 2, 3, 0)


def test_clockwise_owner_map_from_initial_positions():
    obs = {
        "step": 0,
        # Screen-clockwise physical order is owners 2, 0, 3, 1.
        "initial_planets": [
            [20, 2, 10.0, 10.0, 3.0, 50, 1],
            [21, 0, 90.0, 10.0, 3.0, 50, 1],
            [22, 3, 90.0, 90.0, 3.0, 50, 1],
            [23, 1, 10.0, 90.0, 3.0, 50, 1],
        ],
    }
    assert clockwise_owner_map_from_initial_obs(obs, num_players=4) == (3, 2, 0, 1)
    assert math.isclose(
        spatial_rotation_from_initial_obs(obs, (3, 2, 0, 1), num_players=4),
        math.pi / 2,
        abs_tol=1e-6,
    )


def test_swap_observation_seats_remaps_player_owners_and_space_without_mutation():
    obs = _obs()
    swapped = swap_observation_seats(obs, (1, 2, 3, 0))

    assert swapped["player"] == 1
    assert [p[1] for p in swapped["planets"]] == [1, 2, 3, 0, -1]
    assert [p[1] for p in swapped["initial_planets"]] == [1, 2]
    assert [f[1] for f in swapped["fleets"]] == [1, 0]
    assert _xy(swapped["planets"][0]) == (90.0, 10.0)
    assert _xy(swapped["initial_planets"][0]) == (90.0, 10.0)
    assert _xy(swapped["fleets"][0]) == (80.0, 20.0)
    assert math.isclose(swapped["fleets"][0][4], math.pi / 2, abs_tol=1e-6)
    assert _xy(swapped["comets"][0]["paths"][0][0]) == (50.0, 10.0)
    assert _xy(swapped["comets"][0]["paths"][0][1]) == (90.0, 50.0)

    assert obs["player"] == 0
    assert [p[1] for p in obs["planets"]] == [0, 1, 2, 3, -1]
    assert [f[1] for f in obs["fleets"]] == [0, 3]
    assert _xy(obs["planets"][0]) == (10.0, 10.0)
    assert math.isclose(obs["fleets"][0][4], 0.0, abs_tol=1e-6)


def test_swap_observation_seats_can_disable_spatial_rotation():
    obs = _obs()
    swapped = swap_observation_seats(obs, (1, 2, 3, 0), rotate_spatial=False)
    assert [p[1] for p in swapped["planets"]] == [1, 2, 3, 0, -1]
    assert _xy(swapped["planets"][0]) == (10.0, 10.0)
    assert _xy(swapped["fleets"][0]) == (20.0, 20.0)
    assert math.isclose(swapped["fleets"][0][4], 0.0, abs_tol=1e-6)


def test_clockwise_seat_swap_persists_mapping_across_steps():
    layer = ClockwiseSeatSwap(num_players=4)
    out0 = layer.apply(_obs(step=0, player=2))
    out1 = layer.apply(_obs(step=3, player=2))

    assert layer.owner_map == (1, 2, 3, 0)
    assert math.isclose(layer.rotation_radians, math.pi / 2, abs_tol=1e-6)
    assert out0["player"] == 3
    assert out1["player"] == 3
    assert _xy(out0["planets"][0]) == (90.0, 10.0)
    assert _xy(out1["planets"][0]) == (90.0, 10.0)
    assert layer.last_step == 3


def test_clockwise_seat_swap_resets_when_step_moves_backward():
    layer = ClockwiseSeatSwap(num_players=4)
    layer.apply(_obs(step=7, player=0))
    layer.owner_map = (0, 1, 2, 3)
    out = layer.apply(_obs(step=0, player=0))

    assert layer.owner_map == (1, 2, 3, 0)
    assert out["player"] == 1


def test_clockwise_seat_swap_defers_untyped_empty_observation():
    layer = ClockwiseSeatSwap()
    empty = {
        "step": 0,
        "player": 0,
        "planets": [],
        "initial_planets": [],
        "fleets": [],
    }

    out0 = layer.apply(empty)
    out1 = layer.apply(_obs(step=1, player=0))

    assert layer.owner_map == (1, 2, 3, 0)
    assert out0["player"] == 0
    assert out1["player"] == 1


def test_infer_num_players_keeps_two_player_game_two_player():
    obs = {
        "step": 0,
        "player": 0,
        "planets": [
            [10, 0, 10.0, 10.0, 3.0, 50, 1],
            [11, 1, 90.0, 90.0, 3.0, 50, 1],
            [12, -1, 50.0, 50.0, 3.0, 10, 0],
        ],
        "fleets": [[100, 1, 80.0, 80.0, 3.14, 11, 7]],
    }
    assert infer_num_players_from_obs(obs) == 2
    layer = ClockwiseSeatSwap()
    swapped = layer.apply(obs)
    assert layer.owner_map == (1, 0)
    assert swapped["player"] == 1
    assert [p[1] for p in swapped["planets"]] == [1, 0, -1]
    assert swapped["fleets"][0][1] == 0
    assert _xy(swapped["planets"][0]) == (90.0, 90.0)


def test_infer_num_players_uses_initial_planets():
    obs = {
        "step": 0,
        "player": 0,
        "initial_planets": [
            [10, 0, 10.0, 10.0, 3.0, 50, 1],
            [11, 1, 90.0, 90.0, 3.0, 50, 1],
        ],
        "planets": [],
        "fleets": [],
    }

    assert infer_num_players_from_obs(obs) == 2


def _xy(row):
    if len(row) == 2:
        return (round(float(row[0]), 6), round(float(row[1]), 6))
    return (round(float(row[2]), 6), round(float(row[3]), 6))
