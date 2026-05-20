from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


def _load_physics_utils():
    path = Path(__file__).resolve().parents[1] / "agents" / "physics_utils.py"
    spec = importlib.util.spec_from_file_location("physics_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pu = _load_physics_utils()


class PhysicsUtilsCollisionOrderTest(unittest.TestCase):
    def test_fleet_speed_caps_at_env_ship_speed(self) -> None:
        # The env applies min(speed, shipSpeed). Large launches must not
        # lead-aim as if they move faster than the simulator permits.
        self.assertEqual(pu.fleet_speed(1000), pu.MAX_SPEED)
        self.assertEqual(pu.fleet_speed(10_000), pu.MAX_SPEED)

    def test_static_collision_checks_sun_before_planet(self) -> None:
        source = [0, 0, 50.0, 39.0, 1.0, 100, 0]
        target = [1, -1, 50.0, 44.0, 1.0, 0, 0]
        hit = pu.find_first_collision(
            source[pu.P_X],
            source[pu.P_Y],
            source[pu.P_RADIUS],
            source[pu.P_ID],
            math.pi / 2,
            1000,
            [source, target],
            max_distance=20.0,
        )

        self.assertIsNotNone(hit)
        self.assertEqual(hit["kind"], "sun")

    def test_static_collision_checks_boundary_before_planet(self) -> None:
        source = [0, 0, 96.0, 50.0, 1.0, 100, 0]
        target = [1, -1, 99.0, 50.0, 3.0, 0, 0]
        hit = pu.find_first_collision(
            source[pu.P_X],
            source[pu.P_Y],
            source[pu.P_RADIUS],
            source[pu.P_ID],
            0.0,
            1000,
            [source, target],
            max_distance=20.0,
        )

        self.assertIsNotNone(hit)
        self.assertEqual(hit["kind"], "boundary")

    def test_dynamic_collision_checks_sun_before_moving_planet_sweep(self) -> None:
        source = [0, 0, 50.0, 36.0, 1.0, 100, 0]
        moving_target = [1, -1, 62.0, 50.0, 4.0, 0, 0]

        hit = pu._find_first_collision_dynamic(
            source[pu.P_X],
            source[pu.P_Y],
            source[pu.P_RADIUS],
            source[pu.P_ID],
            math.pi / 2,
            1000,
            [source, moving_target],
            angular_velocity=math.pi / 2,
            av_signed=-math.pi / 2,
            comet_lookup={},
            current_step=1,
            max_distance=20.0,
        )

        self.assertIsNotNone(hit)
        self.assertEqual(hit["kind"], "sun")

    def test_validate_launch_rejects_endpoint_boundary_before_planet_hit(self) -> None:
        source = [0, 0, 96.0, 50.0, 1.0, 100, 0]
        target = [1, -1, 99.0, 50.0, 3.0, 0, 0]

        ok, reason = pu.validate_launch(
            source,
            0.0,
            1000,
            target[pu.P_ID],
            [source, target],
            player=0,
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "boundary")

    def test_validate_launch_rejects_wrong_first_hit_even_if_friendly(self) -> None:
        source = [0, 0, 10.0, 80.0, 1.0, 100, 0]
        blocker = [1, 0, 50.0, 80.0, 3.0, 10, 0]
        target = [2, -1, 90.0, 80.0, 3.0, 1, 0]

        ok, reason = pu.validate_launch(
            source,
            0.0,
            20,
            target[pu.P_ID],
            [source, blocker, target],
            player=0,
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "wrong_planet_1")

    def test_plan_launch_rejects_wrong_first_hit_even_if_friendly(self) -> None:
        source = [0, 0, 10.0, 80.0, 1.0, 100, 0]
        blocker = [1, 0, 50.0, 80.0, 3.0, 10, 0]
        target = [2, -1, 90.0, 80.0, 3.0, 1, 0]

        launch = pu.plan_launch(
            source,
            target,
            planets=[source, blocker, target],
            fleets=[],
            player=0,
            angular_velocity=0.0,
            fleet_ships=20,
        )

        self.assertFalse(launch.ok)
        self.assertEqual(launch.reason, "wrong_planet_1")
        self.assertEqual(launch.actual_hit_id, blocker[pu.P_ID])


if __name__ == "__main__":
    unittest.main()
