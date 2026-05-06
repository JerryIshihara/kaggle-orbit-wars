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
    def test_static_collision_checks_planet_before_sun(self) -> None:
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
        self.assertEqual(hit["kind"], "planet")
        self.assertEqual(hit["planet"][pu.P_ID], target[pu.P_ID])

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


if __name__ == "__main__":
    unittest.main()
