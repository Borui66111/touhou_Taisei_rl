"""Source-only regression tests; passing these is NOT Taisei gameplay evidence."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from risk_aware_teacher import risk_aware_action, DIRECTIONS, _risk_for_bullet


def state(x=.5, y=480/560, bullets=None, lasers=None, boss=False):
    return {
        "player": [x, y, 0, 0, 0, 3, 3, 0],
        "boss": [int(boss), 0, -.4, .5, 0, 0],
        "bullets": bullets if bullets is not None else [],
        "lasers": lasers if lasers is not None else [],
    }


class RiskAwareTeacherTests(unittest.TestCase):
    def test_action_range_and_reproducible(self):
        obs = state()
        a = risk_aware_action(obs)
        self.assertTrue(0 <= a < 18)
        self.assertEqual(a, risk_aware_action(obs))

    def test_no_bullets_near_target_prefers_stay(self):
        self.assertEqual(risk_aware_action(state()), 0)

    def test_left_border_does_not_command_further_left(self):
        a = risk_aware_action(state(x=10/480))
        self.assertNotIn(a % 9, (3, 5, 7))

    def test_right_border_does_not_command_further_right(self):
        a = risk_aware_action(state(x=470/480))
        self.assertNotIn(a % 9, (4, 6, 8))

    def test_padding_bullets_and_lasers_ignored(self):
        plain = state()
        padded = state(bullets=[[0]*7 for _ in range(64)],
                       lasers=[[0]*4 for _ in range(16)])
        self.assertEqual(risk_aware_action(plain), risk_aware_action(padded))

    def test_risk_increases_as_clearance_decreases(self):
        close = [0.025, 0, 0, 0, .1, .1, 1]
        distant = [0.5, 0.5, 0, 0, .1, .1, 1]
        self.assertGreater(_risk_for_bullet(close, 0, 0, 18),
                           _risk_for_bullet(distant, 0, 0, 18))

    def test_horizon_guard(self):
        for bad in (0, -1, 61, 1.2, True):
            with self.assertRaises(ValueError):
                risk_aware_action(state(), bad)

    def test_no_game_dependency(self):
        # A pure-state function must not depend on real native DLL presence
        # for its source-level unit test (real gameplay separately required).
        self.assertTrue(isinstance(risk_aware_action(state()), int))


if __name__ == "__main__":
    unittest.main()
