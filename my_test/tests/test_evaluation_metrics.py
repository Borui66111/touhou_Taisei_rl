import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_rl.env import EnvConfig
from native_rl.torch_ppo import evaluate


class FakeEvaluationEnv:
    """Deterministic info stream for evaluation metric contract tests."""
    def __init__(self, config):
        self.episodes = [
            [
                (0.5, False, False, {'boss_damage': 3.0, 'won': False, 'frames': 4, 'requested_skill': 0}),
                (1.0, True, False, {'boss_damage': 2.0, 'won': True, 'frames': 9, 'requested_skill': 0}),
            ],
            [
                (-0.25, False, True, {'boss_damage': -7.0, 'won': True, 'frames': 12, 'requested_skill': 0}),
            ],
        ]
        self.episode_index = -1
        self.step_index = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def reset(self, seed):
        self.episode_index += 1
        self.step_index = 0
        return self._observation(0), {}

    def _observation(self, graze, alive=False):
        return {
            'player': [0.0], 'boss': [0.0], 'bullets': [], 'lasers': [],
            'projectile_count': 0, 'laser_count': 0,
            'graze': graze, 'deaths': 0,
            'score': 1000 + graze, 'point_item_value': 100 + graze,
            'bombs_used': 2, 'logical_frame': 50 + graze,
            'skill_state': [float(alive), 0.0, 0.0],
        }

    def step(self, action):
        reward, terminated, truncated, info = self.episodes[self.episode_index][self.step_index]
        self.step_index += 1
        alive = self.episode_index == 0 and self.step_index == 1
        return self._observation(self.step_index, alive=alive), reward, terminated, truncated, {
            **info, 'digest': 'fake-digest',
        }


class TestEvaluationMetrics(unittest.TestCase):
    def test_alive_frame_mean_uses_per_episode_alive_counts_and_preserves_unknown(self):
        from native_rl.torch_ppo import mean_alive_frames

        rows = [{'alive_frames': 1, 'frames': 9}, {'alive_frames': 0, 'frames': 12}]
        self.assertEqual(mean_alive_frames(rows), 0.5)
        self.assertIsNone(mean_alive_frames([{'alive_frames': None, 'frames': 9}]))

    def test_alive_frame_metrics_are_unknown_for_multi_frame_action_stride(self):
        with patch('native_rl.torch_ppo.TaiseiEnv', FakeEvaluationEnv):
            rows = evaluate(EnvConfig(library_path='not-used-in-fake', frames_per_action=3),
                            None, episodes=1, planner=lambda observation: 0)

        self.assertIsNone(rows[0]['alive_frames'])

    def test_episode_boss_damage_and_alive_frames_use_native_state_boundaries(self):
        with patch('native_rl.torch_ppo.TaiseiEnv', FakeEvaluationEnv):
            rows = evaluate(EnvConfig(library_path='not-used-in-fake'), None,
                            episodes=2, planner=lambda observation: 0)

        self.assertEqual(rows[0]['boss_damage'], 5.0)
        self.assertTrue(rows[0]['cleared'])
        self.assertEqual(rows[1]['boss_damage'], 0.0)
        self.assertFalse(rows[1]['cleared'])
        self.assertFalse(rows[1]['terminated'])
        self.assertTrue(rows[1]['truncated'])
        self.assertEqual(rows[0]['frames'], 9)
        self.assertEqual(rows[1]['frames'], 12)
        self.assertEqual(rows[0]['alive_frames'], 1)
        self.assertEqual(rows[1]['alive_frames'], 0)
        self.assertNotEqual(rows[0]['alive_frames'], rows[0]['frames'])
        self.assertEqual(rows[0]['score'],1002)
        self.assertEqual(rows[0]['point_item_value'],102)
        self.assertEqual(rows[0]['bombs_used'],2)
        self.assertEqual(rows[0]['logical_frame'],52)
        self.assertEqual(rows[0]['skill_events'], [])
        self.assertEqual(rows[0]['bomb_attempts'], 0)
        self.assertEqual(rows[0]['bomb_uses_confirmed'], 0)
        self.assertEqual(rows[0]['special_attempts'], 0)
        self.assertEqual(rows[0]['special_activations'], 0)
        self.assertEqual(rows[0]['surge_active_frames'], 0)

    def test_skill_events_preserve_attempt_vs_confirmation_and_surge_transitions(self):
        class SkillEventEnv(FakeEvaluationEnv):
            def __init__(self, config):
                super().__init__(config)
                self.episodes = [[(0.0, True, False, {
                    'boss_damage': 0.0, 'won': True, 'frames': 1,
                    'requested_skill': 2, 'logical_frame_before': 50,
                    'logical_frame_after': 51, 'player_alive_before': True,
                    'player_alive_after': True, 'bombs_before': 2, 'bombs_after': 2,
                    'bombs_used_before': 0, 'bombs_used_after': 0,
                    'bombs_used_delta': 0, 'bomb_active_before': 0,
                    'bomb_active_after': 0, 'power_surge_active_before': 0,
                    'power_surge_active_after': 1,
                    'power_surge_positive_before': 0.0,
                    'power_surge_positive_after': 99.75,
                    'power_surge_negative_before': 0.0,
                    'power_surge_negative_after': 0.0,
                })]]

        with patch('native_rl.torch_ppo.TaiseiEnv', SkillEventEnv):
            rows = evaluate(EnvConfig(library_path='not-used-in-fake'), None,
                            episodes=1, planner=lambda observation: (0, 2))

        row = rows[0]
        self.assertEqual(len(row['skill_events']), 1)
        event = row['skill_events'][0]
        self.assertEqual(event['logical_frame'], 51)
        self.assertEqual(event['requested_skill'], 2)
        self.assertTrue(event['player_alive_after'])
        self.assertFalse(event['bomb_use_confirmed'])
        self.assertEqual(event['power_surge_active_transition'], [0, 1])
        self.assertEqual(event['power_surge_positive_transition'], [0.0, 99.75])
        self.assertEqual(row['bomb_attempts'], 0)
        self.assertEqual(row['special_attempts'], 1)
        self.assertEqual(row['special_activations'], 1)
        self.assertEqual(row['surge_active_frames'], 1)

    def test_bomb_attempt_without_native_use_is_not_counted_as_use(self):
        class BombAttemptEnv(FakeEvaluationEnv):
            def __init__(self, config):
                super().__init__(config)
                self.episodes = [[(0.0, True, False, {
                    'boss_damage': 0.0, 'won': True, 'frames': 1,
                    'requested_skill': 1, 'logical_frame_before': 50,
                    'logical_frame_after': 51, 'player_alive_before': True,
                    'player_alive_after': True, 'bombs_before': 0, 'bombs_after': 0,
                    'bombs_used_before': 0, 'bombs_used_after': 0,
                    'bombs_used_delta': 0, 'bomb_active_before': 0,
                    'bomb_active_after': 0, 'power_surge_active_before': 0,
                    'power_surge_active_after': 0,
                    'power_surge_positive_before': 0.0,
                    'power_surge_positive_after': 0.0,
                    'power_surge_negative_before': 0.0,
                    'power_surge_negative_after': 0.0,
                })]]

        with patch('native_rl.torch_ppo.TaiseiEnv', BombAttemptEnv):
            row = evaluate(EnvConfig(library_path='not-used-in-fake'), None,
                           episodes=1, planner=lambda observation: (0, 1))[0]
        self.assertEqual(row['bomb_attempts'], 1)
        self.assertEqual(row['bomb_uses_confirmed'], 0)
        self.assertFalse(row['skill_events'][0]['bomb_use_confirmed'])


if __name__ == '__main__':
    unittest.main()
