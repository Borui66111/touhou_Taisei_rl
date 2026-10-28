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
            'player': [-0.25, 0.5], 'boss': [0.0], 'bullets': [], 'lasers': [],
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


    def test_trace_callback_records_pre_post_state_and_distribution_statistics(self):
        import torch
        from torch.distributions import Categorical
        from native_rl.torch_ppo import ActorCritic

        model = ActorCritic(input_dim=5, actions=18, width=4)
        expected_logits = torch.linspace(-1.0, 1.0, 18).unsqueeze(0)
        with torch.no_grad():
            model.actor.weight.zero_()
            model.actor.bias.copy_(expected_logits[0])
            model.base[0].weight.zero_()
            model.base[0].bias.zero_()
            model.base[2].weight.zero_()
            model.base[2].bias.zero_()
        traces = []
        with patch('native_rl.torch_ppo.TaiseiEnv', FakeEvaluationEnv), \
             patch('native_rl.torch_ppo.load_policy', return_value=(model, {})):
            reports = evaluate(EnvConfig(library_path='not-used-in-fake'), 'fake.pt',
                               episodes=1, seed=321, trace_callback=traces.append)

        self.assertEqual(len(traces), 2)
        row = traces[0]
        dist = Categorical(logits=expected_logits)
        self.assertEqual(row['seed'], 321)
        self.assertEqual(row['episode_index'], 0)
        self.assertEqual(row['action_index'], 0)
        self.assertEqual(row['pre']['logical_frame'], 50)
        self.assertEqual(row['pre']['player_x'], -0.25)
        self.assertEqual(row['pre']['player_y'], 0.5)
        self.assertEqual(row['pre']['alive'], False)
        self.assertEqual(row['pre']['deaths'], 0)
        self.assertEqual(row['pre']['projectile_count'], 0)
        self.assertEqual(row['pre']['laser_count'], 0)
        self.assertEqual(row['movement_action'], int(expected_logits.argmax(dim=1).item()))
        self.assertIsNone(row['skill_action'])
        self.assertAlmostEqual(row['movement_entropy'], float(dist.entropy().item()))
        self.assertAlmostEqual(row['movement_max_probability'], float(dist.probs.max().item()))
        self.assertEqual(row['post']['logical_frame'], 51)
        self.assertEqual(row['post']['alive'], True)
        self.assertEqual(row['post']['deaths'], 0)
        self.assertEqual(row['post']['boss_damage'], 3.0)
        self.assertEqual(reports[0]['boss_damage'], 5.0)

    def test_trace_callback_preserves_actions_reports_and_rng(self):
        import hashlib
        import json
        import random
        import numpy as np
        import torch
        from native_rl.torch_ppo import ActorCritic

        model = ActorCritic(input_dim=5, actions=18, width=4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        def run(trace):
            np.random.seed(417)
            random.seed(417)
            torch.manual_seed(417)
            actions = []
            class RecordingEnv(FakeEvaluationEnv):
                def step(self, action):
                    actions.append(action)
                    return super().step(action)
            with patch('native_rl.torch_ppo.TaiseiEnv', RecordingEnv), \
                 patch('native_rl.torch_ppo.load_policy', return_value=(model, {})):
                reports = evaluate(EnvConfig(library_path='not-used-in-fake'), 'fake.pt',
                                   episodes=2, seed=417, sampled=True,
                                   trace_callback=(lambda row: None) if trace else None)
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            numpy_rng_state = (
                numpy_rng_state[0], numpy_rng_state[1].tolist(), *numpy_rng_state[2:]
            )
            torch_rng_state = torch.get_rng_state().tolist()
            return actions, reports, (python_rng_state, numpy_rng_state, torch_rng_state)
        without_trace = run(False)
        with_trace = run(True)
        self.assertEqual(with_trace[0], without_trace[0])
        self.assertEqual(with_trace[1], without_trace[1])
        self.assertEqual(with_trace[2][0], without_trace[2][0])
        self.assertEqual(with_trace[2][1], without_trace[2][1])
        self.assertEqual(with_trace[2][2], without_trace[2][2])
        def result_hash(result):
            payload = json.dumps(result, sort_keys=True, separators=(',', ':'))
            return hashlib.sha256(payload.encode('utf-8')).hexdigest()
        self.assertEqual(result_hash(with_trace), result_hash(without_trace))

    def test_trace_records_nearest_projectile_approach_geometry(self):
        import torch
        from native_rl.torch_ppo import ActorCritic

        class ProjectileEnv(FakeEvaluationEnv):
            def _observation(self, graze, alive=False):
                obs = super()._observation(graze, alive)
                obs['player'] = [120 / 480, 280 / 560, 2 / 20, 0.0, 0, 0, 0, 0]
                obs['boss'] = [0.0] * 6
                obs['bullets'] = [
                    [30 / 480, 10 / 560, 0.0, 0.0, 6 / 30, 8 / 30, 1.0],
                    [20 / 480, 3 / 560, 0.0, 0.0, 9 / 30, 12 / 30, 1.0],
                    [0.0] * 7,
                ]
                return obs

        model = ActorCritic(input_dim=8 + 6 + 3 * 7 + 2, actions=18, width=4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        trace = []
        with patch('native_rl.torch_ppo.TaiseiEnv', ProjectileEnv), \
             patch('native_rl.torch_ppo.load_policy', return_value=(model, {})):
            evaluate(EnvConfig(library_path='unused', max_projectiles=3,
                               max_laser_samples=0), 'fake.pt', episodes=1,
                     trace_callback=trace.append)
        nearest = trace[0]['pre']['nearest_projectile']
        self.assertAlmostEqual(nearest['current_separation_px'], (20 ** 2 + 3 ** 2) ** 0.5)
        self.assertAlmostEqual(nearest['predicted_closest_separation_px'], 3.0)
        self.assertAlmostEqual(nearest['time_to_closest_frames'], 10.0)
        self.assertAlmostEqual(nearest['collision_width_px'], 9.0)
        self.assertAlmostEqual(nearest['collision_height_px'], 12.0)

        class EmptyProjectileEnv(FakeEvaluationEnv):
            pass

        trace = []
        empty_model = ActorCritic(input_dim=5, actions=18, width=4)
        with torch.no_grad():
            for parameter in empty_model.parameters():
                parameter.zero_()
        with patch('native_rl.torch_ppo.TaiseiEnv', EmptyProjectileEnv):
            with patch('native_rl.torch_ppo.load_policy', return_value=(empty_model, {})):
                evaluate(EnvConfig(library_path='unused'), 'fake.pt', episodes=1,
                         trace_callback=trace.append)
        self.assertIsNone(trace[0]['pre']['nearest_projectile'])

    def test_masked_skill_trace_statistics_use_the_inference_mask(self):
        import torch
        from torch.distributions import Categorical
        from native_rl.torch_ppo import SkillActorCritic, SKILL_ACTION_MASK_VERSION

        model = SkillActorCritic(input_dim=22, actions=18, skills=3, width=4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.skill_actor.bias.copy_(torch.tensor([0.0, 9.0, 8.0]))
        obs_masked = dict(FakeEvaluationEnv(EnvConfig(library_path='x'))._observation(0))
        obs_masked.update({'timing': [0.0, 0.0, 1.0], 'boss': [0.0] * 6})
        obs_masked['skill_state'] = [1.0, 0.0, 1.0, 0.0, 0.0, 0.8,
                                     0.0, 0.0, 0.0]
        class MaskEnv(FakeEvaluationEnv):
            def __init__(self, config):
                super().__init__(config)
                self.episodes = [[(0.0, True, False, {
                    'boss_damage': 0.0, 'won': True, 'frames': 1,
                    'requested_skill': 0,
                })]]
            def reset(self, seed):
                self.episode_index += 1
                self.step_index = 0
                return obs_masked, {}
        trace = []
        with patch('native_rl.torch_ppo.TaiseiEnv', MaskEnv), \
             patch('native_rl.torch_ppo.load_policy', return_value=(
                 model, {'skill_action_mask_version': SKILL_ACTION_MASK_VERSION})):
            evaluate(EnvConfig(library_path='not-used-in-fake'), 'fake.pt', episodes=1,
                     trace_callback=trace.append)
        skill_dist = Categorical(logits=torch.tensor([[0.0, -torch.inf, 8.0]]))
        self.assertEqual(trace[0]['skill_action'], 2)
        self.assertAlmostEqual(trace[0]['skill_entropy'], float(skill_dist.entropy().item()))
        self.assertAlmostEqual(trace[0]['skill_max_probability'], float(skill_dist.probs.max().item()))
    def test_trace_writer_is_exclusive_flushes_and_rejects_summary_collision(self):
        import tempfile
        from scripts.evaluate_native import open_trace_file, write_trace_row

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / 'trace.jsonl'
            summary_path = Path(directory) / 'summary.jsonl'
            with open_trace_file(trace_path, summary_path) as trace:
                write_trace_row(trace, {'action_index': 0})
                self.assertTrue(trace_path.exists())
                self.assertEqual(trace_path.read_text(encoding='utf-8'), '{"action_index": 0}\n')
            with self.assertRaises(FileExistsError):
                open_trace_file(trace_path, summary_path)
            with self.assertRaises(ValueError):
                open_trace_file(summary_path, summary_path)
    def test_sampled_evaluation_sampling_seed_repeats_actions_after_model_load(self):
        import torch
        from native_rl.torch_ppo import ActorCritic

        load_count = 0
        def fake_load_policy(*args):
            nonlocal load_count
            load_count += 1
            model = ActorCritic(input_dim=5, actions=18, width=4)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            torch.rand(2 + load_count)
            return model, {}

        def run():
            actions = []
            class RecordingEnv(FakeEvaluationEnv):
                def step(self, action):
                    actions.append(action)
                    return super().step(action)
            trace = []
            with (patch('native_rl.torch_ppo.TaiseiEnv', RecordingEnv),
                  patch('native_rl.torch_ppo.load_policy', side_effect=fake_load_policy)):
                reports = evaluate(
                    EnvConfig(library_path='not-used-in-fake'), 'fake.pt',
                    episodes=2, seed=49024, sampled=True, sampling_seed=49024,
                    trace_callback=trace.append)
            return actions, reports, trace

        first = run()
        second = run()
        self.assertEqual(load_count, 2)
        self.assertEqual(first, second)
        torch.manual_seed(49024)
        expected_distribution = torch.distributions.Categorical(logits=torch.zeros(18))
        expected_actions = [int(expected_distribution.sample().item()) for _ in range(3)]
        self.assertEqual(first[0], expected_actions)
        self.assertEqual(len(first[0]), 3)

    def test_native_cli_sampling_seed_passes_through_to_evaluate(self):
        import tempfile
        from scripts import evaluate_native

        report = {
            'won': False, 'cleared': False, 'truncated': True,
            'boss_damage': 0.0, 'reward': 0.0, 'graze': 0, 'deaths': 0,
            'frames': 1, 'alive_frames': 0, 'score': 0,
            'point_item_value': 0, 'bombs_used': 0, 'logical_frame': 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.evaluate_native.evaluate', return_value=[report]) as mocked_evaluate:
                with patch('builtins.print'):
                    evaluate_native.main([
                        '--library', 'unused', '--model', 'fake.pt', '--sampled',
                        '--sampling-seed', '49024', '--episodes', '1',
                        '--out', str(Path(directory) / 'summary.jsonl'),
                    ])
        self.assertEqual(mocked_evaluate.call_args.kwargs['sampling_seed'], 49024)
        self.assertTrue(mocked_evaluate.call_args.kwargs['sampled'])

    def test_native_cli_rejects_sampling_seed_without_sampled_ppo(self):
        from scripts import evaluate_native

        invalid_combinations = [
            ['--library', 'unused', '--policy', 'heuristic',
             '--sampling-seed', '49024', '--episodes', '1', '--out', 'unused.jsonl'],
            ['--library', 'unused', '--policy', 'ppo', '--model', 'fake.pt',
             '--sampling-seed', '49024', '--episodes', '1', '--out', 'unused.jsonl'],
        ]
        with patch('scripts.evaluate_native.evaluate') as mocked_evaluate:
            for arguments in invalid_combinations:
                with self.subTest(arguments=arguments):
                    with self.assertRaises(SystemExit) as raised:
                        evaluate_native.main(arguments)
                    self.assertEqual(raised.exception.code, 2)
        mocked_evaluate.assert_not_called()

    def test_native_cli_trace_option_streams_rows_without_simulator(self):
        import json
        import tempfile
        from scripts import evaluate_native

        report = {
            'won': False, 'cleared': False, 'truncated': True,
            'boss_damage': 0.0, 'reward': 0.0, 'graze': 0, 'deaths': 0,
            'frames': 1, 'alive_frames': 0, 'score': 0,
            'point_item_value': 0, 'bombs_used': 0, 'logical_frame': 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / 'trace.jsonl'
            summary_path = Path(directory) / 'summary.jsonl'
            def fake_evaluate(*args, trace_callback=None, **kwargs):
                trace_callback({'seed': 7, 'action_index': 0})
                return [report]
            with patch('scripts.evaluate_native.evaluate', side_effect=fake_evaluate):
                evaluate_native.main([
                    '--library', 'unused', '--policy', 'heuristic', '--episodes', '1',
                    '--out', str(summary_path), '--trace-out', str(trace_path),
                ])
            self.assertEqual(json.loads(trace_path.read_text(encoding='utf-8')),
                             {'seed': 7, 'action_index': 0})
            self.assertEqual(len(summary_path.read_text(encoding='utf-8').splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
