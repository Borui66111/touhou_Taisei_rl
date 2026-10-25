"""Resume checkpoint behavior; uses no native engine."""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_rl.env import EnvConfig
from native_rl.torch_ppo import SkillActorCritic, _resume_checkpoint, checkpoint, train


class FakeEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset_seeds = []
        self.steps = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return observation(), {}

    def step(self, action):
        self.steps += 1
        return observation(), 0.0, False, False, {
            'death_delta': 0, 'graze_delta': 1, 'won': False,
        }


def observation():
    return {'player': [0.0] * 8, 'boss': [0.0] * 6,
            'bullets': [[0.0] * 7 for _ in range(64)],
            'lasers': [[0.0] * 4 for _ in range(16)],
            'projectile_count': 0, 'laser_count': 0,
            'timing': [0.0] * 8, 'skill_state': [0.0] * 10}


class TestResumeTraining(unittest.TestCase):
    def fixture(self, root, *, counters=True):
        cfg = EnvConfig(library_path='fake.dll', resource_path='fake-res',
                        stage_id=1, practice=False, difficulty=2,
                        character=0, shot_mode=0, frames_per_action=1,
                        max_frames=10000)
        model = SkillActorCritic(546, 18, 3, 128)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        # Populate real Adam moments, so restoration checks values, not keys.
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
        source = root / 'source.pt'
        checkpoint(model, opt, source, cfg, updates=9, steps=128)
        payload = torch.load(source, map_location='cpu', weights_only=False)
        payload.pop('training_state', None)
        if counters:
            payload['training_state'] = {'episodes': 7, 'wins': 2, 'deaths': 5,
                'grazes': 31, 'python_rng_state': __import__('random').getstate(),
                'numpy_rng_state': np.random.get_state(),
                'torch_rng_state': torch.get_rng_state()}
        torch.save(payload, source)
        return cfg, source, payload

    def test_resume_restores_model_optimizer_and_cumulative_counters(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, source, original = self.fixture(root)
            log = root / 'old.jsonl'
            newlog, out = root / 'new.jsonl', root / 'new.pt'
            captured = {}
            def inspect_optimizer(model, optimizer, rollout, **kwargs):
                captured['state'] = optimizer.state_dict()
                captured['model'] = {k: v.detach().clone() for k, v in model.state_dict().items()}
                return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
            with patch('native_rl.torch_ppo.TaiseiEnv', FakeEnv):
                with patch('native_rl.torch_ppo.optimize', side_effect=inspect_optimizer):
                    train(cfg, outfile=out, total_steps=129, rollout_steps=1,
                          seed=48000, policy_type='skill', resume=source,
                          log_path=newlog)
            resumed = torch.load(out, map_location='cpu', weights_only=False)
            self.assertEqual(resumed['steps'], 129)
            self.assertEqual(resumed['updates'], 10)
            self.assertEqual(resumed['training_state']['episodes'], 7)
            self.assertEqual(resumed['training_state']['wins'], 2)
            original_model = original['model']
            self.assertEqual(original_model.keys(), captured['model'].keys())
            for name, value in original_model.items():
                self.assertTrue(torch.equal(value, captured['model'][name]), name)
            old_opt = original['optimizer']['state']
            new_opt = captured['state']['state']
            self.assertEqual(old_opt.keys(), new_opt.keys())
            key = next(iter(old_opt))
            self.assertTrue(torch.equal(old_opt[key]['exp_avg'], new_opt[key]['exp_avg']))
            self.assertTrue(torch.equal(old_opt[key]['exp_avg_sq'], new_opt[key]['exp_avg_sq']))
            rows = [json.loads(line) for line in newlog.read_text().splitlines()]
            self.assertEqual(rows[0]['steps'], 129)
            self.assertEqual(rows[0]['updates'], 10)
            self.assertEqual(rows[0]['episodes'], 7)

    def test_legacy_checkpoint_requires_matching_resume_log_counters(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, source, _ = self.fixture(root, counters=False)
            with patch('native_rl.torch_ppo.TaiseiEnv', FakeEnv):
                with self.assertRaisesRegex(ValueError, 'resume-log'):
                    train(cfg, outfile=root/'new.pt', total_steps=129,
                          policy_type='skill', resume=source, log_path=root/'new.jsonl')
                badlog = root/'bad.jsonl'
                badlog.write_text(json.dumps({'steps': 127, 'updates': 9})+'\n')
                with self.assertRaisesRegex(ValueError, 'match'):
                    train(cfg, outfile=root/'new.pt', total_steps=129,
                          policy_type='skill', resume=source,
                          resume_log=badlog, log_path=root/'new.jsonl')
                goodlog = root/'good.jsonl'
                goodlog.write_text(json.dumps({'steps': 128, 'updates': 9,
                    'episodes': 7, 'wins': 2, 'deaths': 5, 'grazes': 31})+'\n')
                train(cfg, outfile=root/'new.pt', total_steps=129,
                      policy_type='skill', resume=source, resume_log=goodlog,
                      log_path=root/'new.jsonl')
            saved = torch.load(root/'new.pt', map_location='cpu', weights_only=False)
            self.assertEqual(saved['training_state']['episodes'], 7)
            self.assertTrue(saved['training_state']['rng_reset'])

    def test_resume_starts_fresh_episode_at_seed_plus_cumulative_episode_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, source, _ = self.fixture(root)
            fake = FakeEnv(cfg)
            with patch('native_rl.torch_ppo.TaiseiEnv', return_value=fake):
                train(cfg, outfile=root/'new.pt', total_steps=129,
                      rollout_steps=1, seed=48000, policy_type='skill',
                      resume=source, log_path=root/'new.jsonl')
            self.assertEqual(fake.reset_seeds, [48007])
            rows = [json.loads(line) for line in (root/'new.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['event'], 'resume_boundary')
            self.assertTrue(rows[0]['fresh_episode'])
            self.assertTrue(rows[-1]['steps'] > 128)

    def test_resume_rejects_existing_outputs_before_environment_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, source, _ = self.fixture(root)
            out, log = root/'existing.pt', root/'existing.jsonl'
            out.write_bytes(b'keep-checkpoint')
            with patch('native_rl.torch_ppo.TaiseiEnv') as env:
                with self.assertRaisesRegex(ValueError, 'exist'):
                    train(cfg, outfile=out, total_steps=129, policy_type='skill',
                          resume=source, log_path=log)
                env.assert_not_called()
            out.unlink()
            log.write_bytes(b'keep-log')
            with patch('native_rl.torch_ppo.TaiseiEnv') as env:
                with self.assertRaisesRegex(ValueError, 'exist'):
                    train(cfg, outfile=out, total_steps=129, policy_type='skill',
                          resume=source, log_path=log)
                env.assert_not_called()

    def test_resume_rejects_aliased_log_and_checkpoint_paths_before_environment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, source, _ = self.fixture(root)
            out = root/'new.pt'
            aliased_log = root/'nested'/'..'/'new.pt'
            with patch('native_rl.torch_ppo.TaiseiEnv', side_effect=FakeEnv) as env:
                with self.assertRaisesRegex(ValueError, 'same|collision|alias'):
                    train(cfg, outfile=out, total_steps=129, rollout_steps=1,
                          policy_type='skill', resume=source, log_path=aliased_log)
                env.assert_not_called()

    def test_init_and_resume_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, source, _ = self.fixture(root)
            with patch('native_rl.torch_ppo.TaiseiEnv') as env:
                with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    train(cfg, outfile=root/'new.pt', total_steps=129,
                          policy_type='skill', init=source, resume=source,
                          log_path=root/'new.jsonl')
                env.assert_not_called()

    def test_resume_validates_config_policy_and_step_limit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, source, _ = self.fixture(root)
            mismatch = EnvConfig(library_path='fake.dll', resource_path='wrong')
            with patch('native_rl.torch_ppo.TaiseiEnv') as env:
                with self.assertRaisesRegex(ValueError, 'config'):
                    train(mismatch, outfile=root/'new.pt', total_steps=129,
                          policy_type='skill', resume=source, log_path=root/'new.jsonl')
                with self.assertRaisesRegex(ValueError, 'total_steps'):
                    train(cfg, outfile=root/'new.pt', total_steps=128,
                          policy_type='skill', resume=source, log_path=root/'new.jsonl')
                with self.assertRaisesRegex(ValueError, 'policy'):
                    train(cfg, outfile=root/'new.pt', total_steps=129,
                          policy_type='legacy', resume=source, log_path=root/'new.jsonl')
                env.assert_not_called()

    def test_resume_accepts_different_machine_local_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, source, original = self.fixture(root)
            worker_cfg = replace(cfg, library_path='/worker/native/libtaisei_sim.so',
                                 resource_path='/worker/taisei/resources',
                                 storage_path='/worker/taisei/storage')

            resumed, _, steps, updates = _resume_checkpoint(
                source, worker_cfg, 'skill', 129, None)

            self.assertEqual((steps, updates), (128, 9))
            self.assertEqual(resumed['config'], original['config'])
            stored = torch.load(source, map_location='cpu', weights_only=False)
            self.assertEqual(stored['config'], original['config'])

    def test_resume_rejects_changed_environment_semantics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, source, _ = self.fixture(root)
            changes = {
                'stage_id': 2,
                'difficulty': 3,
                'character': 1,
                'shot_mode': 1,
                'practice': True,
                'frames_per_action': 2,
                'max_frames': 9999,
            }

            for field, value in changes.items():
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, 'env config mismatch'):
                        _resume_checkpoint(source, replace(cfg, **{field: value}),
                                           'skill', 129, None)

    def test_resume_rejects_different_skill_mask_semantics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, source, _ = self.fixture(root)
            payload = torch.load(source, map_location='cpu', weights_only=False)
            payload['skill_action_mask_version'] = 'taisei_skill_mask_v1'
            torch.save(payload, source)

            with self.assertRaisesRegex(ValueError, 'mask version mismatch'):
                _resume_checkpoint(source, cfg, 'skill', 129, None,
                                   skill_action_mask_version='taisei_skill_mask_v2_respawn')


if __name__ == '__main__':
    unittest.main()
