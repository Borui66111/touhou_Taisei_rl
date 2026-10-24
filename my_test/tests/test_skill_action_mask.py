"""State-derived skill-action mask tests; numerical only, no native engine."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_rl.env import EnvConfig
from native_rl.torch_ppo import (
    SkillActorCritic, checkpoint, load_policy, optimize, skill_action_mask,
    train, apply_skill_action_mask,
)


def observation(*, alive=1.0, bombs=1.0, power=200.0, bomb_active=0.0,
                surge_active=0.0, phase=0.0, respawn_timer=0.0):
    return {'skill_state': [alive, 0, 0, bombs / 8.0, 0, power / 400.0,
                            0, bomb_active, surge_active, 0, 0],
            'boss': [0, 0, 0, 0, phase, 0],
            'timing': [0.0, 0.0, respawn_timer, 0.0, 0.0, 0.0, 0.0]}


class TestSkillActionMask(unittest.TestCase):
    def test_death_bomb_remains_valid_when_respawn_timer_is_zero(self):
        obs = observation(alive=0, bombs=1, bomb_active=0, phase=0,
                          respawn_timer=0)
        self.assertEqual(skill_action_mask(obs), [True, True, False])

    def test_bomb_is_invalid_while_respawn_timer_is_positive(self):
        obs = observation(alive=0, bombs=1, bomb_active=0, phase=0,
                          respawn_timer=1)
        self.assertEqual(skill_action_mask(obs), [True, False, False])

    def test_exact_skill_eligibility_boundaries(self):
        self.assertEqual(skill_action_mask(observation(power=200)), [True, True, True])
        self.assertEqual(skill_action_mask(observation(power=199)), [True, True, False])
        self.assertEqual(skill_action_mask(observation(bomb_active=1)), [True, False, False])
        self.assertEqual(skill_action_mask(observation(surge_active=1)), [True, True, False])
        self.assertEqual(skill_action_mask(observation(bombs=0)), [True, False, True])
        self.assertEqual(skill_action_mask(observation(alive=0)), [True, True, False])
        self.assertEqual(skill_action_mask(observation(phase=5)), [True, False, True])

    def test_masked_logits_suppress_invalid_choices_and_single_valid_is_finite(self):
        logits = torch.tensor([[0.0, 100.0, 50.0], [0.0, 1.0, 2.0]])
        mask = torch.tensor([[True, False, False], [True, True, False]])
        masked = apply_skill_action_mask(logits, mask)
        dist = torch.distributions.Categorical(logits=masked)
        self.assertTrue(torch.equal(dist.probs[0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(dist.probs[1, 2], torch.tensor(0.0)))
        self.assertTrue(torch.isfinite(dist.log_prob(torch.tensor([0, 1]))).all())
        self.assertTrue(torch.isfinite(dist.entropy()).all())
        samples = dist.sample((200,))
        self.assertTrue(torch.all(samples[:, 0] == 0))
        self.assertTrue(torch.all(samples[:, 1] != 2))

    def test_checkpoint_mask_metadata_and_old_checkpoint_default(self):
        with tempfile.TemporaryDirectory() as d:
            model = SkillActorCritic(4, width=8)
            opt = torch.optim.Adam(model.parameters())
            masked_path = Path(d) / 'masked.pt'
            checkpoint(model, opt, masked_path, EnvConfig(library_path='fake'),
                       0, 0, skill_action_mask_version='taisei_skill_mask_v2_respawn')
            _, metadata = load_policy(masked_path)
            self.assertEqual(metadata['skill_action_mask_version'], 'taisei_skill_mask_v2_respawn')
            old_path = Path(d) / 'old.pt'
            torch.save({'model': model.state_dict(), 'input_dim': 4,
                        'actions': 18, 'skills': 3, 'width': 8,
                        'policy_type': 'skill'}, old_path)
            _, old = load_policy(old_path)
            self.assertIsNone(old.get('skill_action_mask_version'))

    def test_optimizer_reuses_rollout_masks(self):
        torch.manual_seed(4)
        model = SkillActorCritic(3, width=8)
        states = np.zeros((2, 3), dtype=np.float32)
        with torch.no_grad():
            moves, skill_logits, values = model(torch.from_numpy(states))
            masks = torch.tensor([[True, False, False], [True, True, False]])
            skills = torch.zeros(2, dtype=torch.long)
            logp, _ = SkillActorCritic.joint_logprob_entropy(
                moves, skill_logits, torch.zeros(2, dtype=torch.long), skills,
                skill_masks=masks)
            move_entropy = torch.distributions.Categorical(logits=moves).entropy().mean()
            masked_skill_logits = apply_skill_action_mask(skill_logits, masks)
            skill_entropy = torch.distributions.Categorical(
                logits=masked_skill_logits).entropy().mean()
        batch = {'states': list(states), 'actions': [[0, 0], [0, 0]],
                 'logprobs': logp.numpy().tolist(), 'values': values.numpy().tolist(),
                 'next_values': [0., 0.], 'rewards': [0., 1.],
                 'terminated': [False, True], 'truncated': [False, False],
                 'skill_masks': masks.numpy().tolist()}
        stats = optimize(model, torch.optim.Adam(model.parameters(), lr=1e-3),
                         batch, epochs=1, batch_size=2)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        self.assertAlmostEqual(stats['movement_entropy'], float(move_entropy), places=6)
        self.assertAlmostEqual(stats['skill_entropy'], float(skill_entropy), places=6)
        self.assertAlmostEqual(stats['entropy'], float(move_entropy + skill_entropy), places=6)
        one_valid_entropy = torch.distributions.Categorical(
            logits=apply_skill_action_mask(skill_logits[:1], masks[:1])
        ).entropy()
        self.assertEqual(float(one_valid_entropy.item()), 0.0)

    def test_evaluation_uses_checkpoint_mask_for_greedy_and_sampled_actions(self):
        from native_rl.torch_ppo import evaluate

        obs = observation(bombs=0, power=0)
        obs.update({'player': [0.0] * 8, 'boss': [0.0] * 6,
                    'bullets': [[0.0] * 7 for _ in range(64)],
                    'lasers': [[0.0] * 4 for _ in range(16)],
                    'projectile_count': 0, 'laser_count': 0, 'timing': [0.0] * 7,
                    'graze': 0, 'deaths': 0, 'score': 0, 'point_item_value': 0,
                    'bombs_used': 0, 'logical_frame': 0})

        class FakeEnv:
            actions = []
            def __init__(self, cfg): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def reset(self, seed=None): return obs, {}
            def step(self, action):
                self.actions.append(action)
                return obs, 0.0, True, False, {
                    'won': False, 'frames': 1, 'requested_skill': action[1],
                    'bombs_used_delta': 0, 'bomb_active_before': 0,
                    'bomb_active_after': 0, 'power_surge_active_before': 0,
                    'power_surge_active_after': 0, 'logical_frame_after': 1,
                    'logical_frame_before': 0, 'boss_damage': 0, 'digest': 0,
                }

        with tempfile.TemporaryDirectory() as d:
            model = SkillActorCritic(546)
            with torch.no_grad():
                model.skill_actor.weight.zero_()
                model.skill_actor.bias.copy_(torch.tensor([0.0, 100.0, 50.0]))
            path = Path(d) / 'masked.pt'
            checkpoint(model, torch.optim.Adam(model.parameters()), path,
                       EnvConfig(library_path='fake'), 0, 0,
                       skill_action_mask_version='taisei_skill_mask_v2_respawn')
            for sampled in (False, True):
                FakeEnv.actions.clear()
                with patch('native_rl.torch_ppo.TaiseiEnv', FakeEnv):
                    evaluate(EnvConfig(library_path='fake'), path, episodes=1,
                             sampled=sampled)
                self.assertEqual(FakeEnv.actions[0][1], 0)
            unmasked_path = Path(d) / 'unmasked.pt'
            checkpoint(model, torch.optim.Adam(model.parameters()), unmasked_path,
                       EnvConfig(library_path='fake'), 0, 0)
            FakeEnv.actions.clear()
            with patch('native_rl.torch_ppo.TaiseiEnv', FakeEnv):
                evaluate(EnvConfig(library_path='fake'), unmasked_path,
                         episodes=1, sampled=False)
            self.assertEqual(FakeEnv.actions[0][1], 1)

    def test_mask_flag_is_rejected_for_legacy_policy(self):
        with self.assertRaisesRegex(ValueError, 'requires.*skill'):
            train(EnvConfig(library_path='fake'), policy_type='legacy',
                  skill_action_mask=True)

    def test_resume_rejects_mask_version_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            model = SkillActorCritic(546)
            optimizer = torch.optim.Adam(model.parameters())
            source = Path(d) / 'source.pt'
            cfg = EnvConfig(library_path='fake.dll')
            checkpoint(model, optimizer, source, cfg, 1, 1,
                       skill_action_mask_version=None)
            with patch('native_rl.torch_ppo.TaiseiEnv') as env:
                with self.assertRaisesRegex(ValueError, 'mask version'):
                    train(cfg, outfile=Path(d)/'new.pt', total_steps=2,
                          policy_type='skill', resume=source, skill_action_mask=True)
                env.assert_not_called()


if __name__ == '__main__':
    unittest.main()
