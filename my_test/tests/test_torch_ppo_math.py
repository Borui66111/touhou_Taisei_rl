"""Numerical unit tests only. These do not validate Taisei gameplay."""
import math,sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from native_rl.env import TaiseiEnv,EnvConfig
from native_rl.torch_ppo import ActorCritic, gae, optimize, features, checkpoint, load_model, SkillActorCritic, skill_features, load_policy


class TestLearningNumerics(unittest.TestCase):
    def test_skill_policy_joint_logprob_and_entropy_are_factorized_sums(self):
        model=SkillActorCritic(4,width=8)
        x=torch.zeros((2,4))
        move_logits,skill_logits,values=model(x)
        move=torch.tensor([2,4]);skill=torch.tensor([1,2])
        logp,entropy=model.joint_logprob_entropy(move_logits,skill_logits,move,skill)
        md=torch.distributions.Categorical(logits=move_logits)
        sd=torch.distributions.Categorical(logits=skill_logits)
        self.assertTrue(torch.allclose(logp,md.log_prob(move)+sd.log_prob(skill)))
        self.assertTrue(torch.allclose(entropy,md.entropy()+sd.entropy()))
        self.assertEqual(values.shape,(2,))

    def test_skill_features_are_extended_and_exclude_telemetry(self):
        obs={'player':[0.0]*8,'boss':[0.0]*6,'bullets':[],'lasers':[],
             'projectile_count':0,'laser_count':0,'timing':[1.0,2.0],
             'skill_state':[3.0,4.0],'score':123,'items':[[1,2,3]],'point_item_value':99}
        legacy=features(obs)
        extended=skill_features(obs)
        self.assertEqual(legacy.shape,(16,))
        self.assertEqual(extended.shape,(20,))
        np.testing.assert_array_equal(extended[-4:],[1,2,3,4])

    def test_checkpoint_policy_type_is_explicit_and_legacy_stays_loadable(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'legacy.pt'
            model=ActorCritic(14,18,32)
            torch.save({'model':model.state_dict(),'input_dim':14,'actions':18,'width':32},path)
            loaded,meta=load_policy(path)
            self.assertEqual(meta['policy_type'],'legacy')
            self.assertIsInstance(loaded,ActorCritic)
            skill=SkillActorCritic(20,width=16)
            opt=torch.optim.Adam(skill.parameters())
            checkpoint(skill,opt,Path(d)/'skill.pt',EnvConfig(library_path='fake'),0,0)
            skill_loaded,skill_meta=load_policy(Path(d)/'skill.pt')
            self.assertEqual(skill_meta['policy_type'],'skill')
            self.assertIsInstance(skill_loaded,SkillActorCritic)
            with self.assertRaises(ValueError):
                load_model(Path(d)/'skill.pt')

    def test_skill_policy_optimize_uses_joint_actions_and_updates_both_heads(self):
        torch.manual_seed(3);np.random.seed(3)
        model=SkillActorCritic(5,width=16)
        states=np.random.randn(12,5).astype('float32')
        with torch.no_grad():
            ml,sl,v=model(torch.from_numpy(states))
            md=torch.distributions.Categorical(logits=ml)
            sd=torch.distributions.Categorical(logits=sl)
            moves=md.sample();skills=sd.sample()
            logp=md.log_prob(moves)+sd.log_prob(skills)
        batch={'states':list(states),'actions':np.stack((moves.numpy(),skills.numpy()),axis=1).tolist(),
               'values':v.numpy().tolist(),'logprobs':logp.numpy().tolist(),
               'next_values':[0.0]*12,'rewards':np.linspace(-1,1,12).tolist(),
               'terminated':[False]*11+[True],'truncated':[False]*12}
        optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
        old_move=model.actor.weight.detach().clone()
        old_skill=model.skill_actor.weight.detach().clone()
        stats=optimize(model,optimizer,batch,epochs=1,batch_size=12)
        self.assertGreater((model.actor.weight.detach()-old_move).abs().sum().item(),0)
        self.assertGreater((model.skill_actor.weight.detach()-old_skill).abs().sum().item(),0)
        self.assertTrue(all(math.isfinite(v) for v in stats.values()))

    def test_gae_bootstrap_time_limit_vs_terminal(self):
        a,_=gae(rewards=[0.0],values=[1.0],next_values=[8.0],
                 terminated=[False],truncated=[True],gamma=0.5,lam=0.95)
        self.assertAlmostEqual(float(a[0]),3.0)
        b,_=gae(rewards=[0.0],values=[1.0],next_values=[8.0],
                 terminated=[True],truncated=[False],gamma=0.5,lam=0.95)
        self.assertAlmostEqual(float(b[0]),-1.0)

    def test_gae_no_leakage_across_reset(self):
        advantage,_=gae([0,0],[0,0],[2,2],
                        terminated=[False,False],truncated=[True,False],
                        gamma=0.5,lam=1.0)
        self.assertAlmostEqual(float(advantage[0]),1.0)
        self.assertAlmostEqual(float(advantage[1]),1.0)

    def test_optimize_updates_network_and_creates_checkpoint(self):
        torch.manual_seed(0);np.random.seed(0)
        model=ActorCritic(14,width=32)
        opt=torch.optim.Adam(model.parameters(),lr=0.002)
        states=np.random.randn(72,14).astype('float32')
        with torch.no_grad():
            logits,values=model(torch.from_numpy(states))
            dist=torch.distributions.Categorical(logits=logits)
            actions=dist.sample();logprobs=dist.log_prob(actions)
        b={'states':list(states),'actions':actions.tolist(),'values':values.numpy().tolist(),
           'logprobs':logprobs.numpy().tolist(),
           'next_values':[0.0]*72,
           'rewards':np.random.randn(72).tolist(),
           'terminated':[False]*71+[True], 'truncated':[False]*72}
        before=model.actor.weight.detach().clone()
        stats=optimize(model,opt,b,epochs=2,batch_size=24)
        self.assertGreater((model.actor.weight.detach()-before).abs().sum().item(),0.0)
        self.assertTrue(all(math.isfinite(v) for v in stats.values()))
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.pt'
            checkpoint(model,opt,path,EnvConfig(library_path='/not/real'),1,72)
            same,meta=load_model(path)
            self.assertEqual(meta['steps'],72)
            self.assertTrue(torch.allclose(same.actor.weight,model.actor.weight))

    def test_features_size_and_masks(self):
        from test_native_interface import FakeNative
        with TaiseiEnv(EnvConfig(library_path='fake',max_projectiles=5,max_laser_samples=3),native=FakeNative()) as env:
            obs,_=env.reset()
            f=features(obs)
            self.assertEqual(len(f),8+6+5*7+3*4+2)
            self.assertEqual(len(skill_features(obs))-len(f),18)
            self.assertEqual(obs['bullets'][0][-1],1)
            self.assertEqual(obs['bullets'][1][-1],0)
        with TaiseiEnv(EnvConfig(library_path='fake'),native=FakeNative()) as env:
            obs,_=env.reset()
            self.assertEqual(features(obs).shape,(528,))
            self.assertEqual(skill_features(obs).shape,(546,))

if __name__=='__main__':unittest.main()
