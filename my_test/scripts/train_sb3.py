#!/usr/bin/env python3
"""Only trains with a real compiled Taisei library. No game substitution."""
import argparse,sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from native_rl.env import EnvConfig
from native_rl.gym_adapter import TaiseiGym
from stable_baselines3 import PPO
p=argparse.ArgumentParser()
p.add_argument('--library',required=True)
p.add_argument('--resource-path')
p.add_argument('--stage',type=int,default=1)
p.add_argument('--difficulty',type=int,default=2)
p.add_argument('--practice',action='store_true',help='Enable spell-practice mode (default: off)')
p.add_argument('--steps',type=int,default=100000)
p.add_argument('--out',default='taisei_ppo_stage1.zip')
a=p.parse_args()
cfg=EnvConfig(library_path=a.library,resource_path=a.resource_path,stage_id=a.stage,
              difficulty=a.difficulty,practice=a.practice)
env=TaiseiGym(cfg)
try:
    agent=PPO('MlpPolicy',env,verbose=1,seed=42,n_steps=512,batch_size=64,device='auto')
    agent.learn(total_timesteps=a.steps)
    agent.save(a.out)
finally:env.close()
