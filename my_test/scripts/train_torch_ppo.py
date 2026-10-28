#!/usr/bin/env python3
"""Train PPO ONLY on a loaded native Taisei engine. No simulated substitute."""
import argparse, pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from native_rl.env import EnvConfig
from native_rl.torch_ppo import train
p=argparse.ArgumentParser()
p.add_argument('--library',required=True)
p.add_argument('--resource-path')
p.add_argument('--stage',type=int,default=1)
p.add_argument('--difficulty',type=int,default=2)
p.add_argument('--practice',action='store_true',help='Enable spell-practice mode (default: off)')
p.add_argument('--frames-per-action',type=int,default=1)
p.add_argument('--max-frames',type=int,default=10000)
p.add_argument('--steps',type=int,default=200000)
p.add_argument('--rollout-steps',type=int,default=512)
p.add_argument('--lr',type=float,default=3e-4)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--gamma',type=float,default=0.995)
p.add_argument('--device',default='cpu')
p.add_argument('--policy-type',choices=['legacy','skill'],default='legacy',
               help='legacy 528/18 policy (default) or skill-aware movement+skill policy')
p.add_argument('--skill-action-mask',action='store_true',
               help='Enable the legacy v2 skill action mask')
p.add_argument('--skill-action-mask-version',choices=['taisei_skill_mask_v2_respawn',
               'taisei_skill_mask_v3_surge_discharge'],
               help='Explicit versioned skill action mask for new training')
resume_group=p.add_mutually_exclusive_group()
resume_group.add_argument('--init',help='Initialize model weights only from an earlier checkpoint')
resume_group.add_argument('--resume',help='Resume model, optimizer, and cumulative training state')
p.add_argument('--resume-log',help='Paired JSONL log for legacy checkpoints without counters')
p.add_argument('--out',default='results/ppo_taisei_native.pt')
p.add_argument('--log',default='results/ppo_training.jsonl')
a=p.parse_args()
cfg=EnvConfig(library_path=a.library,resource_path=a.resource_path,stage_id=a.stage,
              difficulty=a.difficulty,practice=a.practice,
              frames_per_action=a.frames_per_action,max_frames=a.max_frames)
train(cfg,outfile=a.out,total_steps=a.steps,rollout_steps=a.rollout_steps,
      learning_rate=a.lr,seed=a.seed,device=a.device,gamma=a.gamma,init=a.init,
      resume=a.resume,resume_log=a.resume_log,log_path=a.log,
      policy_type=a.policy_type,skill_action_mask=a.skill_action_mask,
      skill_action_mask_version=a.skill_action_mask_version)
