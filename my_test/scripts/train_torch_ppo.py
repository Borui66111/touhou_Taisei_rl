#!/usr/bin/env python3
"""Train the ONE live-compatible Taisei policy family: SkillActorCritic mask-v2.

New checkpoints use 546 inputs, 18 movement actions, 3 skill actions and
``taisei_skill_mask_v2_respawn``. Historical 528-feature BC artifacts can be
adapted *for live comparison*, never misrepresented as newly trained skill PPO.
All dynamics still require the authentic pinned native Taisei simulator.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from native_rl.env import EnvConfig
from native_rl.torch_ppo import train, SKILL_ACTION_MASK_VERSION


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True)
    p.add_argument('--resource-path')
    p.add_argument('--stage',type=int,default=1)
    p.add_argument('--difficulty',type=int,default=2)
    p.add_argument('--practice',action='store_true')
    p.add_argument('--frames-per-action',type=int,default=1)
    p.add_argument('--max-frames',type=int,default=10000)
    p.add_argument('--steps',type=int,default=200000)
    p.add_argument('--rollout-steps',type=int,default=512)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--gamma',type=float,default=0.995)
    p.add_argument('--device',default='cpu')
    # Accept the old option only as an explicit assertion of the one supported
    # training contract. Do not permit it to silently create legacy PT files.
    p.add_argument('--policy-type',choices=['skill'],default='skill',
                   help='Only live-compatible skill PPO; legacy 528 training retired')
    p.add_argument('--skill-action-mask',action='store_true',default=True,
                   help='Compatibility flag; canonical mask-v2 is always enabled')
    p.add_argument('--skill-action-mask-version',choices=[SKILL_ACTION_MASK_VERSION],
                   default=SKILL_ACTION_MASK_VERSION,
                   help='Only the current live-control mask ABI is supported')
    resume_group=p.add_mutually_exclusive_group()
    resume_group.add_argument('--init')
    resume_group.add_argument('--resume')
    p.add_argument('--resume-log')
    p.add_argument('--out',default='results/ppo_taisei_native.pt')
    p.add_argument('--log',default='results/ppo_training.jsonl')
    a=p.parse_args(argv)
    if a.steps <= 0 or a.rollout_steps <= 0:
        p.error('--steps and --rollout-steps must be positive')
    if a.skill_action_mask_version != SKILL_ACTION_MASK_VERSION:
        p.error('noncanonical mask version is not permitted')
    cfg=EnvConfig(library_path=a.library,resource_path=a.resource_path,
                  stage_id=a.stage,difficulty=a.difficulty,practice=a.practice,
                  frames_per_action=a.frames_per_action,max_frames=a.max_frames)
    # The underlying train API in this branch accepts skill_action_mask but
    # NOT a skill_action_mask_version kwarg; it records the pinned v2 itself.
    return train(cfg,outfile=a.out,total_steps=a.steps,rollout_steps=a.rollout_steps,
                 learning_rate=a.lr,seed=a.seed,device=a.device,gamma=a.gamma,
                 init=a.init,resume=a.resume,resume_log=a.resume_log,
                 log_path=a.log,policy_type='skill',skill_action_mask=True)


if __name__ == '__main__':
    main()
