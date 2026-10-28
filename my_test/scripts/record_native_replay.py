#!/usr/bin/env python3
"""Save the REAL Taisei replay after completing one native game episode."""
import argparse,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from native_rl.env import TaiseiEnv,EnvConfig
from native_rl.planner import heuristic_action
p=argparse.ArgumentParser()
p.add_argument('--library',required=True);p.add_argument('--resource-path')
p.add_argument('--stage',type=int,default=1);p.add_argument('--difficulty',type=int,default=2)
p.add_argument('--practice',action='store_true',help='Enable spell-practice mode (default: off)')
p.add_argument('--seed',type=int,default=2401);p.add_argument('--max-frames',type=int,default=30000)
p.add_argument('--out',default='results/native_agent_replay.tsr')
a=p.parse_args()
cfg=EnvConfig(library_path=a.library,resource_path=a.resource_path,stage_id=a.stage,
    difficulty=a.difficulty,practice=a.practice,max_frames=a.max_frames)
with TaiseiEnv(cfg) as env:
    obs,_=env.reset(seed=a.seed)
    while True:
        act=heuristic_action(obs)
        obs,reward,done,truncated,info=env.step(act)
        if done or truncated:break
    if not done:
        sys.exit('Episode reached max-frames without native termination; replay not saved. Increase --max-frames.')
    path=pathlib.Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    env.native.save_replay(str(path.resolve()))
    print(json.dumps({'replay':str(path.resolve()),'native_terminal':True,
        'won':info['won'],'frames':info['frames'],'graze':obs['graze'],'deaths':obs['deaths']},indent=2))
