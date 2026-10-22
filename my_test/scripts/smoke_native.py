#!/usr/bin/env python3
"""Runs an ACTUAL native Taisei episode, refuses any toy fallback."""
import argparse, json, pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from native_rl import EnvConfig, TaiseiEnv

p=argparse.ArgumentParser()
p.add_argument('--library',required=True,help='Path to libtaisei_sim.so')
p.add_argument('--resource-path',default=None)
p.add_argument('--stage',type=int,default=1)
p.add_argument('--difficulty',type=int,default=2)
p.add_argument('--practice',action='store_true',help='Enable spell-practice mode (default: off)')
p.add_argument('--seed',type=int,default=1234)
p.add_argument('--frames',type=int,default=600)
a=p.parse_args()
cfg=EnvConfig(library_path=a.library,resource_path=a.resource_path,
              stage_id=a.stage,difficulty=a.difficulty,practice=a.practice,max_frames=a.frames)
with TaiseiEnv(cfg) as env:
    obs,info=env.reset(seed=a.seed)
    print('RESET:',json.dumps({**info,'player':obs['player'],'digest':obs['gameplay_digest']}))
    for i in range(a.frames):
        obs,reward,terminated,truncated,info=env.step(0)  # stay and auto-shoot: test controls first
        if i%60==0 or terminated or truncated:
            print(json.dumps({'frame':info['frames'],'reward':reward,
                'projectiles':obs['projectile_count'],'lasers':obs['laser_count'],
                'graze':obs['graze'],'deaths':obs['deaths'],'digest':info['digest']}))
        if terminated or truncated: break
