#!/usr/bin/env python3
"""Independent episode metrics from the actual native engine only."""
import argparse,json,pathlib,random,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from native_rl.env import EnvConfig
from native_rl.planner import heuristic_action
from native_rl.torch_ppo import evaluate,mean_alive_frames
p=argparse.ArgumentParser()
p.add_argument('--library',required=True);p.add_argument('--resource-path')
p.add_argument('--stage',type=int,default=1);p.add_argument('--difficulty',type=int,default=2)
p.add_argument('--practice',action='store_true',help='Enable spell-practice mode (default: off)')
p.add_argument('--max-frames',type=int,default=10000)
p.add_argument('--model');p.add_argument('--sampled',action='store_true')
p.add_argument('--policy',choices=['ppo','heuristic','random'],default='ppo')
p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,default=91000)
p.add_argument('--out',default='results/native_evaluation.jsonl')
a=p.parse_args()
if a.policy=='ppo' and not a.model:p.error('--model required for PPO')
cfg=EnvConfig(library_path=a.library,resource_path=a.resource_path,stage_id=a.stage,
    difficulty=a.difficulty,practice=a.practice,max_frames=a.max_frames)
if a.policy=='random':
    rng=random.Random(a.seed)
    plan=lambda _:rng.randrange(18)
elif a.policy=='heuristic':plan=heuristic_action
else:plan=None
rows=evaluate(cfg,a.model,episodes=a.episodes,seed=a.seed,sampled=a.sampled,planner=plan)
pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True)
with open(a.out,'w',encoding='utf-8') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
summary={'policy':a.policy,'episodes':len(rows),'win_rate':sum(int(x['won']) for x in rows)/len(rows),
         'clear_rate':sum(int(x['cleared']) for x in rows)/len(rows),
         'truncation_rate':sum(int(x['truncated']) for x in rows)/len(rows),
         'mean_boss_damage':sum(x['boss_damage'] for x in rows)/len(rows),
         'mean_reward':sum(x['reward'] for x in rows)/len(rows),
         'mean_graze':sum(x['graze'] for x in rows)/len(rows),
         'mean_deaths':sum(x['deaths'] for x in rows)/len(rows),
         'mean_frames':sum(x['frames'] for x in rows)/len(rows),
         'mean_survival_frames':mean_alive_frames(rows),
         'mean_score':sum(x['score'] for x in rows)/len(rows),
         'mean_point_item_value':sum(x['point_item_value'] for x in rows)/len(rows),
         'mean_bombs_used':sum(x['bombs_used'] for x in rows)/len(rows),
         'mean_logical_frame':sum(x['logical_frame'] for x in rows)/len(rows)}
print(json.dumps(summary,indent=2))
