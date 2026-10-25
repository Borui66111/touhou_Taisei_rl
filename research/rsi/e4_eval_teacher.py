#!/usr/bin/env python3
"""E4 unchanged risk-aware teacher baseline on NEW native development seeds."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"my_test"))
from native_rl.env import EnvConfig
from native_rl.torch_ppo import evaluate
from risk_aware_teacher import risk_aware_action
SEEDS=list(range(42132,42148))

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--library",required=True)
    p.add_argument("--resource-path",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args(argv)
    if not Path(a.library).is_file() or not Path(a.resource_path).is_dir():
        p.error("Authentic pinned native Taisei required")
    target=Path(a.out)
    if target.exists():p.error("Refusing overwrite")
    cfg=EnvConfig(library_path=a.library,resource_path=a.resource_path,
                  stage_id=1,difficulty=1,practice=False,max_frames=10000,
                  frames_per_action=1)
    rows=evaluate(cfg,checkpoint_path=None,episodes=len(SEEDS),seed=SEEDS[0],
                  sampled=False,planner=risk_aware_action)
    if [int(r["seed"]) for r in rows]!=SEEDS or any(int(r["bombs_used"])!=0 for r in rows):
        raise ValueError("Unexpected seed order or native resources used")
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",encoding="utf-8") as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True)+"\n")
    print(json.dumps({"policy":"risk_teacher_v1","n":len(rows),
          "mean_alive_frames":sum(x["alive_frames"] for x in rows)/len(rows),
          "clears":sum(int(x["cleared"]) for x in rows)}))

if __name__=="__main__":main()
