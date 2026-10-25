#!/usr/bin/env python3
"""Native Taisei E1: movement-only PPO reward formulation ablation.

Both arms use the SAME 18-action legacy movement policy: no Bomb or Surge action
exists. Only reward coefficients differ. This is a small research pilot, not
an automatic champion promotion. No toy physics or sealed final seeds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BASELINE = dict(survival_weight=1.0, damage_weight=0.02,
    graze_weight=0.001, death_penalty=-10.0, loss_penalty=-10.0,
    win_bonus=20.0)
CHALLENGER = dict(survival_weight=0.002, damage_weight=0.001,
    graze_weight=0.0, death_penalty=-30.0, loss_penalty=-30.0,
    win_bonus=100.0)

def reward(arm):
    if arm == "baseline":
        return BASELINE.copy()
    if arm == "survival_first":
        return CHALLENGER.copy()
    raise ValueError(f"Unexpected arm {arm}")

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--arm", choices=["baseline","survival_first"], required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--resource-path", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=48020)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--preflight-only", action="store_true")
    a=p.parse_args(argv)
    if a.seed in range(43000,43100) or a.steps < 512 or a.steps > 20000:
        p.error("Sealed seed forbidden; bounded training must be between 512 and 20000 steps")
    if a.stage != 1 or a.difficulty != 1:
        p.error("E1 preregistered task is genuine Stage 1 Easy, not another difficulty")
    out=Path(a.outdir)
    manifest=dict(
        experiment="E1_movement_only_reward_formulation", arm=a.arm,
        status="preflight_only" if a.preflight_only else "training_requested",
        code_sha=os.environ.get("GITHUB_SHA","unknown"),
        native_library=a.library, resource_path=a.resource_path,
        train_seed=a.seed, stage_id=a.stage, difficulty=1, practice=False,
        max_frames=10000, frames_per_action=1, interaction_steps=a.steps,
        policy_type="legacy_18_movement_no_skills",
        learning_rate=0.0003, gamma=0.995, rollout_steps=512,
        reward=reward(a.arm), sealed_final_seeds="43000-43099 unused",
        hypothesis="Survival/clear-oriented reward improves authentic native movement-only survival versus prior dense reward",
        limitations="One training seed pilot; do not infer general policy competence from this result alone")
    print(json.dumps(manifest,indent=2),flush=True)
    if a.preflight_only:
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    prefix=f"{a.arm}_seed{a.seed}"
    checkpoint=out/(prefix+".pt")
    log=out/(prefix+"_train.jsonl")
    manifest_file=out/(prefix+"_manifest.json")
    if any(q.exists() for q in (checkpoint, log, manifest_file)):
        raise FileExistsError(f"Refusing to overwrite existing E1 evidence for {prefix}")
    manifest_file.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"my_test"))
    from native_rl.env import EnvConfig
    from native_rl.torch_ppo import train
    cfg=EnvConfig(library_path=a.library,resource_path=a.resource_path,
        stage_id=1,difficulty=1,practice=False,frames_per_action=1,
        max_frames=10000,**reward(a.arm))
    train(cfg,outfile=str(checkpoint),total_steps=a.steps,rollout_steps=512,
        learning_rate=0.0003,gamma=0.995,seed=a.seed,device="cpu",
        policy_type="legacy",skill_action_mask=False,log_path=str(log))
    manifest["status"]="training_completed_unpromoted"
    manifest["checkpoint_sha256"]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest_file.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("E1 native checkpoint:",manifest["checkpoint_sha256"],flush=True)
    return manifest

if __name__ == "__main__":
    main()
