#!/usr/bin/env python3
"""E3 matched native evaluation: teacher BC policy vs BC-initialized PPO policy."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from statistics import mean
from e2_compare_teacher import read,bootstrap,describe,SEEDS

def compare(a,b,manifest):
    bc=read(a);post=read(b)
    delta=[y["alive_frames"]-x["alive_frames"] for x,y in zip(bc,post)]
    m=json.loads(Path(manifest).read_text(encoding="utf-8"))
    if m.get("status")!="bc_and_ppo_completed_unpromoted" or m.get("native_interactions_max")!=20000:
        raise ValueError("E3 training manifest incomplete or beyond prescribed budget")
    folder=Path(manifest).parent
    for key,name in (("bc_checkpoint_sha256","teacher_bc_seed48030.pt"),
                     ("ppo_checkpoint_sha256","teacher_bc_ppo_seed48020.pt")):
        file=folder/name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=m[key]:
            raise ValueError(f"E3 checkpoint SHA-256 validation failed: {name}")
    return {"experiment":"E3_BC_TEACHER_WARMSTART_NATIVE",
            "status":"paired_native_diagnostic_no_promotion",
            "native_training_steps":m["native_interactions_max"],
            "teacher_demo_steps":m["demonstrations"],
            "ppo_training_steps":m["ppo_decisions"],
            "teacher_training_seed":m["demo_seed"],
            "ppo_training_seed":m["ppo_seed"],
            "eval_development_seeds":SEEDS,
            "bc_only":describe(bc),
            "bc_then_ppo":describe(post),
            "paired_alive_delta_after_ppo":mean(delta),
            "paired_eval_seed_bootstrap_95pct_ci":bootstrap(delta),
            "paired_clear_delta_after_ppo":sum(int(y["cleared"])-int(x["cleared"]) for x,y in zip(bc,post))/len(bc),
            "context_reference_only_not_matched_this_run":{
                "E1_movement_original_reward_mean_alive_frames":1036.0,
                "E2_existing_heuristic_mean_alive_frames":1659.875,
                "E2_risk_teacher_mean_alive_frames":7598.1875},
            "model_sha256":{"bc":m["bc_checkpoint_sha256"],"bc_ppo":m["ppo_checkpoint_sha256"]},
            "promote_policy":False,
            "limitations":["Single training seed; 16 development game seeds and no unseen spells.",
                           "Teacher demonstrations share correlated adjacent frames; random-frame BC validation accuracy does not prove gameplay generalization.",
                           "All training interactions capped at 20k; 12k demonstrator decisions plus 8k on-policy PPO.",
                           "No stage-clearing competence is certified without actual native clears and vulnerable-state traces.",
                           "Historical E1/E2 mean-alive comparisons use separate GitHub runs; they are contextual, not within-run paired control."],
            "decision":"Inspect the two learned policies' survival, clears and timeout behavior, then decide whether to extend teacher imitation, make curriculum shorter, or modify observations/rewards."}

def main():
    p=argparse.ArgumentParser()
    for s in ("bc","ppo","manifest","out"):p.add_argument("--"+s,required=True)
    a=p.parse_args()
    dest=Path(a.out)
    if dest.exists():p.error("Refusing overwrite of existing E3 comparison")
    result=compare(a.bc,a.ppo,a.manifest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
