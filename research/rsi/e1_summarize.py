#!/usr/bin/env python3
"""Validate and summarize paired real-native E1 movement-only evaluations."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import sys

EXPECTED_SEEDS=list(range(42100,42116))
def read(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    seeds=[x["seed"] for x in rows]
    if seeds!=EXPECTED_SEEDS:
        raise ValueError(f"{path}: expected same 16 ordered development seeds; received {seeds}")
    if any(x["bombs_used"]!=0 or x.get("bomb_attempts",0)!=0 for x in rows):
        raise ValueError(f"{path}: no-skill baseline unexpectedly used Bombs")
    for x in rows:
        if x["alive_frames"] is None or x["frames"]<x["alive_frames"]:
            raise ValueError(f"{path}: invalid alive-frame record")
    return rows

def ci(values, repetitions=2000):
    rng=random.Random(20261025)
    n=len(values)
    draws=sorted(mean(rng.choice(values) for _ in range(n)) for _ in range(repetitions))
    return [draws[int(.025*repetitions)],draws[int(.975*repetitions)]]

def describe(rows):
    return dict(n=len(rows),clears=sum(bool(x["cleared"]) for x in rows),
        mean_alive_frames=mean(x["alive_frames"] for x in rows),
        mean_deaths=mean(x["deaths"] for x in rows),
        mean_boss_damage=mean(x["boss_damage"] for x in rows),
        mean_graze=mean(x["graze"] for x in rows),
        mean_bombs=mean(x["bombs_used"] for x in rows))

def main():
    if len(sys.argv)!=2:
        raise SystemExit("usage: e1_summarize.py OUTPUT_DIRECTORY")
    out=Path(sys.argv[1])
    paths={name:out/(name+"_evaluation.jsonl") for name in ("heuristic","baseline","survival_first")}
    rows={name:read(path) for name,path in paths.items()}
    arms={}
    for arm in ("baseline","survival_first"):
        ckpt=out/(arm+"_seed48020.pt")
        manifest=json.loads((out/(arm+"_seed48020_manifest.json")).read_text())
        sha=hashlib.sha256(ckpt.read_bytes()).hexdigest()
        if manifest.get("checkpoint_sha256")!=sha or manifest.get("status")!="training_completed_unpromoted":
            raise ValueError(f"{arm}: training checkpoint manifest invalid")
        if manifest.get("policy_type")!="legacy_18_movement_no_skills" or manifest.get("difficulty")!=1:
            raise ValueError(f"{arm}: invalid training task or action space")
        arms[arm]={"checkpoint_sha256":sha,"reward":manifest["reward"],
                   "training_steps":manifest["interaction_steps"]}
    original=rows["baseline"];candidate=rows["survival_first"]
    paired=[{"seed":a["seed"],"baseline_alive_frames":a["alive_frames"],
              "survival_first_alive_frames":b["alive_frames"],
              "difference_survival_first_minus_baseline":b["alive_frames"]-a["alive_frames"],
              "baseline_deaths":a["deaths"],"survival_first_deaths":b["deaths"]}
              for a,b in zip(original,candidate)]
    alive_diffs=[x["difference_survival_first_minus_baseline"] for x in paired]
    result={
        "status":"diagnostic_native_movement_reward_ablation_no_promotion",
        "task":"genuine Taisei Stage 1 Easy, no Bomb/Surge actions",
        "development_seeds":EXPECTED_SEEDS,
        "sealed_final_seeds":"43000-43099 unused",
        "training_seed":48020,"arms":arms,
        "results":{name:describe(rs) for name,rs in rows.items()},
        "paired_alive_improvement":{
            "mean":mean(alive_diffs),"bootstrap_95pct_ci_eval_seed_only":ci(alive_diffs)},
        "per_seed":paired,
        "limitations":"16 development game seeds, one training seed, 20k interactions/arm; changed reward vector as one bundle; confidence interval excludes between-training-seed variability. No automatic promotion."
    }
    path=out/"e1_summary.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    main()
