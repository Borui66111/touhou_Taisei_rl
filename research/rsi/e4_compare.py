#!/usr/bin/env python3
"""E4 native paired comparison on fresh development seeds, no promotion."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from statistics import mean
import random
SEEDS=list(range(42132,42148))

def load(path):
    rows=[json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if [int(x["seed"]) for x in rows]!=SEEDS:
        raise ValueError("Expected the exact E4 fresh development seeds in order")
    for row in rows:
        for field in ("alive_frames","deaths","bombs_used","boss_damage","cleared"):
            if field not in row:raise ValueError("Incomplete native episode")
        if row["bombs_used"]!=0:
            raise ValueError("Zero-Bomb E4 requirement violated")
    return rows

def descr(rows):
    return {"n":len(rows),"clears":sum(bool(x["cleared"]) for x in rows),
            "mean_alive_frames":mean(x["alive_frames"] for x in rows),
            "mean_deaths":mean(x["deaths"] for x in rows),
            "mean_boss_damage":mean(x["boss_damage"] for x in rows),
            "mean_bombs_used":mean(x["bombs_used"] for x in rows),
            "truncated_count":sum(bool(x.get("truncated",False)) for x in rows)}

def ci(delta):
    gen=random.Random(20261026)
    a=sorted(mean(gen.choice(delta) for _ in delta) for _ in range(4000))
    return [a[100],a[3900]]

def report(baseline,candidate,teacher,offline,models_dir):
    old=load(baseline);new=load(candidate);expert=load(teacher)
    offline_report=json.loads(Path(offline).read_text(encoding="utf-8"))
    if offline_report.get("status")!="offline_learner_created_native_gameplay_not_yet_tested":
        raise ValueError("No verified E4 offline learner")
    manifest_sha=offline_report["candidate_sha256"]
    candidate_model=Path(models_dir)/"teacher_bc_rebalanced_seed48030.pt"
    baseline_model=Path(models_dir)/"teacher_bc_seed48030.pt"
    hashfile=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
    if hashfile(candidate_model)!=manifest_sha or hashfile(baseline_model)!=offline_report["baseline_bc_sha256"]:
        raise ValueError("Learned model identity mismatch")
    delta=[float(y["alive_frames"])-float(x["alive_frames"]) for x,y in zip(old,new)]
    gap=[float(z["alive_frames"])-float(y["alive_frames"]) for y,z in zip(new,expert)]
    return {"experiment":"E4_AUTHENTIC_TAISEI_FREQUENCY_AWARE_BC",
            "source_run":"E3 40219788151, source data 12k native decisions",
            "native_training_interactions_this_run":0,
            "new_development_seeds":SEEDS,
            "same_native_task":"Taisei Stage1 Easy, 18 movement-only actions, 10k frame limit",
            "models":{"baseline_sha256":offline_report["baseline_bc_sha256"],
                      "candidate_sha256":manifest_sha},
            "baseline_bc":descr(old),"frequency_aware_bc":descr(new),
            "unchanged_e2_teacher":descr(expert),
            "paired_candidate_minus_baseline_mean_alive":mean(delta),
            "paired_eval_seed_95pct_bootstrap_ci":ci(delta),
            "candidate_minus_baseline_clear_delta":sum(int(y["cleared"])-int(x["cleared"]) for x,y in zip(old,new))/len(old),
            "paired_teacher_minus_candidate_mean_alive":mean(gap),
            "offline_diagnostics":{k:offline_report[k] for k in ("baseline_holdout","candidate_holdout")},
            "promote_policy":False,
            "limits":["Fresh development seeds but same Easy stage; not sealed final seeds.",
                      "Class-rebalanced BC uses a single original teacher dataset; no cross-stage generalization.",
                      "Truncated episodes are not clears; survival gain alone cannot certify risk-conditioned dodging.",
                      "These are real game evaluations but no new on-policy PPO or extra native training interactions."],
            "decision":"If rebalanced BC makes legitimate paired native gains, investigate coverage-aware short curriculum or constrained PPO to preserve teacher behavior. If not, pivot to better observation architecture / temporally diverse demonstrations rather than rebalancing again."}

def main():
    p=argparse.ArgumentParser()
    for x in ("baseline","candidate","teacher","offline","models-dir","out"):
        p.add_argument("--"+x,required=True)
    a=p.parse_args()
    result=report(a.baseline,a.candidate,a.teacher,a.offline,a.models_dir)
    dest=Path(a.out)
    if dest.exists():p.error("Will not overwrite native comparison evidence")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
