#!/usr/bin/env python3
"""E5 archived native evidence re-audit and independent native-policy replay verifier.

The audit does not execute the game. The replay mode requires output from two
independent REAL Taisei processes and checks both action/state traces and terminal digests.
Neither offline audit nor deterministic replay proves gameplay improvement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import numpy as np

DATA_SHA="02dd11feb60e0dcb5710ae8af3cf33dca7e7e77756d6467f822ad94d0f5a0d8d"
BC_SHA="87b43918d0181355435f105e3fcec44d7fd314c12aec6978b63ec33e9955cf54"
E4_SHA="06c344d1c9415d1277e89c7d7c1a4519f274952f159e1cc5095d7fecf696e6fc"
EVAL_SEEDS=list(range(42132,42148))
REPLAY_SEEDS=[42132,42133]

def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def jsonl(path):
    result=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not result:raise ValueError("Empty native evidence: "+str(path))
    return result

def episodes(path,seeds):
    data=jsonl(path)
    found=[int(row["seed"]) for row in data]
    if found!=seeds or any(43000<=s<=43099 for s in found):
        raise ValueError("Wrong or sealed native evaluation seeds")
    fields={"seed","digest","alive_frames","deaths","bombs_used","boss_damage",
            "cleared","frames","truncated","terminated","score","logical_frame"}
    for row in data:
        if not fields.issubset(row):raise ValueError("Incomplete native episode evidence")
        if row["bombs_used"]!=0:raise ValueError("Bomb use prohibited in E5")
        if row["cleared"] and row["truncated"]:raise ValueError("Timeout mislabeled clear")
        if int(row["frames"])<1:raise ValueError("Invalid frame count")
    return data

def stats(data):
    return {"n":len(data),"clears":sum(bool(r["cleared"]) for r in data),
            "mean_alive_frames":mean(r["alive_frames"] for r in data),
            "mean_deaths":mean(r["deaths"] for r in data),
            "mean_boss_damage":mean(r["boss_damage"] for r in data),
            "mean_bombs_used":mean(r["bombs_used"] for r in data),
            "truncated_count":sum(bool(r["truncated"]) for r in data)}

def max_run(seq):
    best=cur=0
    prev=None
    for x in seq:
        cur=cur+1 if int(x)==prev else 1
        best=max(best,cur)
        prev=int(x)
    return best

def audit(e3,e4):
    e3=Path(e3);e4=Path(e4)
    for path,expected in [(e3/"teacher_demonstrations_12000.npz",DATA_SHA),
                          (e3/"teacher_bc_seed48030.pt",BC_SHA),
                          (e4/"teacher_bc_seed48030.pt",BC_SHA),
                          (e4/"teacher_bc_rebalanced_seed48030.pt",E4_SHA)]:
        if checksum(path)!=expected:raise ValueError("Artifact hash mismatch: "+str(path))
    manifest=json.loads((e3/"e3_manifest.json").read_text())
    if manifest["teacher_demo_completed_episodes"]!=1 or manifest["demonstrations"]!=12000:
        raise ValueError("Unexpected demonstration count")
    lines=(e3/"e3_training.stdout.log").read_text().splitlines()
    reset=[json.loads(s) for s in lines if '"event": "native_teacher_reset_begin"' in s]
    completed=[json.loads(s) for s in lines if '"event": "native_teacher_reset_complete"' in s]
    if len(reset)!=1 or len(completed)!=1 or reset[0]["step"]!=completed[0]["step"]:
        raise ValueError("Native teacher reset evidence inconsistent")
    boundary=int(reset[0]["step"])
    if not 0<boundary<12000 or reset[0]["next_seed"]!=48031:
        raise ValueError("Unexpected native episode boundary")
    with np.load(e3/"teacher_demonstrations_12000.npz",allow_pickle=False) as z:
        states=z["states"]; actions=z["actions"]
    if states.shape!=(12000,528) or actions.shape!=(12000,) or not np.isfinite(states).all():
        raise ValueError("Invalid native demonstration tensor")
    if not np.issubdtype(actions.dtype,np.integer) or np.any((actions<0)|(actions>=18)):
        raise ValueError("Invalid native teacher actions")
    if np.bincount(actions,minlength=18).tolist()!=manifest["demo_action_frequencies"]:
        raise ValueError("Demonstration action counts differ from native manifest")
    spans=[]
    for ep,(start,end) in enumerate(((0,boundary),(boundary,len(actions)))):
        seq=actions[start:end]
        spans.append({"episode_index":ep,"seed":48030+ep,"start":start,"end_exclusive":end,
                      "decisions":len(seq),"complete":ep==0,
                      "stay_fraction":float(np.mean(seq==0)),
                      "action_counts":np.bincount(seq,minlength=18).tolist(),
                      "changes":int(np.count_nonzero(seq[1:]!=seq[:-1])),
                      "longest_identical_run":max_run(seq)})
    names=[("baseline_bc","teacher_bc_seed48030_native.jsonl"),
           ("frequency_aware_bc","teacher_bc_rebalanced_seed48030_native.jsonl"),
           ("unchanged_e2_teacher","risk_teacher_native.jsonl")]
    archived=json.loads((e4/"e4_native_comparison.json").read_text())
    by_policy={}
    records={}
    for key,filename in names:
        records[key]=episodes(e4/filename,EVAL_SEEDS)
        by_policy[key]=stats(records[key])
        for metric,val in by_policy[key].items():
            if abs(val-archived[key][metric])>1e-7:
                raise ValueError("Native E4 report metric mismatch: "+key+"."+metric)
    old=records["baseline_bc"];new=records["frequency_aware_bc"]
    delta=mean(y["alive_frames"]-x["alive_frames"] for x,y in zip(old,new))
    if abs(delta-archived["paired_candidate_minus_baseline_mean_alive"])>1e-7:
        raise ValueError("Paired native E4 survival delta mismatch")
    det=json.loads((e4/"determinism.log").read_text())
    if (det.get("result"),det.get("identical_frames"),det.get("seed"))!=("PASS",600,42000):
        raise ValueError("Native engine determinism record invalid")
    return {"status":"pass_archived_native_artifact_reaudit",
            "source_e3_run":40219788151,"source_e4_run":40220975924,
            "teacher_dataset_sha256":DATA_SHA,"teacher_episodes":spans,
            "teacher_complete_episodes":1,"teacher_partial_episodes":1,
            "overall_stay_fraction":float(np.mean(actions==0)),
            "independent_episode_holdout_is_statistically_unsupported":True,
            "e4_policies":by_policy,"e4_paired_alive_delta":delta,
            "engine_determinism_frames":600,
            "checkpoint_policy_action_replay_verified":False,
            "limitations":["Only two teacher seed episodes, one incomplete",
                           "Random-frame holdout leaks temporal context",
                           "Recomputing archived metrics does not re-execute native Taisei",
                           "Engine determinism is not learned-checkpoint replay"]}

def validate_replay(first_summary,second_summary,first_trace,second_trace,seeds=REPLAY_SEEDS):
    first=episodes(first_summary,seeds);second=episodes(second_summary,seeds)
    trace_a=jsonl(first_trace);trace_b=jsonl(second_trace)
    if len(trace_a)!=len(trace_b):raise ValueError("Native replay trace length mismatch")
    counts={s:0 for s in seeds}
    for i,(a,b) in enumerate(zip(trace_a,trace_b)):
        if a!=b:raise ValueError(f"Native policy action/state replay mismatch at index {i}")
        s=int(a["seed"])
        if s not in counts or int(a["action_index"])!=counts[s]:
            raise ValueError("Replay seed or action index discontinuity")
        if not isinstance(a.get("movement_action"),int) or not 0<=a["movement_action"]<18:
            raise ValueError("Invalid replay policy action")
        if any(k not in a for k in ("pre","post","movement_argmax")):
            raise ValueError("Native synchronized trace evidence missing")
        counts[s]+=1
    if any(v==0 for v in counts.values()):raise ValueError("Missing native replay seed")
    for a,b in zip(first,second):
        if a!=b:raise ValueError("Native terminal replay mismatch at seed "+str(a["seed"]))
    return {"status":"pass_native_policy_replay_two_independent_processes",
            "seeds":seeds,"identical_actions_and_native_states":sum(counts.values()),
            "frames_per_seed":counts,"identical_terminal_digest":True,
            "clears":sum(bool(x["cleared"]) for x in first),
            "mean_alive_frames":mean(x["alive_frames"] for x in first),
            "not_a_policy_gain_claim":True}

def main(argv=None):
    p=argparse.ArgumentParser()
    subs=p.add_subparsers(dest="command",required=True)
    a=subs.add_parser("audit");a.add_argument("--e3",required=True);a.add_argument("--e4",required=True)
    b=subs.add_parser("replay")
    for s in ("first-summary","second-summary","first-trace","second-trace"):
        b.add_argument("--"+s,required=True)
    for s in (a,b):s.add_argument("--out",required=True)
    args=p.parse_args(argv)
    result=audit(args.e3,args.e4) if args.command=="audit" else validate_replay(
        args.first_summary,args.second_summary,args.first_trace,args.second_trace)
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
