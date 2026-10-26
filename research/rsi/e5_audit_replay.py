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
    import numpy as np  # Only archived tensor audit needs NumPy; replay validator is stdlib-only.
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
    """Fail closed unless *every* native decision and terminal state is covered.

    E5 runs deterministic legacy movement inference with frames_per_action=1.
    Comparing two identical but incomplete traces is NOT proof of replay.
    """
    first=episodes(first_summary,seeds);second=episodes(second_summary,seeds)
    trace_a=jsonl(first_trace);trace_b=jsonl(second_trace)
    if len(trace_a)!=len(trace_b):
        raise ValueError("Native replay trace length mismatch")
    counts={s:0 for s in seeds}
    alive_counts={s:0 for s in seeds}
    last_post={}
    episode_cursor=0
    for i,(a,b) in enumerate(zip(trace_a,trace_b)):
        if a!=b:
            raise ValueError(f"Native policy action/state replay mismatch at index {i}")
        if not isinstance(a,dict):
            raise ValueError("Native replay trace row must be an object")
        s=a.get("seed")
        if type(s) is not int or s not in counts:
            raise ValueError("Unexpected native replay seed")
        if episode_cursor>=len(seeds) or s!=seeds[episode_cursor]:
            raise ValueError("Native replay seed order or episode coverage mismatch")
        if type(a.get("episode_index")) is not int or a["episode_index"]!=episode_cursor:
            raise ValueError("Native replay episode index mismatch")
        if type(a.get("action_index")) is not int or a["action_index"]!=counts[s]:
            raise ValueError("Replay seed or action index discontinuity")
        move=a.get("movement_action")
        argmax=a.get("movement_argmax")
        if type(move) is not int or not 0<=move<18:
            raise ValueError("Invalid replay policy action")
        if type(argmax) is not int or move!=argmax:
            raise ValueError("Deterministic E5 movement action differs from argmax")
        if a.get("skill_action") is not None:
            raise ValueError("E5 legacy no-skill replay unexpectedly used skill action")
        pre=a.get("pre");post=a.get("post")
        if not isinstance(pre,dict) or not isinstance(post,dict):
            raise ValueError("Native synchronized trace evidence missing")
        for state in (pre,post):
            if (type(state.get("logical_frame")) is not int
                    or type(state.get("alive")) is not bool
                    or type(state.get("deaths")) is not int
                    or state["deaths"]<0):
                raise ValueError("Missing or invalid native state fields")
        if post["logical_frame"]!=pre["logical_frame"]+1:
            raise ValueError("Native one-frame logical progression mismatch")
        if s in last_post:
            prior=last_post[s]
            if (pre["logical_frame"]!=prior["logical_frame"]
                    or pre["alive"]!=prior["alive"]
                    or pre["deaths"]!=prior["deaths"]):
                raise ValueError("Native pre/post state continuity mismatch")
        if post["deaths"]<pre["deaths"]:
            raise ValueError("Native death counter regressed")
        counts[s]+=1
        alive_counts[s]+=int(post["alive"])
        last_post[s]=post
        if counts[s]>first[episode_cursor]["frames"]:
            raise ValueError("Native replay trace exceeds terminal frame count")
        if counts[s]==first[episode_cursor]["frames"]:
            episode_cursor+=1
    for j,s in enumerate(seeds):
        if (counts[s]!=first[j]["frames"]
                or counts[s]!=second[j]["frames"]):
            raise ValueError("Native replay trace coverage differs from terminal frames")
        if (last_post.get(s,{}).get("logical_frame")!=first[j]["logical_frame"]
                or last_post[s]["logical_frame"]!=second[j]["logical_frame"]):
            raise ValueError("Native replay terminal logical frame mismatch")
        if (alive_counts[s]!=first[j]["alive_frames"]
                or alive_counts[s]!=second[j]["alive_frames"]):
            raise ValueError("Native replay alive-frame count mismatch")
        if first[j]!=second[j]:
            raise ValueError("Native terminal replay mismatch at seed "+str(s))
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
