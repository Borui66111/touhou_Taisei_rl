#!/usr/bin/env python3
"""Native Taisei trace diagnostics for movement, not a dodge-skill certification.

Input must be real-game per-action traces emitted by evaluate_native.py --trace-out.
This script does not simulate physics, assign hypothetical rewards, or promote policies.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean

EXPECTED_SEEDS = list(range(42100, 42104))
ACTIONS = 18
# Native action mapping: stay/up/down/left/right/UL/UR/DL/DR × normal/focus.
LEFT = {3, 5, 7}
RIGHT = {4, 6, 8}
UP = {1, 5, 6}
DOWN = {2, 7, 8}


def read_trace(path):
    events = defaultdict(list)
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            seed = row.get("seed")
            if seed not in EXPECTED_SEEDS:
                raise ValueError(f"{path}:{line_number} unexpected or sealed seed {seed}")
            act = row.get("movement_action")
            if isinstance(act, bool) or not isinstance(act, int) or not 0 <= act < ACTIONS:
                raise ValueError(f"{path}:{line_number} invalid native movement action")
            pre = row.get("pre")
            post = row.get("post")
            if not isinstance(pre, dict) or not isinstance(post, dict):
                raise ValueError(f"{path}:{line_number} missing observation trace")
            if not isinstance(pre.get("player_x"), (float, int)) or not isinstance(pre.get("player_y"), (float, int)):
                raise ValueError(f"{path}:{line_number} missing player coordinates")
            events[seed].append(row)
    if sorted(events) != EXPECTED_SEEDS:
        raise ValueError(f"{path}: expected four complete development seeds; got {sorted(events)}")
    for seed, rows in events.items():
        indices = [row["action_index"] for row in rows]
        if indices != list(range(len(indices))):
            raise ValueError(f"{path}: nonconsecutive action indices for seed {seed}")
    return events


def summarize_one(rows):
    n = len(rows)
    if n == 0:
        raise ValueError("empty trajectory")
    actions = [row["movement_action"] for row in rows]
    changes = sum(a != b for a, b in zip(actions, actions[1:]))
    longest = 1
    run = 1
    for a, b in zip(actions, actions[1:]):
        run = run+1 if a == b else 1
        longest = max(longest, run)
    near = 0
    near_switch = 0
    edge = 0
    border_push = 0
    first_death_frame = None
    old_deaths = int(rows[0]["pre"]["deaths"])
    for i, row in enumerate(rows):
        pre = row["pre"]
        post = row["post"]
        px, py = float(pre["player_x"]), float(pre["player_y"])
        move = actions[i] % 9
        at_edge = px <= .05 or px >= .95 or py <= .05 or py >= .95
        edge += int(at_edge)
        border_push += int(
            (px <= .05 and move in LEFT)
            or (px >= .95 and move in RIGHT)
            or (py <= .05 and move in UP)
            or (py >= .95 and move in DOWN)
        )
        nearest = pre.get("nearest_projectile")
        # This is only a geometric warning proxy; bullet dynamics and real
        # collision dimensions can differ, especially during invulnerability.
        is_near = isinstance(nearest, dict) and nearest.get("predicted_closest_separation_px", 1e99) <= 20
        near += int(is_near)
        near_switch += int(is_near and i > 0 and actions[i] != actions[i-1])
        post_deaths = int(post["deaths"])
        if first_death_frame is None and post_deaths > old_deaths:
            first_death_frame = int(post["logical_frame"])
        old_deaths = post_deaths
    return {
        "recorded_actions": n,
        "movement_action_changes_per_100_frames": round(changes*100/n, 2),
        "longest_identical_action_run": longest,
        "stay_action_fraction": round(sum(a % 9 == 0 for a in actions)/n, 4),
        "focused_fraction": round(sum(a >= 9 for a in actions)/n, 4),
        "border_proximity_fraction": round(edge/n, 4),
        "border_pushing_proxy_fraction": round(border_push/n, 4),
        "projectile_close_approach_proxy_fraction": round(near/n, 4),
        "action_changes_during_close_approach_proxy": near_switch,
        "first_recorded_death_increment_frame": first_death_frame,
        "top_action_choices": Counter(actions).most_common(5),
    }


def summarize_trace(path):
    src = read_trace(path)
    by_seed = {str(seed): summarize_one(rows) for seed, rows in src.items()}
    metrics = ("movement_action_changes_per_100_frames", "focused_fraction",
               "border_proximity_fraction", "border_pushing_proxy_fraction",
               "projectile_close_approach_proxy_fraction")
    return {
        "seed_reports": by_seed,
        "aggregate_mean": {k: round(mean(row[k] for row in by_seed.values()), 4)
                           for k in metrics},
    }


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--trace",action="append",nargs=2,metavar=("POLICY","JSONL"),required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args(argv)
    result={
        "status":"DESCRIPTIVE_NATIVE_TRAJECTORY_DIAGNOSTIC",
        "policy_promoted":False,
        "task":"Taisei Stage1 Easy development seeds 42100-42103",
        "policy_reports":{},
        "limitations":(
            "Proxy risk uses closest linear bullet approach <= 20 pixels, not actual collision/hitbox; "
            "border metrics are coordinate thresholds, not verified wall contact; "
            "first recorded death-increment frame need not equal first hit; "
            "observations follow policies' different trajectories. "
            "No inference of successful dodging or causal superiority from these proxies."
        )
    }
    for policy,path in a.trace:
        if policy in result["policy_reports"]:
            p.error("policy supplied more than once")
        result["policy_reports"][policy]=summarize_trace(path)
    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
