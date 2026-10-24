#!/usr/bin/env python3
"""Competence-first, fail-closed evaluator for authentic Taisei episode JSONL.

This evaluates RECORDS, not an imitation of Taisei. Missing checkpoint/task
provenance => descriptive comparison only; never auto-promote a policy.
"""
from __future__ import annotations
import argparse
import json
import math
import pathlib
import random
from statistics import mean

FINAL_SEED_START = 43000
FINAL_SEED_STOP = 43100
REQUIRED = ("seed", "cleared", "won", "deaths", "alive_frames",
            "frames", "bombs_used", "boss_damage")


def read_episode_rows(path):
    rows = []
    for line_no, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
        missing = [k for k in REQUIRED if k not in item]
        if missing:
            raise ValueError(f"{path}:{line_no}: missing {missing}")
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Seed must be an integer")
        if FINAL_SEED_START <= seed < FINAL_SEED_STOP:
            raise ValueError(f"Refusing sealed final evaluation seed {seed}")
        for name in ("deaths", "alive_frames", "frames", "bombs_used"):
            x = item[name]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
                raise ValueError(f"Invalid nonnegative metric {name}")
        if item["alive_frames"] > item["frames"]:
            raise ValueError("alive_frames cannot exceed frames")
        rows.append(item)
    keys = [r["seed"] for r in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError("Evaluation file empty or contains duplicated episode seeds")
    return {r["seed"]: r for r in rows}


def report_metrics(rows):
    rs = list(rows.values())
    n = len(rs)
    return {
        "n_episodes": n,
        "clears": sum(bool(r["cleared"]) for r in rs),
        "death_free_clears": sum(bool(r["cleared"]) and r["deaths"] == 0 for r in rs),
        "mean_deaths": mean(r["deaths"] for r in rs),
        "mean_alive_frames": mean(r["alive_frames"] for r in rs),
        "mean_bombs": mean(r["bombs_used"] for r in rs),
        "mean_boss_damage": mean(r["boss_damage"] for r in rs),
        "confirmed_item_pickups": (
            sum(r["confirmed_item_pickups"] for r in rs)
            if all(isinstance(r.get("confirmed_item_pickups"), int) for r in rs)
            else None
        ),
        "first_death_frame": (
            mean(r["first_death_frame"] for r in rs)
            if all(isinstance(r.get("first_death_frame"), (int, float)) for r in rs)
            else None
        ),
    }


def paired_ci(deltas, seed=1901, bootstrap_samples=2000):
    """Basic nonparametric paired percentile interval, not a guarantee of significance."""
    if not deltas:
        raise ValueError("No paired samples")
    gen = random.Random(seed)
    n = len(deltas)
    draws = sorted(mean(gen.choice(deltas) for _ in range(n))
                   for _ in range(bootstrap_samples))
    return [draws[int(0.025 * bootstrap_samples)],
            draws[min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))]]


def compare(baseline, candidate):
    if set(baseline) != set(candidate):
        raise ValueError("Paired comparison requires the same exact development seeds")
    order = sorted(baseline)
    diffs = {
        "alive_frames": [candidate[k]["alive_frames"] - baseline[k]["alive_frames"] for k in order],
        "deaths": [candidate[k]["deaths"] - baseline[k]["deaths"] for k in order],
        "death_free_clear": [
            int(bool(candidate[k]["cleared"]) and candidate[k]["deaths"] == 0)
            - int(bool(baseline[k]["cleared"]) and baseline[k]["deaths"] == 0)
            for k in order
        ],
        "bombs_used": [candidate[k]["bombs_used"] - baseline[k]["bombs_used"] for k in order],
    }
    return {
        "seeds": order,
        "baseline": report_metrics(baseline),
        "candidate": report_metrics(candidate),
        "paired_differences_candidate_minus_baseline": {
            k: {"mean": mean(v), "bootstrap_95pct_ci": paired_ci(v)} for k, v in diffs.items()
        },
        "comparability": "DESCRIPTIVE_ONLY",
        "promotion_allowed": False,
        "reason": "Episode JSONL does not establish identical build, task, checkpoint and training-budget provenance. Incomplete skill pickup/first-hit telemetry. No automatic champion promotion.",
        "sealed_final_seed_range": "43000-43099 untouched",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = compare(read_episode_rows(args.baseline),
                     read_episode_rows(args.candidate))
    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["comparability"],
                      "promotion_allowed": False,
                      "n": len(result["seeds"]),
                      "results_file": str(dest)}))


if __name__ == "__main__":
    main()
