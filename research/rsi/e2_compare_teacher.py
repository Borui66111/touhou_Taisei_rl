#!/usr/bin/env python3
"""Compare two preregistered authentic-Ta​​isei deterministic movement controllers.

Native episode summaries only. Bootstrap uncertainty covers game seeds,
not model-training initialization, and result never promotes a policy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import mean

SEEDS = list(range(42100, 42116))


def read(path):
    data = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if [row["seed"] for row in data] != SEEDS:
        raise ValueError("Expected the exact 16 ordered Easy development seeds")
    required = ("cleared", "deaths", "alive_frames", "bombs_used", "boss_damage")
    for row in data:
        if any(key not in row for key in required):
            raise ValueError("Incomplete native episode row")
        if row["alive_frames"] is None:
            raise ValueError("Exact one-frame native aliveness measurement required")
        if row["bombs_used"] != 0 or row.get("bomb_attempts", 0):
            raise ValueError("This movement-only experiment must use no Bombs")
        if any(43000 <= int(row["seed"]) <= 43099 for _ in [0]):
            raise ValueError("Sealed seed not allowed")
    return data


def bootstrap(values, count=4000):
    gen = random.Random(20261026)
    arr = sorted(mean(gen.choice(values) for _ in range(len(values))) for _ in range(count))
    return [arr[int(.025 * count)], arr[int(.975 * count)]]


def describe(rows):
    return {
        "n": len(rows),
        "clears": sum(bool(x["cleared"]) for x in rows),
        "mean_alive_frames": mean(x["alive_frames"] for x in rows),
        "mean_deaths": mean(x["deaths"] for x in rows),
        "mean_bombs_used": mean(x["bombs_used"] for x in rows),
        "mean_boss_damage": mean(x["boss_damage"] for x in rows),
    }


def compare(baseline_path, teacher_path):
    old = read(baseline_path)
    new = read(teacher_path)
    differences = [y["alive_frames"] - x["alive_frames"] for x,y in zip(old,new)]
    baseline = describe(old)
    candidate = describe(new)
    return {
        "experiment": "E2_native_risk_teacher_v1",
        "status": "diagnostic_only_not_promoted",
        "task": "authentic Taisei Stage 1 Easy, no Bomb/Surge, 1 frame/action",
        "game_seeds": SEEDS,
        "baseline": baseline,
        "candidate": candidate,
        "paired_alive_delta": {"mean": mean(differences),
                               "eval_seed_bootstrap_95pct_ci": bootstrap(differences)},
        "paired_clear_delta": sum(bool(y["cleared"]) - bool(x["cleared"]) for x,y in zip(old,new))/len(old),
        "baseline_sha256": hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest(),
        "promote_policy": False,
        "limits": [
            "Teacher is handcrafted, not learned RSI policy.",
            "One native task and 16 development seeds; cannot claim generalization.",
            "Simplified linear bullet/laser model is not exact collision logic.",
            "Aliveness and encounter clears do not certify risk-conditioned dodging or item pickup.",
            "Must inspect native trace/border and skill/invulnerability confounds separately."
        ],
        "research_decision": (
            "If teacher beats baseline, consider teacher distillation as a separately "
            "registered learning experiment; otherwise revise/falsify risk representation "
            "before consuming more PPO training compute."
        )
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dest = Path(a.out)
    if dest.exists():
        p.error("Refusing to overwrite comparison evidence")
    result = compare(a.baseline, a.teacher)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
