#!/usr/bin/env python3
"""E2 risk-aware movement teacher against native Taisei; no training or toy mode."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "my_test"))
from native_rl.env import EnvConfig
from native_rl.planner import heuristic_action
from native_rl.torch_ppo import evaluate
from risk_aware_teacher import risk_aware_action


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--library", required=True)
    p.add_argument("--resource-path", required=True)
    p.add_argument("--policy", required=True, choices=("existing_heuristic", "risk_teacher_v1"))
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42100)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=10000)
    a = p.parse_args(argv)
    if (a.stage, a.difficulty, a.seed, a.episodes) != (1, 1, 42100, 16):
        p.error("E2 preregistered native task: Stage 1 Easy, seeds 42100..42115")
    if not 1 <= a.max_frames <= 10000:
        p.error("max-frames must be within 1..10000")
    if not Path(a.library).is_file() or not Path(a.resource_path).exists():
        p.error("Actual Taisei native library and resources are mandatory")
    out = Path(a.out)
    if out.exists():
        p.error(f"Refusing overwrite of evidence: {out}")
    return a


def main(argv=None):
    a = parse_args(argv)
    cfg = EnvConfig(library_path=a.library, resource_path=a.resource_path,
                    stage_id=1, difficulty=1, practice=False,
                    frames_per_action=1, max_frames=a.max_frames)
    planner = heuristic_action if a.policy == "existing_heuristic" else risk_aware_action
    results = evaluate(cfg, checkpoint_path=None, episodes=16, seed=42100,
                       sampled=False, planner=planner)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        for row in results:
            if 43000 <= int(row["seed"]) <= 43099:
                raise RuntimeError("Sealed seed encountered")
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"policy": a.policy, "n": len(results),
        "clears": sum(bool(x["cleared"]) for x in results),
        "mean_deaths": sum(x["deaths"] for x in results)/len(results),
        "mean_alive_frames": sum(x["alive_frames"] for x in results)/len(results),
        "mean_bombs": sum(x["bombs_used"] for x in results)/len(results)},
        sort_keys=True))


if __name__ == "__main__":
    main()
