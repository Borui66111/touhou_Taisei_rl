#!/usr/bin/env python3
"""E1: one-factor reward ablation on genuine native Taisei PPO.

Control and challenger share ALL environment, model and training settings except
the reward coefficients. No toy fallback. No champion promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib

CONTROL = {
    "survival_weight": 1.0, "damage_weight": 0.02, "graze_weight": 0.001,
    "death_penalty": -10.0, "loss_penalty": -10.0, "win_bonus": 20.0,
}
CHALLENGER = {
    "survival_weight": 0.002, "damage_weight": 0.001, "graze_weight": 0.0,
    "death_penalty": -30.0, "loss_penalty": -30.0, "win_bonus": 100.0,
}


def reward_config(variant):
    if variant == "control":
        return CONTROL.copy()
    if variant == "challenger":
        return CHALLENGER.copy()
    raise ValueError("Unknown reward arm")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=("control", "challenger"))
    p.add_argument("--library", required=True)
    p.add_argument("--resource-path")
    p.add_argument("--seed", type=int, default=48020)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--difficulty", type=int, default=2)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if args.steps < 512 or args.seed in range(43000, 43100):
        p.error("Too few steps or sealed evaluation seed is forbidden")
    output = pathlib.Path(args.output_dir)
    weights = reward_config(args.variant)
    settings = {
        "experiment": "E1_one_factor_reward_ablation",
        "status": "untrained_preflight" if args.preflight_only else "training_requested",
        "variant": args.variant,
        "reward": weights, "training_seed": args.seed,
        "stage": args.stage, "difficulty": args.difficulty,
        "practice": False, "frames_per_action": 1,
        "steps": args.steps, "rollout_steps": 512,
        "learning_rate": 0.0003, "gamma": 0.995,
        "policy_type": "skill", "skill_action_mask": True,
        "final_test_seeds": "43000-43099 sealed",
        "note": "Mask v2 excludes early Power Surge discharge in both arms. The experiment isolates reward weights only.",
    }
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(settings, indent=2), flush=True)
    if args.preflight_only:
        return
    from native_rl.env import EnvConfig
    from native_rl.torch_ppo import train
    cfg = EnvConfig(library_path=args.library, resource_path=args.resource_path,
                    stage_id=args.stage, difficulty=args.difficulty, **weights)
    checkpoint = output / f"{args.variant}_seed{args.seed}.pt"
    training_log = output / f"{args.variant}_seed{args.seed}_train.jsonl"
    manifest = output / f"{args.variant}_seed{args.seed}_manifest.json"
    if checkpoint.exists() or training_log.exists() or manifest.exists():
        raise RuntimeError("Refusing to overwrite existing training evidence")
    manifest.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    train(cfg, outfile=str(checkpoint), total_steps=args.steps, rollout_steps=512,
          learning_rate=0.0003, gamma=0.995, seed=args.seed,
          device="cpu", policy_type="skill", skill_action_mask=True,
          log_path=str(training_log))
    settings["status"] = "training_completed_not_validated"
    settings["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"E1 checkpoint sha256: {settings['checkpoint_sha256']}", flush=True)


if __name__ == "__main__":
    main()
