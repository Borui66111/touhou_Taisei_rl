"""Source + archived-native reward audit (no simulator, no training).

Reads the reward coefficients straight from ``my_test/native_rl/env.py``
(``EnvConfig`` defaults, parsed with ``ast`` so nothing native is loaded),
reports config fields that ``TaiseiEnv.step`` never reads, re-derives every
archived native episode return from its logged components, and computes the
discounted per-event value of deaths / survival / boss damage under the PPO
discount.

Escrowed files are SHA-256 checked against the escrow manifest before use.
Output is descriptive evidence about the objective, not gameplay evidence.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import sys

REWARD_FIELDS = ("survival_weight", "damage_weight", "graze_weight",
                 "death_penalty", "loss_penalty", "win_bonus", "no_bomb_bonus")
EPISODE_FILES = ("e3/teacher_bc_seed48030_evaluation.jsonl",
                 "e3/teacher_bc_ppo_seed48020_evaluation.jsonl",
                 "e4/teacher_bc_seed48030_native.jsonl",
                 "e4/teacher_bc_rebalanced_seed48030_native.jsonl",
                 "e4/risk_teacher_native.jsonl")
SEALED = range(43000, 43100)


def env_reward_config(env_py: pathlib.Path):
    tree = ast.parse(env_py.read_text(encoding="utf-8"))
    defaults, read_in_step = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EnvConfig":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) \
                        and item.target.id in REWARD_FIELDS:
                    defaults[item.target.id] = ast.literal_eval(item.value)
        if isinstance(node, ast.FunctionDef) and node.name == "step":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in REWARD_FIELDS:
                    read_in_step.add(sub.attr)
    missing = set(REWARD_FIELDS) - set(defaults)
    if missing:
        raise ValueError(f"EnvConfig reward fields not found: {sorted(missing)}")
    return defaults, sorted(set(defaults) - read_in_step)


def module_dict_literal(path: pathlib.Path, name: str) -> dict:
    """Return a module-level ``NAME = dict(k=literal, ...)`` without importing it."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            call = node.value
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "dict":
                return {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    raise ValueError(f"{path}: {name} = dict(...) not found")


def verify_escrow(root: pathlib.Path, rel: str, manifest: dict) -> str:
    group, name = rel.split("/", 1)
    want = manifest["artifacts"][group]["files"][name]["sha256"]
    got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    if got != want:
        raise ValueError(f"{rel}: sha256 {got} != manifest {want}")
    return got


def decompose(ep: dict, w: dict) -> dict:
    lost = bool(ep["terminated"]) and not bool(ep["won"])
    parts = {
        "survival": ep["alive_frames"] * w["survival_weight"],
        "boss_damage": ep["boss_damage"] * w["damage_weight"],
        "graze": ep["graze"] * w["graze_weight"],
        "death": ep["deaths"] * w["death_penalty"],
        "terminal": (w["loss_penalty"] if lost else 0.0) + (w["win_bonus"] if ep["won"] else 0.0),
    }
    return {"parts": parts, "predicted": sum(parts.values()),
            "residual": ep["reward"] - sum(parts.values()),
            "non_alive_frames": ep["frames"] - ep["alive_frames"]}


def discounted_costs(w: dict, gamma: float, dead_frames_per_death: float) -> dict:
    horizon = 1.0 / (1.0 - gamma)
    lost_alive = (1.0 - gamma ** dead_frames_per_death) / (1.0 - gamma)
    s = w["survival_weight"]
    nonfinal = -w["death_penalty"] + s * lost_alive
    final = -w["death_penalty"] - w["loss_penalty"] + s * horizon
    return {"gamma": gamma, "effective_horizon_frames": horizon,
            "value_survive_indefinitely": s * horizon,
            "nonfinal_death_cost": nonfinal, "final_death_cost": final,
            "nonfinal_over_final": nonfinal / final,
            "nonfinal_death_in_survival_frames": nonfinal / s if s else None,
            "reward_for_24000_boss_hp": 24000 * w["damage_weight"],
            "weight_of_event_1000_frames_ahead": gamma ** 1000}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=".")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--out")
    a = p.parse_args(argv)
    repo = pathlib.Path(a.repo)
    env_py = repo / "my_test/native_rl/env.py"
    escrow = repo / "research/rsi/evidence/escrow_e3_e4_20261026_v1"
    manifest = json.loads((escrow / "manifest.json").read_text(encoding="utf-8"))
    w, unread = env_reward_config(env_py)

    per_file, all_eps = {}, []
    for rel in EPISODE_FILES:
        sha = verify_escrow(escrow, rel, manifest)
        eps = [json.loads(l) for l in (escrow / rel).read_text(encoding="utf-8").splitlines() if l.strip()]
        if any(int(e["seed"]) in SEALED for e in eps):
            raise ValueError(f"{rel}: sealed final seed present")
        rows = [decompose(e, w) for e in eps]
        total = sum(e["reward"] for e in eps)
        deaths = sum(e["deaths"] for e in eps)
        per_file[rel] = {
            "sha256": sha, "n": len(eps),
            "mean_return": total / len(eps),
            "return_share": {k: sum(r["parts"][k] for r in rows) / total for k in rows[0]["parts"]},
            "max_abs_residual": max(abs(r["residual"]) for r in rows),
            "non_alive_frames_per_death": (sum(r["non_alive_frames"] for r in rows) / deaths) if deaths else None,
            "truncated": sum(bool(e["truncated"]) for e in eps),
            "bombs_used": sum(int(e["bombs_used"]) for e in eps),
            "distinct_boss_damage_totals": sorted({float(e["boss_damage"]) for e in eps}),
        }
        all_eps += [(e, r) for e, r in zip(eps, rows)]

    terminated = [(e, r) for e, r in all_eps if e["terminated"] and e["deaths"]]
    dead_per_death = sum(r["non_alive_frames"] for _, r in terminated) / sum(e["deaths"] for e, _ in terminated)
    challenger = module_dict_literal(repo / "research/rsi/e1_train_movement.py", "CHALLENGER")
    report = {
        "schema": "taisei_rsi_reward_audit_v1",
        "native_game_executed": False, "training_executed": False,
        "env_reward_defaults": w,
        "config_fields_never_read_by_step": unread,
        "episodes_reconstructed": len(all_eps),
        "max_abs_residual_all": max(abs(r["residual"]) for _, r in all_eps),
        "non_alive_frames_per_death_terminated_eps": dead_per_death,
        "per_file": per_file,
        "discounted_costs_baseline": discounted_costs(w, a.gamma, dead_per_death),
        "e1_survival_first_challenger_weights": challenger,
        "discounted_costs_e1_survival_first_challenger": discounted_costs(
            dict(w, **challenger), a.gamma, dead_per_death),
    }
    out = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if a.out:
        pathlib.Path(a.out).write_text(out, encoding="utf-8")
    sys.stdout.write(out)
    return 0 if report["max_abs_residual_all"] < 1e-2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
