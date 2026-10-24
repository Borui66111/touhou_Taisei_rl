"""Summarize genuine native pilot evaluations. Does not promote policies."""
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sets = {}
for name, filename in [
    ("heuristic", "heuristic.jsonl"),
    ("random", "random.jsonl"),
    ("pilot_skill", "pilot_skill_evaluation.jsonl"),
]:
    p = root / filename
    data = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    seeds = [int(row["seed"]) for row in data]
    if seeds != list(range(42100, 42104)):
        raise SystemExit(f"Seed mismatch in {name}: {seeds}")
    sets[name] = {
        "n": len(data),
        "seed_range": [min(seeds), max(seeds)],
        "clears": sum(bool(row["cleared"]) for row in data),
        "mean_deaths": sum(row["deaths"] for row in data) / len(data),
        "mean_bombs": sum(row["bombs_used"] for row in data) / len(data),
        "mean_alive_frames": sum(row["alive_frames"] for row in data) / len(data),
        "mean_boss_damage": sum(row["boss_damage"] for row in data) / len(data),
    }
checkpoint = root / "pilot_skill.pt"
report = {
    "status": "experimental_pilot_not_promoted",
    "note": "1024-step pilot and 4 development episodes per policy are integration checks, not statistical evidence of improvement.",
    "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "results": sets,
    "sealed_final_seeds": "43000-43099 unused",
}
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
