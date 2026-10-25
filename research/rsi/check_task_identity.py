#!/usr/bin/env python3
"""Fail-closed source-level curriculum identity check (not native gameplay)."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SCRIPT = (ROOT / "run_pilot.sh").read_text(encoding="utf-8")
assert re.search(r"^EASY_DIFFICULTY=1$", SCRIPT, re.MULTILINE), "Easy must equal native D_Easy=1"
assert SCRIPT.count('--difficulty "$EASY_DIFFICULTY"') == 4, (
    "Native smoke, training, heuristic/random loop, and PPO evaluation must all share Easy"
)
assert not re.search(r"--difficulty\s+[02-9](?:\s|$)", SCRIPT), "Wrong/native-invalid difficulty literal"
assert "difficulty_id=$EASY_DIFFICULTY" in SCRIPT, "Experiment manifest lacks difficulty"
assert "stage_id=1" in SCRIPT, "Experiment manifest lacks stage"
assert SCRIPT.count('meson devenv -C "$ROOT/build" python') == 4, (
    "Meson runtime invocation must directly specify python; separator -- previously failed"
)
assert 'meson devenv -C "$ROOT/build" -- python' not in SCRIPT, (
    "Meson interpreted -- as a command on GitHub runner"
)

assert "42100" in SCRIPT and "43000-43099" in SCRIPT, "Development/final seed separation lost"
print("PASS: all pilot codepaths explicitly pin native Easy=1; manifest and seed guard present.")
