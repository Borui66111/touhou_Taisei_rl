#!/usr/bin/env python3
"""Validate a rebuilt libtaisei_sim against archived *original* taisei-sim evidence.

The original simulator source (touhourl/taisei-sim@789200ff) is unavailable. This
script checks that a reconstruction built from public Taisei reproduces what the
original produced, using only escrowed native evidence from E3/E4 (run 40219788151,
40220975924):

  1. smoke      - scripts/smoke_native.py, stage 1 Easy, seed 42000, 300 frames:
                  every printed field except the gameplay digest.
  2. determinism- scripts/check_native_determinism.py (same settings as E3/E4):
                  PASS, identical_frames, observed_max_projectiles.
  3. demo_replay- the 12,000 archived teacher actions (seed 48030, then 48031 after the
                  game over) replayed open-loop; every 528-d observation vector must be
                  identical to the archived one.
  4. teacher_e4 - the unchanged risk-aware teacher on development seeds 42132-42147
                  (closed loop); every per-episode metric except the digest.
  5. teacher_e2 - E2 (run 40213025937): the geometric heuristic and the risk-aware teacher
                  on seeds 42100-42115; all eight published per-policy aggregates.
  6. replay_roundtrip (optional, needs --taisei-exe built from the same patched tree):
                  the first archived teacher episode (through the stage-1 boss dialogue and
                  game over) is saved as .tsr and must pass the real game's headless
                  --verify-replay; without TAISEI_SIM_SKIP_DIALOG_WAITS it must desync
                  (negative control).

The gameplay digest is a hash whose original definition is unknown; it is never compared.
Each check runs in its own process (one simulation per process). No sealed seeds are used.
Exit status 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
MY_TEST = ROOT / "my_test"
ESCROW = ROOT / "research/rsi/evidence/escrow_e3_e4_20261026_v1"
DEMOS = ESCROW / "e3/teacher_demonstrations_12000.npz"

# Import-only torch stand-in so planner-only evaluation (no neural policy) can import
# native_rl.torch_ppo where PyTorch is not installed. Any tensor use raises.
TORCH_STUB = r'''
import sys, types
try:
    import torch  # noqa: F401
except Exception:
    class _U:
        def __init__(self,*a,**k): pass
        def __call__(self,*a,**k): raise RuntimeError("torch unavailable")
    t=types.ModuleType("torch"); nn=types.ModuleType("torch.nn")
    class Module:
        def __init__(self,*a,**k): pass
    nn.Module=Module
    for n in ("Sequential","Linear","Tanh","ReLU","Identity"): setattr(nn,n,_U)
    d=types.ModuleType("torch.distributions"); d.Categorical=_U
    o=types.ModuleType("torch.optim"); o.Adam=_U
    t.nn,t.distributions,t.optim=nn,d,o
    sys.modules.update({"torch":t,"torch.nn":nn,"torch.distributions":d,"torch.optim":o})
'''

DEMO_REPLAY = r'''
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from native_rl.env import EnvConfig, TaiseiEnv
lib, res, npz = sys.argv[2], sys.argv[3], sys.argv[4]
def features(obs):  # identical to native_rl.torch_ppo.features (legacy 528-d vector)
    return np.asarray(obs['player'] + obs['boss'] +
        [v for b in obs['bullets'] for v in b] + [v for l in obs['lasers'] for v in l] +
        [obs['projectile_count']/1000.0, obs['laser_count']/50.0], dtype=np.float32)
d = np.load(npz); S, A = d['states'], d['actions']
cfg = EnvConfig(library_path=lib, resource_path=res, stage_id=1, difficulty=1,
                practice=False, max_frames=10000, frames_per_action=1)
seed = 48030; episodes = 0; worst = 0.0; first_bad = None; exact = 0; resets = [0]
with TaiseiEnv(cfg) as env:
    obs, _ = env.reset(seed=seed)
    for i in range(len(A)):
        x = features(obs)
        if x.shape != S[i].shape:
            raise SystemExit(f"feature shape {x.shape} != {S[i].shape}")
        e = float(np.max(np.abs(x - S[i])))
        worst = max(worst, e)
        exact += int(e == 0.0)
        if e != 0.0 and first_bad is None:
            first_bad = {"step": i, "features": np.where(x != S[i])[0][:8].tolist()}
        obs, _, term, trunc, _ = env.step(int(A[i]))
        if term or trunc:
            episodes += 1; resets.append(i + 1); obs, _ = env.reset(seed=seed + episodes)
print(json.dumps({"steps": int(len(A)), "bitwise_identical_rows": exact,
                  "max_abs_diff": worst, "first_divergence": first_bad,
                  "episode_boundaries": resets}))
'''

TEACHER = TORCH_STUB + r'''
import json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from native_rl.env import EnvConfig
from native_rl.torch_ppo import evaluate
from risk_aware_teacher import risk_aware_action
cfg = EnvConfig(library_path=sys.argv[3], resource_path=sys.argv[4], stage_id=1, difficulty=1,
                practice=False, max_frames=10000, frames_per_action=1)
rows = evaluate(cfg, checkpoint_path=None, episodes=16, seed=42132, sampled=False,
                planner=risk_aware_action)
print(json.dumps(rows))
'''


TEACHER_E2 = TORCH_STUB + r'''
import json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from native_rl.env import EnvConfig
from native_rl.planner import heuristic_action
from native_rl.torch_ppo import evaluate
from risk_aware_teacher import risk_aware_action
cfg = EnvConfig(library_path=sys.argv[3], resource_path=sys.argv[4], stage_id=1, difficulty=1,
                practice=False, max_frames=10000, frames_per_action=1)
planner = heuristic_action if sys.argv[5] == "baseline" else risk_aware_action
print(json.dumps(evaluate(cfg, checkpoint_path=None, episodes=16, seed=42100, sampled=False,
                          planner=planner)))
'''

RECORD_EPISODE = r'''
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from native_rl.env import EnvConfig, TaiseiEnv
cfg = EnvConfig(library_path=sys.argv[2], resource_path=sys.argv[3], stage_id=1, difficulty=1,
                practice=False, max_frames=10000, frames_per_action=1)
actions = np.load(sys.argv[4])['actions']
with TaiseiEnv(cfg) as env:
    env.reset(seed=48030)
    for i, a in enumerate(actions):
        obs, _, term, trunc, info = env.step(int(a))
        if term or trunc:
            break
    env.native.save_replay(sys.argv[5])
print(json.dumps({"steps": i + 1, "status": info["status"], "deaths": obs["deaths"]}))
'''


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(MY_TEST))
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[:3]} failed ({p.returncode}): {p.stderr[-2000:]}")
    return p.stdout


def strip_digest(line):
    d = json.loads(line.split("RESET:", 1)[1] if line.startswith("RESET:") else line)
    d.pop("digest", None)
    return d


def check_smoke(lib, res):
    out = run([sys.executable, "scripts/smoke_native.py", "--library", lib, "--resource-path", res,
               "--stage", "1", "--difficulty", "1", "--seed", "42000", "--frames", "300"])
    got = [strip_digest(l) for l in out.splitlines() if l.startswith(("RESET:", "{"))]
    results = {}
    for run_name in ("e3", "e4"):
        ref_lines = (ESCROW / run_name / "native_smoke.log").read_text().splitlines()
        ref = [strip_digest(l) for l in ref_lines if l.startswith(("RESET:", "{"))]
        results[run_name] = {"lines": len(ref), "identical": got == ref}
    return {"pass": all(r["identical"] for r in results.values()), "archives": results}


def check_determinism(lib, res):
    out = run([sys.executable, "scripts/check_native_determinism.py", "--library", lib,
               "--resource-path", res, "--stage", "1", "--difficulty", "1",
               "--seed", "42000", "--frames", "600"])
    got = json.loads(out[out.index("{"):])
    results = {}
    for run_name in ("e3", "e4"):
        ref = json.loads((ESCROW / run_name / "determinism.log").read_text())
        keys = ("result", "identical_frames", "seed", "stage", "observed_max_projectiles")
        results[run_name] = {k: {"original": ref[k], "rebuilt": got[k]} for k in keys}
        results[run_name]["identical"] = all(ref[k] == got[k] for k in keys)
    return {"pass": all(r["identical"] for r in results.values()), "archives": results}


def check_demo_replay(lib, res):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(DEMO_REPLAY)
    try:
        out = json.loads(run([sys.executable, f.name, str(MY_TEST), lib, res, str(DEMOS)]).splitlines()[-1])
    finally:
        os.unlink(f.name)
    out["pass"] = out["bitwise_identical_rows"] == out["steps"] == 12000
    return out


def check_teacher(lib, res):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(TEACHER)
    try:
        rows = json.loads(run([sys.executable, f.name, str(MY_TEST), str(ROOT / "research/rsi"),
                               lib, res]).splitlines()[-1])
    finally:
        os.unlink(f.name)
    ref = [json.loads(l) for l in (ESCROW / "e4/risk_teacher_native.jsonl").read_text().splitlines()]
    keys = sorted(set(ref[0]) - {"digest"})
    mism = [{"seed": a["seed"], "field": k, "original": a[k], "rebuilt": b.get(k)}
            for a, b in zip(ref, rows) for k in keys if a[k] != b.get(k)]
    return {"pass": len(rows) == len(ref) == 16 and not mism, "episodes": len(rows),
            "fields_compared": keys, "field_mismatches": mism[:50],
            "n_field_mismatches": len(mism),
            "mean_alive_frames": {"original": sum(r["alive_frames"] for r in ref) / len(ref),
                                  "rebuilt": sum(r["alive_frames"] for r in rows) / max(1, len(rows))}}


def check_teacher_e2(lib, res):
    ref = json.loads((ROOT / "research/rsi/evidence/native_e2_teacher_40213025937.json").read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(TEACHER_E2)
    out = {}
    try:
        for key in ("baseline", "candidate"):
            rows = json.loads(run([sys.executable, f.name, str(MY_TEST), str(ROOT / "research/rsi"),
                                   lib, res, key]).splitlines()[-1])
            n = len(rows)
            got = {"clears": sum(int(r["cleared"]) for r in rows),
                   "mean_alive_frames": sum(r["alive_frames"] for r in rows) / n,
                   "mean_deaths": sum(r["deaths"] for r in rows) / n,
                   "mean_bombs": sum(r["bombs_used"] for r in rows) / n,
                   "mean_boss_damage": sum(r["boss_damage"] for r in rows) / n,
                   "episodes_truncated": sum(int(r["truncated"]) for r in rows),
                   "special_activations": sum(r["special_activations"] for r in rows),
                   "mean_graze": sum(r["graze"] for r in rows) / n}
            out[key] = {k: {"original": ref[key][k], "rebuilt": v,
                            "identical": abs(ref[key][k] - v) <= 1e-9} for k, v in got.items()}
            out[key]["episodes"] = n
    finally:
        os.unlink(f.name)
    out["pass"] = all(all(v["identical"] for k, v in out[key].items() if k != "episodes")
                      and out[key]["episodes"] == 16 for key in ("baseline", "candidate"))
    return out


def check_replay_roundtrip(lib, res, exe):
    with tempfile.TemporaryDirectory() as td:
        script = pathlib.Path(td) / "record.py"
        script.write_text(RECORD_EPISODE)
        tsr = pathlib.Path(td) / "teacher_48030.tsr"
        rec = json.loads(run([sys.executable, str(script), str(MY_TEST), lib, res, str(DEMOS),
                              str(tsr)]).splitlines()[-1])
        results = {"recorded": rec, "tsr_bytes": tsr.stat().st_size}
        for skip in ("1", "0"):
            storage = pathlib.Path(td) / f"storage{skip}"
            env = dict(os.environ, TAISEI_SIM_SKIP_DIALOG_WAITS=skip, TAISEI_RES_PATH=res,
                       TAISEI_STORAGE_PATH=str(storage), SDL_AUDIODRIVER="dummy",
                       SDL_VIDEODRIVER="dummy")
            p = subprocess.run([exe, "--verify-replay", str(tsr)], env=env, capture_output=True,
                               text=True, timeout=1800)
            desync = [l for l in (p.stdout + p.stderr).splitlines() if "desync detected" in l]
            results[f"verify_skip_dialog_{skip}"] = {"returncode": p.returncode, "desync": desync[:2]}
    results["pass"] = (results["verify_skip_dialog_1"]["returncode"] == 0
                       and not results["verify_skip_dialog_1"]["desync"]
                       and results["verify_skip_dialog_0"]["returncode"] != 0
                       and rec["steps"] == 7857 and rec["status"] == 3)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--library", required=True)
    p.add_argument("--resource-path", required=True)
    p.add_argument("--out", help="write the JSON report here (refuses to overwrite)")
    p.add_argument("--upstream-commit", default=None, help="recorded in the report")
    p.add_argument("--patch", default=None, help="reconstruction patch; its sha256 is recorded")
    p.add_argument("--taisei-exe", default=None,
                   help="patched game executable from the same build: enables replay_roundtrip")
    a = p.parse_args(argv)
    lib, res = str(pathlib.Path(a.library).resolve()), str(pathlib.Path(a.resource_path).resolve())
    report = {
        "schema": "rebuilt_sim_validation_v1",
        "library_sha256": sha256(lib),
        "upstream_commit": a.upstream_commit,
        "patch_sha256": sha256(a.patch) if a.patch else None,
        "reference": "escrow_e3_e4_20261026_v1 (original taisei-sim 789200ff; runs 40219788151, 40220975924)",
        "digest_compared": False,
        "sealed_final_seeds_used": 0,
        "checks": {},
    }
    checks = [("smoke", check_smoke), ("determinism", check_determinism),
              ("demo_replay", check_demo_replay), ("teacher_e4", check_teacher),
              ("teacher_e2", check_teacher_e2)]
    if a.taisei_exe:
        exe = str(pathlib.Path(a.taisei_exe).resolve())
        checks.append(("replay_roundtrip", lambda l, r: check_replay_roundtrip(l, r, exe)))
    for name, fn in checks:
        try:
            report["checks"][name] = fn(lib, res)
        except Exception as exc:  # report, do not hide
            report["checks"][name] = {"pass": False, "error": str(exc)[-2000:]}
        print(json.dumps({"check": name, "pass": report["checks"][name]["pass"]}), flush=True)
    report["pass"] = all(c["pass"] for c in report["checks"].values())
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if a.out:
        with open(a.out, "x", encoding="utf-8") as f:
            f.write(text)
    print(json.dumps({"pass": report["pass"]}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
