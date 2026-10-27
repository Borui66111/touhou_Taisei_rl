#!/usr/bin/env python3
"""Local-first RSI operator: provenance, archive audit, native replay and evidence gates.

No GitHub Actions, no substitute physics or rendered frames, no champion changes.
Source-only/offline work is allowed without the pinned native Taisei checkout.
With a verified checkout, the native E5 evaluation runs in fresh processes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

PIN = "789200ff255c681b3ed4230a3f26f0b8c12c78fe"
URL = "https://github.com/touhourl/taisei-sim.git"
DATA_SHA = "02dd11feb60e0dcb5710ae8af3cf33dca7e7e77756d6467f822ad94d0f5a0d8d"
BC_SHA = "87b43918d0181355435f105e3fcec44d7fd314c12aec6978b63ec33e9955cf54"
E4_SHA = "06c344d1c9415d1277e89c7d7c1a4519f274952f159e1cc5095d7fecf696e6fc"
SEALED_FIRST, SEALED_LAST = 43000, 43099
E3_SEEDS = list(range(42100, 42116))
E4_SEEDS = list(range(42132, 42148))
E5_SEEDS = [42132, 42133]
REQUIRED_REPO = (
    "my_test/UPSTREAM.lock", "my_test/scripts/smoke_native.py",
    "my_test/scripts/check_native_determinism.py", "my_test/scripts/evaluate_native.py",
    "my_test/native_rl/taisei_ffi.py", "my_test/native_rl/torch_ppo.py",
    "research/rsi/e5_audit_replay.py", "research/rsi/verify_pinned_local_checkout.py",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never risk a half-written research state, including process termination.
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def log_event(path, stage, status, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "stage": stage, "status": status, **extra}
    # Each event is one complete line, flushed to disk before proceeding.
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(cmd, *, cwd=None, timeout=20, env=None, log=None):
    """Run one real process, kill process group on deadline, log return code."""
    if timeout <= 0:
        raise TimeoutError("Total execution budget exhausted")
    if not cmd or any(not isinstance(x, str) for x in cmd):
        raise ValueError("Subprocess command must be explicit arguments")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True, errors="replace")
    try:
        stdout, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        stdout, _ = p.communicate()
        if log is not None:
            Path(log).write_text(stdout, encoding="utf-8")
        raise TimeoutError(f"Process timed out: {cmd[0]} (limit {timeout:.0f}s)")
    if log is not None:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(stdout, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"Process failed ({p.returncode}): {' '.join(cmd[:3])}; see {log or 'captured log'}; tail={stdout[-500:]}")
    return stdout


def checkout_status(checkout, lock):
    """Independent strict, offline verification (not just HEAD and non-'+' submodule list)."""
    checkout = Path(checkout).resolve()
    lock = Path(lock)
    result = {"status": "blocked", "expected_commit": PIN,
              "source_path": str(checkout), "native_game_executed": False,
              "origin_verified": False, "commit_verified": False,
              "clean_verified": False, "submodules_verified": False, "reason": None}
    if not checkout.is_dir():
        result["reason"] = "Exact pinned native simulator Git checkout absent"
        return result
    if not lock.is_file() or lock.read_text(encoding="utf-8").splitlines() != [f"REPO_URL={URL}", f"COMMIT={PIN}"]:
        result["reason"] = "Missing/incorrect exact source lock my_test/UPSTREAM.lock"
        return result
    if not (checkout / ".git").exists():
        result["reason"] = "Checkout lacks Git metadata; ZIP or simulator binary cannot certify pinned revision"
        return result
    def git(*args):
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/false", GCM_INTERACTIVE="never")
        return run(["git", "-C", str(checkout), *args], timeout=20, env=env).strip("\n")
    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != checkout:
            raise ValueError("Checkout path is not its Git worktree root")
        if git("remote", "get-url", "origin") != URL:
            raise ValueError("Checkout Git origin differs from pinned upstream")
        result["origin_verified"] = True
        if git("rev-parse", "HEAD") != PIN or git("cat-file", "-t", PIN) != "commit":
            raise ValueError("Git HEAD or commit object does not equal simulator pin")
        result["commit_verified"] = True
        if git("status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"):
            raise ValueError("Simulator working tree or recursive submodules dirty")
        result["clean_verified"] = True
        if not (checkout / ".gitmodules").is_file():
            raise ValueError("Submodule manifest absent")
        submodules = git("submodule", "status", "--recursive").splitlines()
        if not submodules or any(not re.fullmatch(r" [0-9a-f]{40} \S+(?: .*)?", s) for s in submodules):
            raise ValueError("Recursive submodule(s) missing, mismatched or uninitialized")
        result["submodules_verified"] = True
        result["submodule_count"] = len(submodules)
    except (ValueError, TimeoutError, RuntimeError, OSError) as e:
        result["reason"] = str(e)
        return result
    result["status"] = "verified"
    result["reason"] = "Exact pinned Git origin/HEAD/cleanliness/submodules verified; game not yet executed"
    return result


def policy_archive_paths(root):
    root = Path(root)
    # Use immutable local evidence when present; otherwise canonical escrow stages.
    escrow = root / "research/rsi/evidence/escrow_e3_e4_20261026_v1"
    candidate = [(root / "archival_e3", root / "archival_e4"),
                 (escrow / "e3", escrow / "e4"),
                 (root / "my_test/results/e3_artifact", root / "my_test/results/e4_artifact")]
    return next(((a, b) for a, b in candidate if a.is_dir() and b.is_dir()), candidate[0])


def archived_evidence(root):
    e3, e4 = policy_archive_paths(root)
    check = [(e3 / "teacher_demonstrations_12000.npz", DATA_SHA),
             (e3 / "teacher_bc_seed48030.pt", BC_SHA),
             (e4 / "teacher_bc_seed48030.pt", BC_SHA),
             (e4 / "teacher_bc_rebalanced_seed48030.pt", E4_SHA)]
    files = []
    for p, expected in check:
        if not p.is_file():
            raise FileNotFoundError("Missing authentic archived evidence: " + str(p))
        actual = sha256(p)
        if actual != expected:
            raise ValueError(f"Archived evidence SHA-256 mismatch: {p}: {actual}")
        files.append({"file": str(p), "bytes": p.stat().st_size, "sha256": actual})
    return {"e3": str(e3), "e4": str(e4), "verified_files": files}


def checked_episodes(file, seeds):
    rows = [json.loads(x) for x in Path(file).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != len(seeds) or [x.get("seed") for x in rows] != seeds:
        raise ValueError("Missing, reordered, unexpected or sealed development seeds: " + str(file))
    keys = ("alive_frames", "frames", "deaths", "bombs_used", "boss_damage", "cleared",
            "truncated", "terminated", "digest", "logical_frame")
    for x in rows:
        if any(k not in x for k in keys) or x["frames"] < 1:
            raise ValueError("Incomplete genuine native episode row")
        if x["cleared"] and x["truncated"]:
            raise ValueError("Truncated native episode mislabeled a clear")
        if x["bombs_used"] != 0:
            raise ValueError("No-skill development evaluation unexpectedly used Bomb")
    return rows


def paired_comparison(root):
    """Read-only re-audit of authentic archived gameplay results, never new gameplay."""
    p = archived_evidence(root)
    e3, e4 = Path(p["e3"]), Path(p["e4"])
    groups = {
      "e3_bc": checked_episodes(e3 / "teacher_bc_seed48030_evaluation.jsonl", E3_SEEDS),
      "e3_bc_ppo": checked_episodes(e3 / "teacher_bc_ppo_seed48020_evaluation.jsonl", E3_SEEDS),
      "e4_bc": checked_episodes(e4 / "teacher_bc_seed48030_native.jsonl", E4_SEEDS),
      "e4_weighted": checked_episodes(e4 / "teacher_bc_rebalanced_seed48030_native.jsonl", E4_SEEDS),
      "e4_teacher": checked_episodes(e4 / "risk_teacher_native.jsonl", E4_SEEDS),
    }
    def summary(rows):
        n = len(rows)
        return {"n": n, "mean_alive_frames": sum(x["alive_frames"] for x in rows) / n,
                "stage_clears": sum(bool(x["cleared"]) for x in rows),
                "mean_deaths": sum(x["deaths"] for x in rows) / n,
                "timeouts": sum(bool(x["truncated"]) for x in rows),
                "mean_boss_damage": sum(x["boss_damage"] for x in rows) / n,
                "mean_bombs": sum(x["bombs_used"] for x in rows) / n}
    groups_sum = {k: summary(v) for k, v in groups.items()}
    e3_delta = groups_sum["e3_bc_ppo"]["mean_alive_frames"] - groups_sum["e3_bc"]["mean_alive_frames"]
    e4_delta = groups_sum["e4_weighted"]["mean_alive_frames"] - groups_sum["e4_bc"]["mean_alive_frames"]
    recorded_e3 = json.loads((e3 / "e3_comparison.json").read_text())
    recorded_e4 = json.loads((e4 / "e4_native_comparison.json").read_text())
    if abs(e3_delta - recorded_e3["paired_alive_delta_after_ppo"]) > 1e-8:
        raise ValueError("E3 archived comparison disagrees with independent per-episode arithmetic")
    if abs(e4_delta - recorded_e4["paired_candidate_minus_baseline_mean_alive"]) > 1e-8:
        raise ValueError("E4 archived comparison disagrees with independent per-episode arithmetic")
    return {"status": "archived_native_episode_reaudit_passed", "evidence": p,
            "archived_native_episode_groups": groups_sum,
            "paired_e3_ppo_minus_bc": e3_delta, "paired_e4_weighted_minus_bc": e4_delta,
            "fresh_native_game_executed": False, "new_gameplay_improvement_verified": False,
            "genuine_video_examined": False, "sealed_seeds_used": False,
            "limitation": "Historical genuine engine results are rechecked, not re-executed or visually inspected"}


def doctor(root):
    root = Path(root).resolve()
    source = root / "my_test/upstream/taisei-sim-pinned"
    checkout = checkout_status(source, root / "my_test/UPSTREAM.lock")
    tools = {name: bool(shutil.which(name)) for name in
             ("python", "git", "meson", "ninja", "cmake", "gcc", "g++", "pkg-config", "ffmpeg", "ffprobe", "xvfb-run")}
    missing = [x for x in REQUIRED_REPO if not (root / x).is_file()]
    current_branch = None
    if (root / ".git").exists():
        try:
            current_branch = run(["git", "-C", str(root), "branch", "--show-current"], timeout=8).strip()
        except (RuntimeError, TimeoutError):
            current_branch = "UNKNOWN"
    lib = list((source / "build").rglob("libtaisei_sim.so")) if (source / "build").is_dir() else []
    return {"root": str(root), "canonical_research_branch_required": "research/rsi-automation-v1",
            "source": checkout, "tools": tools, "native_library": str(lib[0]) if lib else None,
            "resources_available": (source / "resources").is_dir(), "missing_canonical_source_files": missing,
            "current_repo_branch": current_branch,
            "canonical_checkout_verified": current_branch == "research/rsi-automation-v1",
            "offline_dataset_available": (policy_archive_paths(root)[0] / "teacher_demonstrations_12000.npz").is_file(),
            "native_ready": checkout["status"] == "verified" and bool(lib) and
                   (source / "resources").is_dir() and not missing and tools["meson"]
                   and current_branch == "research/rsi-automation-v1",
            "no_github_actions_fallback": True, "fresh_native_game_executed": False}


def remaining(deadline):
    left = deadline - time.monotonic()
    if left <= 1:
        raise TimeoutError("RSI native run exceeded global execution budget")
    return left


def native_step(root, source, library, args, output, name, deadline):
    cmd = ["meson", "devenv", "-C", str(source / "build"), sys.executable, str(root / args[0]), *args[1:]]
    run(cmd, cwd=root, timeout=remaining(deadline), log=output / f"{name}.log")
    return {"name": name, "command": cmd, "log": str(output / f"{name}.log")}


def native_build_if_needed(root, deadline, logdir):
    """Build only an exact verified local checkout; never clone a substitute game.

    Catches missing dependencies before expensive compilation. It does not
    invoke GitHub Actions, an alternative game, or an unbounded installer.
    """
    root = Path(root).resolve()
    logdir = Path(logdir)
    st = doctor(root)
    source = root / "my_test/upstream/taisei-sim-pinned"
    if st["source"]["status"] != "verified":
        raise RuntimeError("Exact native source identity gate failed: " + st["source"]["reason"])
    if not st["canonical_checkout_verified"]:
        raise RuntimeError("Canonical research branch checkout is required for native execution")
    if st["missing_canonical_source_files"]:
        raise RuntimeError("Full native Python/replay workbench missing: " + str(st["missing_canonical_source_files"]))
    if not shutil.which("meson"):
        raise RuntimeError("Meson absent (>=1.8 required): install from an authorized local wheel/system package")
    if not shutil.which("ninja") or not shutil.which("cmake"):
        raise RuntimeError("Ninja/CMake build prerequisites absent")
    if not (source / "resources").is_dir():
        raise RuntimeError("Pinned game resources directory absent")
    if st["native_library"]:
        return {"status": "cached_native_library", "path": st["native_library"],
                "sha256": sha256(st["native_library"])}
    # Require local SDL3 (or a separately prepared shared SDL3 prefix). The
    # upstream Meson fallback builds a non-PIC SDL3_static, which cannot be
    # linked into the simulator shared library. Do not retry that known error.
    prefix = os.environ.get("TAISEI_CI_SDL_PREFIX")
    pkg_path = os.environ.get("PKG_CONFIG_PATH", "")
    env = os.environ.copy()
    if prefix:
        pkg_path = str(Path(prefix) / "lib/pkgconfig") + (":" + pkg_path if pkg_path else "")
        env["PKG_CONFIG_PATH"] = pkg_path
        env["LD_LIBRARY_PATH"] = str(Path(prefix) / "lib") + (":" + env.get("LD_LIBRARY_PATH", "") if env.get("LD_LIBRARY_PATH") else "")
    try:
        run(["pkg-config", "--modversion", "sdl3"], timeout=15, env=env)
    except (RuntimeError, TimeoutError):
        raise RuntimeError("Shared SDL3 pkg-config package missing; prepare compatible taisei-3.4.12 SDL3 dependency offline before native build")
    build_script = root / "my_test/scripts/build_taisei_sim.sh"
    if not build_script.is_file():
        raise RuntimeError("Canonical native build script missing")
    outlog = logdir / "native_build.log"
    run(["bash", str(build_script)], cwd=root / "my_test", timeout=remaining(deadline), env=env, log=outlog)
    next_state = doctor(root)
    if not next_state["native_ready"]:
        raise RuntimeError("Native build finished without verified pinned simulator library/ABI: " + str(next_state))
    return {"status": "native_compiled_locally", "path": next_state["native_library"],
            "sha256": sha256(next_state["native_library"]), "log": str(outlog)}


def native_e5(root, output, *, max_seconds=3600, deadline=None):
    """Real, bounded, fresh-OS-process E5; requires full repo and native source."""
    root = Path(root).resolve()
    state = doctor(root)
    if not state["native_ready"]:
        raise RuntimeError("Native E5 fail closed: " + state["source"]["reason"] +
                           f"; missing canonical source files={state['missing_canonical_source_files']}")
    archive = archived_evidence(root)
    source = root / "my_test/upstream/taisei-sim-pinned"
    library = Path(state["native_library"])
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output experiment directory already exists: " + str(output))
    output.mkdir(parents=True)
    if deadline is None:
        deadline = time.monotonic() + min(max_seconds, 3600)
    report = {"experiment": "E5_TWO_PROCESS_NATIVE_POLICY_REPLAY_LOCAL_V1",
              "status": "started", "pinned_native_commit": PIN, "native_library_sha256": sha256(library),
              "eval_seeds": E5_SEEDS, "max_frames": 3000, "training_decisions": 0,
              "archive": archive, "checks": [], "visual_inspection_status": "NOT_PERFORMED"}
    atomic_json(output / "manifest.json", report)
    try:
        commands = [
            ("native_smoke", ["my_test/scripts/smoke_native.py", "--library", str(library),
                              "--resource-path", str(source / "resources"), "--stage", "1",
                              "--difficulty", "1", "--seed", "42000", "--frames", "300"]),
            ("native_determinism", ["my_test/scripts/check_native_determinism.py", "--library", str(library),
                                    "--resource-path", str(source / "resources"), "--stage", "1",
                                    "--difficulty", "1", "--seed", "42000", "--frames", "600"]),
        ]
        for name, args in commands:
            report["checks"].append(native_step(root, source, library, args, output, name, deadline))
            atomic_json(output / "manifest.json", report)
        for name, model in [("original_bc", "teacher_bc_seed48030.pt"),
                            ("rebalanced_bc", "teacher_bc_rebalanced_seed48030.pt")]:
            model_path = Path(archive["e4"]) / model
            for pass_name in ("first", "second"):
                summary = output / f"{name}_{pass_name}.jsonl"
                trace = output / f"{name}_{pass_name}.trace.jsonl"
                args = ["my_test/scripts/evaluate_native.py", "--library", str(library),
                        "--resource-path", str(source / "resources"), "--stage", "1",
                        "--difficulty", "1", "--max-frames", "3000", "--policy", "ppo",
                        "--model", str(model_path), "--episodes", "2", "--seed", "42132",
                        "--out", str(summary), "--trace-out", str(trace)]
                report["checks"].append(native_step(root, source, library, args, output,
                                                    f"{name}_{pass_name}", deadline))
                atomic_json(output / "manifest.json", report)
            cmd = [sys.executable, str(root / "research/rsi/e5_audit_replay.py"), "replay",
                   "--first-summary", str(output / f"{name}_first.jsonl"),
                   "--second-summary", str(output / f"{name}_second.jsonl"),
                   "--first-trace", str(output / f"{name}_first.trace.jsonl"),
                   "--second-trace", str(output / f"{name}_second.trace.jsonl"),
                   "--out", str(output / f"{name}_native_replay_verification.json")]
            run(cmd, cwd=root, timeout=remaining(deadline), log=output / f"{name}_verify.log")
            report["checks"].append({"name": name + "_replay_verified", "output": str(output / f"{name}_native_replay_verification.json")})
            atomic_json(output / "manifest.json", report)
        report["status"] = "native_replay_passed_visual_not_verified"
    except Exception as exc:
        report["status"] = "failed_or_incomplete"
        report["error"] = repr(exc)
        atomic_json(output / "manifest.json", report)
        raise
    report["output_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}
    atomic_json(output / "manifest.json", report)
    return report


def replay_preflight(path):
    """Reject a fake/header-only TSR without misclassifying it as actual playback."""
    blob = Path(path).read_bytes()
    result = {"file": str(path), "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
              "playable_verified": False, "visual_inspection_performed": False}
    magic = b"hono\xe2\x9d\xa4umi"
    if len(blob) < 21 or blob[:10] != magic:
        result["status"] = "replay_invalid_header"
        return result
    import struct
    version = struct.unpack_from("<H", blob, 10)[0]
    offset = struct.unpack_from("<I", blob, 17)[0]
    if offset <= 21 or offset >= len(blob):
        result["status"] = "replay_invalid_sections"
    elif blob[21:offset].startswith(bytes.fromhex("28b52ffd2000010000")) or blob[offset:].startswith(bytes.fromhex("28b52ffd2000010000")):
        result["status"] = "replay_empty_payload"
    elif version != 0x800e:
        result["status"] = "replay_version_needs_native_decoder"
    else:
        result["status"] = "replay_has_header_requires_real_playback"
    return result


def visual_video_preflight(video, output):
    """Extract verifiable *genuine* video frames; visual assessment is external."""
    video = Path(video).resolve()
    if not video.is_file() or video.stat().st_size < 1024:
        raise ValueError("No nonempty genuine captured gameplay video")
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        raise RuntimeError("ffprobe/ffmpeg unavailable; cannot inspect actual frames")
    info = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                           "-show_entries", "stream=width,height,nb_frames,duration:format=duration",
                           "-of", "json", str(video)], timeout=30))
    tracks = info.get("streams", [])
    if not tracks or not int(tracks[0].get("width", 0)) or not int(tracks[0].get("height", 0)):
        raise ValueError("Video has no decodable image stream")
    duration = float(info.get("format", {}).get("duration") or tracks[0].get("duration") or 0)
    if duration <= 0:
        raise ValueError("Video lacks positive duration")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to replace existing visual evidence folder")
    output.mkdir(parents=True)
    timestamps = sorted({round(duration * x / 8, 3) for x in range(1, 8)})
    files = []
    for idx, t in enumerate(timestamps):
        dest = output / f"frame_{idx:02d}_{t:.3f}s.png"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(t),
             "-i", str(video), "-frames:v", "1", str(dest)], timeout=30)
        if not dest.is_file() or not dest.stat().st_size:
            raise RuntimeError("Genuine captured video frame extraction failed")
        files.append({"file": dest.name, "timestamp_seconds": t, "sha256": sha256(dest)})
    obj = {"status": "video_valid_frames_extracted_visual_review_pending",
           "source_video_sha256": sha256(video), "source_video_bytes": video.stat().st_size,
           "video_metadata": info, "frame_samples": files,
           "native_policy_identity_verified": False, "genuine_taisei_rendered_origin_verified": False,
           "visual_inspection_performed": False, "gameplay_learning_verified": False,
           "note": "Image samples must be visually inspected and cross-checked against authentic native actions and checkpoint identity"}
    atomic_json(output / "capture_preflight.json", obj)
    return obj


def file_lock(path):
    """Refuse overlapping RSI runs; remove lock only when same-host PID is dead."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "hostname": socket.gethostname(), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    for retry in range(2):
        try:
            with path.open("x") as f:
                f.write(json.dumps(payload)); f.flush(); os.fsync(f.fileno())
            return
        except FileExistsError:
            try:
                recorded = json.loads(path.read_text())
                if recorded.get("hostname") == socket.gethostname():
                    pid = int(recorded.get("pid", -1))
                    if pid > 0:
                        try: os.kill(pid, 0)
                        except ProcessLookupError:
                            path.unlink()
                            continue
                        except PermissionError: pass
            except (ValueError, OSError, json.JSONDecodeError): pass
            raise RuntimeError("Another RSI run may be active; not risking overlapping experiments")
    raise RuntimeError("Could not acquire local RSI lock")


def execute(root, out, *, attempt_native=False):
    """Cheap offline check every invocation; native E5 only when explicitly enabled."""
    root = Path(root).resolve()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Experiment evidence directory exists; refusing overwrite: " + str(out))
    lock = out.parent / ".taisei_rsi_single_run.lock"
    file_lock(lock)
    out.mkdir(parents=True, exist_ok=True)
    events = out / "events.jsonl"
    report = {"run_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "execution_backend": "local", "github_actions_launched": False,
              "source": None, "offline_report": None, "native_report": None,
              "visual_inspection_performed": False, "new_gameplay_improvement_verified": False}
    try:
        log_event(events, "start", "running")
        report["doctor"] = doctor(root)
        log_event(events, "source_preflight", report["doctor"]["source"]["status"],
                  reason=report["doctor"]["source"]["reason"])
        try:
            report["offline_report"] = paired_comparison(root)
            log_event(events, "archived_native_comparison", "passed")
        except (FileNotFoundError, ValueError, KeyError) as e:
            report["offline_error"] = str(e)
            log_event(events, "archived_native_comparison", "blocked", reason=str(e))
        if not attempt_native:
            report["status"] = "offline_validation_only"
        elif report["offline_report"] is None:
            report["status"] = "blocked_missing_provenance"
        elif report["doctor"]["source"]["status"] != "verified":
            report["status"] = "blocked_native_source_or_build"
            report["blocker"] = report["doctor"]["source"]["reason"]
        else:
            registry = root / "results/local_operator/E5_NATIVE_VERIFIED_REGISTRY.json"
            experiment_identity = hashlib.sha256((PIN + BC_SHA + E4_SHA + ",".join(map(str,E5_SEEDS))).encode()).hexdigest()
            registered = None
            if registry.is_file():
                try: registered = json.loads(registry.read_text())
                except (ValueError, OSError): pass
            if registered and registered.get("experiment_identity") == experiment_identity and registered.get("status") == "native_replay_passed_visual_not_verified":
                # A prior run is not a reason to spend more evaluation decisions.
                report["status"] = "native_e5_already_verified_visual_capture_pending"
                report["prior_native_replay_report"] = registered
                log_event(events, "native_e5", "idempotent_skip", reason="Verified same model/game/seed experiment already recorded")
            else:
                deadline = time.monotonic() + 3600
                report["native_build"] = native_build_if_needed(root, deadline, out)
                log_event(events, "native_build", report["native_build"]["status"])
                target = out / "e5_native"
                report["native_report"] = native_e5(root, target, deadline=deadline)
                report["status"] = "native_e5_verified_visual_capture_pending"
                atomic_json(registry, {"experiment_identity": experiment_identity,
                           "status": report["native_report"]["status"],
                           "verified_manifest": str(target / "manifest.json"),
                           "native_library_sha256": report["native_report"]["native_library_sha256"],
                           "note": "Native action/state replay only; rendered visual verification still pending"})
        log_event(events, "end", report["status"])
        return report
    except Exception as e:
        report["status"] = "failed"
        report["error"] = repr(e)
        log_event(events, "end", "failed", reason=repr(e))
        return report
    finally:
        report["journal_sha256"] = sha256(events)
        atomic_json(out / "report.json", report)
        try:
            if json.loads(lock.read_text()).get("pid") == os.getpid():lock.unlink()
        except (OSError, ValueError, json.JSONDecodeError):pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["doctor", "audit", "run", "replay-preflight", "video-preflight"])
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out", type=Path)
    p.add_argument("--attempt-native", action="store_true", help="Explicitly allow genuine bounded native E5, never training")
    p.add_argument("--input", type=Path, help="Genuine game TSR or video, never synthetic placeholder")
    a = p.parse_args(argv)
    if a.operation in ("replay-preflight", "video-preflight") and not a.input:
        p.error("--input requires an actual replay/video path")
    if a.operation == "video-preflight" and not a.out:
        p.error("--out required for extracted genuine frames")
    try:
        if a.operation == "doctor": report = doctor(a.root)
        elif a.operation == "audit": report = paired_comparison(a.root)
        elif a.operation == "run":
            dest = a.out or (a.root / "results" / "local_operator" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            if dest.exists(): raise FileExistsError("Run output exists: " + str(dest))
            report = execute(a.root, dest, attempt_native=a.attempt_native)
        elif a.operation == "replay-preflight": report = replay_preflight(a.input)
        else: report = visual_video_preflight(a.input, a.out)
        print(json.dumps(report, indent=2, sort_keys=True))
        if a.operation in ("doctor", "run", "replay-preflight"):
            allowed = {"doctor": bool(report.get("native_ready")),
                       "run": report.get("status") in {"offline_validation_only", "native_e5_verified_visual_capture_pending", "native_e5_already_verified_visual_capture_pending"},
                       "replay-preflight": report.get("status") == "replay_has_header_requires_real_playback"}
            return 0 if allowed[a.operation] else 2
        return 0
    except Exception as e:
        print(json.dumps({"status": "blocked", "operation": a.operation, "reason": str(e),
                          "native_game_executed": False}, indent=2), file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
