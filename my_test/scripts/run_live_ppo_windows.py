#!/usr/bin/env python3
"""Launch the local Windows live-PPO build with fresh runtime paths."""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from taisei_resource_doctor import select_resources


ROOT = Path(__file__).resolve().parents[1]
DRY_RUN_MARKER = "TAISEI_LIVE_PPO_DRY_RUN_OK"
PATHS = {
    "python": ROOT / ".venv-winbuild" / "Scripts" / "python.exe",
    "build_dir": ROOT / "rl_skill_v2",
    "game": ROOT / "rl_skill_v2" / "src" / "taisei.exe",
    "resources": ROOT / "windows-build" / "wlive_agent_demo" / "resources",
    "locale": ROOT / "rl_skill_v2" / "po",
    "toolchain_bin": ROOT / "toolchains" / "llvm-mingw-20261006" / "llvm-mingw-20261006-ucrt-x86_64" / "bin",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_metadata(path):
    try:
        with Path(path).open("rb") as stream:
            header = stream.read(48)
            if len(header) != 48 or header[:8] != b"TSKILL2\0":
                return {"status": "unknown", "reason": "invalid_or_unknown_format"}
            meta_len, payload_len = struct.unpack("<II", header[8:16])
            metadata_bytes = stream.read(meta_len)
            if len(metadata_bytes) != meta_len or hashlib.sha256(metadata_bytes).digest() != header[16:48]:
                return {"status": "unknown", "reason": "invalid_metadata_checksum"}
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            payload = stream.read(payload_len)
            if len(payload) != payload_len or stream.read(1):
                return {"status": "unknown", "reason": "invalid_payload_length"}
            if metadata.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
                return {"status": "unknown", "reason": "invalid_payload_checksum"}
            checkpoint_sha = metadata.get("checkpoint_sha256")
            if not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint_sha)):
                return {"status": "unknown", "reason": "invalid_checkpoint_sha256"}
            return {"status": "valid", "checkpoint_sha256": checkpoint_sha,
                    "mask_version": metadata.get("skill_action_mask_version")}
    except (OSError, ValueError, UnicodeError, struct.error):
        return {"status": "unknown", "reason": "metadata_unreadable_or_invalid"}


def _snapshot_log(path):
    path = Path(path)
    try:
        stat = path.stat()
        if not path.is_file():
            raise OSError("not a regular log file")
        return {"path": path, "identity": (stat.st_dev, stat.st_ino), "size": stat.st_size}
    except OSError:
        return {"path": path, "identity": None, "size": None}


def _read_appended_log(snapshot):
    path = snapshot["path"]
    try:
        stat = path.stat()
        if snapshot["identity"] is None or (stat.st_dev, stat.st_ino) != snapshot["identity"] or stat.st_size < snapshot["size"]:
            return {"status": "unknown", "reason": "rotated_truncated_or_missing", "lines": [], "bytes_scanned": 0}
        lines = []
        bytes_scanned = 0
        with path.open("rb") as stream:
            stream.seek(snapshot["size"])
            for raw_line in stream:
                bytes_scanned += len(raw_line)
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if _LOG_RELEVANT_RE.search(line):
                    lines.append(line)
        after = path.stat()
        if ((after.st_dev, after.st_ino) != snapshot["identity"] or after.st_size != snapshot["size"] + bytes_scanned):
            return {"status": "unknown", "reason": "rotated_truncated_or_missing", "lines": [], "bytes_scanned": bytes_scanned}
        return {"status": "captured", "lines": lines, "bytes_scanned": bytes_scanned}
    except OSError as exc:
        return {"status": "unknown", "reason": type(exc).__name__, "lines": [], "bytes_scanned": 0}


_LOG_ENVELOPE_RE = re.compile(r"^\d+\s+[A-Z]+\s+\S+\s+[^:]+:\s+(.*?)\s+\[[^]]+\]$")
_LOG_RELEVANT_RE = re.compile(r"\b(?:RL skill-v2 policy loaded |RL run |RL decision |RL policy outcome=|Saving |RSI_REPLAY_WRITE_(?:BEGIN|END|INVALID)|ZSTD_compressStream2\(\) failed|SDL_WriteIO\(\) returned|VFS error:|Too many events in replay|No stages in replay|No events in stage)", re.IGNORECASE)


def _parse_log_lines(lines, expected_checkpoint_sha, appdata_taisei=None):
    result = {"loaded": None, "logged_checkpoint_sha256": None, "metadata_checkpoint_sha256": expected_checkpoint_sha,
              "attribution": "unknown", "outcome": None, "replay": None,
              "events": {"policy_loaded": False, "run": None, "run_records": [], "decisions": [],
                         "policy_loaded_count": 0, "run_count": 0, "decision_count": 0,
                         "outcome_count": 0, "replay_save_count": 0}}
    replay_root = Path(appdata_taisei).resolve() if appdata_taisei is not None else None
    observed_shas = []
    sequence = []
    for index, raw_line in enumerate(lines):
        envelope = _LOG_ENVELOPE_RE.match(raw_line)
        message = envelope.group(1) if envelope else raw_line
        if message.startswith("RL skill-v2 policy loaded "):
            result["events"]["policy_loaded"] = True
            result["events"]["policy_loaded_count"] += 1
            sequence.append((index, "policy_loaded"))
            fields = dict(re.findall(r"([a-z0-9_]+)=([^ ]+)", message))
            sha = fields.get("checkpoint_sha256")
            if sha:
                result["events"]["loaded_event_checkpoint_sha256"] = sha
                observed_shas.append(sha)
        elif message.startswith("RL run "):
            fields = dict(re.findall(r"([a-z0-9_]+)=([^ ]+)", message))
            result["events"]["run_count"] += 1
            result["events"]["run_records"].append(fields)
            sequence.append((index, "run"))
            if fields.get("policy_type") == "skill_v2" and fields.get("loaded") in {"0", "1"}:
                result["loaded"] = fields["loaded"] == "1"
                result["logged_checkpoint_sha256"] = fields.get("checkpoint_sha256")
                result["events"]["run"] = fields
                if fields.get("loaded") == "1":
                    observed_shas.append(fields.get("checkpoint_sha256", ""))
        elif message.startswith("RL decision "):
            fields = dict(re.findall(r"([a-z0-9_]+)=([^ ]+)", message))
            if fields:
                result["events"]["decisions"].append(fields)
                result["events"]["decision_count"] += 1
                sequence.append((index, "decision"))
                observed_shas.append(fields.get("checkpoint_sha256", ""))
        elif message.startswith("RL policy outcome="):
            result["events"]["outcome_count"] += 1
            sequence.append((index, "outcome"))
            result["outcome"] = dict(re.findall(r"([a-z0-9_]+)=([^ ]+)", message))
            observed_shas.append(result["outcome"].get("checkpoint_sha256", ""))
        elif message.startswith("Saving ") and replay_root is not None:
            result["events"]["replay_save_count"] += 1
            sequence.append((index, "replay_save"))
            candidate = Path(message[len("Saving "):].strip())
            if not candidate.is_absolute():
                candidate = replay_root / candidate
            try:
                resolved = candidate.resolve(strict=True)
                relative = resolved.relative_to(replay_root)
                if resolved.is_file():
                    result["replay"] = {"path": str(resolved), "relative_path": relative.as_posix(),
                                        "size": resolved.stat().st_size, "sha256": _sha256(resolved)}
            except (OSError, ValueError):
                pass
    sha_values = [sha for sha in observed_shas if sha]
    sha_mismatch = (len(set(sha_values)) > 1
                    or (expected_checkpoint_sha is not None and any(sha != expected_checkpoint_sha for sha in sha_values)))
    decisions_complete = bool(result["events"]["decisions"]) and all(
        {"frame", "movement", "skill", "accepted", "action_count", "checkpoint_sha256"}.issubset(decision)
        and decision.get("checkpoint_sha256") == expected_checkpoint_sha
        for decision in result["events"]["decisions"])
    if sha_mismatch:
        result["attribution"] = "mismatch"
    event_types = [kind for _, kind in sequence]
    run_position = event_types.index("run") if event_types.count("run") == 1 else -1
    load_position = event_types.index("policy_loaded") if event_types.count("policy_loaded") == 1 else -1
    outcome_position = event_types.index("outcome") if event_types.count("outcome") == 1 else -1
    save_position = event_types.index("replay_save") if event_types.count("replay_save") == 1 else -1
    sequence_matches = (result["events"]["policy_loaded_count"] == 1 and result["events"]["run_count"] == 1
                        and result["events"]["decision_count"] > 0 and result["events"]["outcome_count"] == 1
                        and result["events"]["replay_save_count"] == 1
                        and 0 <= load_position < run_position < outcome_position < save_position
                        and all(run_position < pos < outcome_position for pos, kind in sequence if kind == "decision"))
    if (not sha_mismatch and expected_checkpoint_sha is not None and sequence_matches and result["loaded"] is True
          and result["events"]["policy_loaded"] and result["logged_checkpoint_sha256"] == expected_checkpoint_sha
          and result["events"].get("loaded_event_checkpoint_sha256") == expected_checkpoint_sha
          and decisions_complete and result["outcome"] is not None
          and result["outcome"].get("checkpoint_sha256") == expected_checkpoint_sha and result["replay"] is not None):
        result["attribution"] = "matched"
    return result


def _other_taisei_active():
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq taisei.exe", "/FO", "CSV", "/NH"],
                                capture_output=True, text=True, check=True)
        return any("taisei.exe" in line.lower() for line in result.stdout.splitlines())
    except (OSError, subprocess.SubprocessError):
        return True


def _capture_conflict():
    return _other_taisei_active()


def _write_manifest_exclusive(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _finalize_manifest(path, manifest, exit_status):
    final = dict(manifest)
    final["finished_at"] = datetime.now(timezone.utc).isoformat()
    final["exit_status"] = exit_status
    final["status"] = ("partial" if manifest.get("status") == "partial" else
                       "complete" if exit_status == 0 else "failed")
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(final, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return final


def _capture_paths():
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return appdata / "taisei" / "log.txt", appdata / "taisei"


def _create_run_manifest(plan):
    scratch = Path.home() / "AppData" / "Local" / "hermes" / "cache" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    manifest_path = scratch / f"taisei-live-ppo-{run_id}.json"
    paths = plan["verified_paths"]
    game, policy = Path(paths["game"]), Path(paths["policy"])
    log_path, _ = _capture_paths()
    snapshot = _snapshot_log(log_path)
    manifest = {"run_id": run_id, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "argv": plan["meson_argv"], "game_argv": plan["game_argv"], "paths": paths,
                "sha256": {"game": _sha256(game), "policy": _sha256(policy)},
                "policy_metadata": _policy_metadata(policy), "resource_preflight": plan["resource_preflight"], "log_capture": {"path": str(log_path), "status": "pending",
                    "initial_identity": snapshot["identity"], "initial_size": snapshot["size"]}}
    _write_manifest_exclusive(manifest_path, manifest)
    return manifest_path, manifest, snapshot


def _child_wait_code(game_argv):
    return "import subprocess,sys;sys.exit(subprocess.run(" + repr(game_argv) + ",check=False).returncode)"


def _child_code(paths, renderer, dry_run=False):
    resources = paths["resources"].as_posix()
    locale = paths["locale"].as_posix()
    toolchain = paths["toolchain_bin"].as_posix()
    game = paths["game"].as_posix()
    policy = paths["policy"].as_posix()
    game_argv = [game, "--renderer", renderer, "--rl-policy", policy]
    runtime_setup = (
        "import os,sys;"
        f"os.environ['TAISEI_RES_PATH']={resources!r};"
        f"os.environ['TAISEI_LOCALE_PATH']={locale!r};"
        f"os.environ['PATH']={toolchain!r}+os.pathsep+os.environ.get('PATH','');"
    )
    if dry_run:
        return (
            runtime_setup
            + f"assert os.environ['TAISEI_RES_PATH']=={resources!r};"
            + f"assert os.environ['TAISEI_LOCALE_PATH']=={locale!r};"
            + f"assert os.environ['PATH'].split(os.pathsep)[0]=={toolchain!r};"
            + f"game_argv={game_argv!r};"
            + f"assert game_argv=={game_argv!r};"
            + f"print({DRY_RUN_MARKER!r},flush=True)"
        )
    return runtime_setup + _child_wait_code(game_argv)


def _build_plan(policy, renderer="gl33", dry_run=False, resources=None):
    policy_path = Path(policy)
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    # The raw launcher is an internal dependency of W2CTRL, not a validation
    # bypass. Use its canonical parser before preparing any subprocess plan.
    from w2ctrl import parse_policy as parse_canonical_policy
    parse_canonical_policy(policy_path)
    override = None
    if resources is not None:
        override = Path(resources)
        if not override.is_absolute():
            override = ROOT / override
    resource_root, resource_preflight = select_resources(
        ROOT, PATHS["resources"], override)
    required_paths = {**PATHS, "resources": resource_root, "policy": policy_path}
    missing = []
    for name, path in required_paths.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
        elif name in {"build_dir", "resources", "locale", "toolchain_bin"} and not path.is_dir():
            missing.append(f"{name} is not a directory: {path}")
        elif name in {"python", "game", "policy"} and not path.is_file():
            missing.append(f"{name} is not a file: {path}")
    if missing:
        raise FileNotFoundError("Required live-PPO path(s) missing or invalid:\n" + "\n".join(missing))

    paths = {name: path.resolve() for name, path in required_paths.items()}
    game = str(paths["game"])
    policy = str(paths["policy"])
    game_argv = [game, "--renderer", renderer, "--rl-policy", policy]
    child_code = _child_code(paths, renderer, dry_run=dry_run)
    return {
        "dry_run": dry_run,
        "resource_preflight": resource_preflight,
        "verified_paths": {name: str(path) for name, path in paths.items()},
        "game_argv": game_argv,
        "meson_argv": [
            str(paths["python"]),
            "-m",
            "mesonbuild.mesonmain",
            "devenv",
            "-C",
            str(paths["build_dir"]),
            "--workdir",
            str(ROOT.resolve()),
            str(paths["python"]),
            "-c",
            child_code,
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renderer",
        default="gl33",
        help="Taisei renderer to use (default: gl33)",
    )
    parser.add_argument(
        "--rl-policy",
        required=True,
        metavar="PATH",
        help="PPO policy file; relative paths are resolved from the my_test directory",
    )
    parser.add_argument(
        "--resources",
        help="optional compiled resource root; checked for portraits.tsatlas.zst",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify runtime paths through Meson without starting Taisei",
    )
    args = parser.parse_args(argv)
    try:
        plan = _build_plan(
            policy=args.rl_policy,
            renderer=args.renderer,
            dry_run=args.dry_run,
            resources=args.resources,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(plan, separators=(",", ":")), flush=True)
        return subprocess.run(plan["meson_argv"], cwd=ROOT, check=False).returncode

    manifest_path = None
    manifest = None
    try:
        manifest_path, manifest, log_snapshot = _create_run_manifest(plan)
        if _capture_conflict():
            manifest["status"] = "partial"
            manifest["log_capture"].update({"path": str(log_snapshot["path"]), "status": "unknown",
                                            "reason": "concurrent_taisei_process"})
            _finalize_manifest(manifest_path, manifest, 1)
            print(f"live-PPO capture refused; manifest: {manifest_path}", file=sys.stderr)
            return 1
        exit_status = subprocess.run(plan["meson_argv"], cwd=ROOT, check=False).returncode
        if _capture_conflict():
            capture = {"status": "unknown", "reason": "concurrent_taisei_process", "lines": [], "bytes_scanned": 0}
        elif log_snapshot["identity"] is None:
            capture = {"status": "unknown", "reason": "log_missing_before_launch", "lines": [], "bytes_scanned": 0}
        else:
            capture = _read_appended_log(log_snapshot)
        log_path, appdata_taisei = _capture_paths()
        manifest["log_capture"].update({"path": str(log_path), "status": capture["status"],
                                        "initial_identity": list(log_snapshot["identity"]) if log_snapshot["identity"] is not None else None,
                                        "initial_size": log_snapshot["size"],
                                        "bytes_scanned": capture.get("bytes_scanned", 0),
                                        **({"reason": capture["reason"]} if capture["status"] == "unknown" else {}),
                                        **({"relevant_lines": capture["lines"]} if capture["status"] == "captured" else {})})
        if capture["status"] == "captured":
            metadata = manifest["policy_metadata"]
            expected_sha = metadata.get("checkpoint_sha256") if metadata.get("status") == "valid" else None
            parsed = _parse_log_lines(capture["lines"], expected_sha, appdata_taisei)
            manifest["log_events"] = parsed
        else:
            manifest["status"] = "partial"
        _finalize_manifest(manifest_path, manifest, exit_status)
        return exit_status
    except Exception as exc:
        if manifest_path is not None and manifest is not None:
            manifest["status"] = "partial"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            try:
                _finalize_manifest(manifest_path, manifest, 1)
            except OSError:
                pass
        print(f"live-PPO launch/provenance failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
