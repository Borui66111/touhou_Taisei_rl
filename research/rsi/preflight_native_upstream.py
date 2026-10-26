"""Cheap fail-closed access check for the *pinned* authentic Taisei upstream.

This only verifies that a CI runner can read the remote's advertised HEAD.
It does NOT establish that the locked commit or its submodules are available;
my_test/scripts/build_taisei_sim.sh remains the authority for source identity.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

EXPECTED_URL = "https://github.com/touhourl/taisei-sim.git"
EXPECTED_COMMIT = "789200ff255c681b3ed4230a3f26f0b8c12c78fe"


def check_upstream(lock: Path, timeout: int = 20, runner=subprocess.run) -> dict:
    """Return a structured access gate result; never mutate the source checkout."""
    result = {
        "schema": "taisei_upstream_access_preflight_v1",
        "status": "blocked",
        "remote_url": EXPECTED_URL,
        "locked_commit": EXPECTED_COMMIT,
        "locked_commit_verified": False,
        "native_game_executed": False,
        "reason": None,
    }
    try:
        lines = lock.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        result["reason"] = f"cannot read pinned lock: {type(exc).__name__}"
        return result
    if lines != [f"REPO_URL={EXPECTED_URL}", f"COMMIT={EXPECTED_COMMIT}"]:
        result["reason"] = "unexpected upstream lock identity; refusing remote check"
        return result
    env = dict(os.environ)
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "/bin/false", "GCM_INTERACTIVE": "never"})
    cmd = ["git", "-c", "credential.helper=", "ls-remote", "--exit-code", EXPECTED_URL, "HEAD"]
    try:
        completed = runner(cmd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        result["reason"] = f"remote access timed out after {timeout}s"
        return result
    except OSError as exc:
        result["reason"] = f"git unavailable: {type(exc).__name__}"
        return result
    if completed.returncode != 0:
        result["reason"] = f"git ls-remote failed (exit {completed.returncode}); remote inaccessible or authentication required"
        return result
    records = [line.split("\t", 1) for line in completed.stdout.splitlines()]
    heads = [sha for sha, ref in records if ref == "HEAD" and len(sha) == 40
             and all(c in "0123456789abcdef" for c in sha)]
    if len(heads) != 1:
        result["reason"] = "remote did not advertise one valid HEAD"
        return result
    result.update({"status": "reachable", "remote_head": heads[0],
                   "reason": "advertised HEAD readable; locked commit and submodules NOT verified"})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("my_test/UPSTREAM.lock"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 45:
        parser.error("timeout must be 1..45 seconds")
    report = check_upstream(args.lock, timeout=args.timeout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "reachable" else 2


if __name__ == "__main__":
    sys.exit(main())
