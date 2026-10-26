"""Offline, fail-closed verification of a recovered authentic Taisei Git checkout.

This does not clone, fetch, modify, compile, or run the game. It certifies
source identity and recursive submodule initialization for a local checkout.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

EXPECTED_URL = "https://github.com/touhourl/taisei-sim.git"
EXPECTED_COMMIT = "789200ff255c681b3ed4230a3f26f0b8c12c78fe"


def verify(lock: Path, checkout: Path, runner=subprocess.run, timeout: int = 20) -> dict:
    result = {
        "schema": "taisei_offline_source_verification_v1",
        "status": "blocked", "source_mode": "preexisting_local_git_checkout",
        "expected_url": EXPECTED_URL, "locked_commit": EXPECTED_COMMIT,
        "locked_commit_verified": False, "recursive_submodules_verified": False,
        "native_game_executed": False, "reason": None,
    }
    try:
        lines = lock.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        result["reason"] = f"lock unreadable: {type(exc).__name__}"
        return result
    if lines != [f"REPO_URL={EXPECTED_URL}", f"COMMIT={EXPECTED_COMMIT}"]:
        result["reason"] = "unexpected lock identity"
        return result
    try:
        root = checkout.resolve(strict=True)
    except OSError:
        result["reason"] = "checkout path unavailable"
        return result
    if not root.is_dir():
        result["reason"] = "checkout path is not a directory"
        return result
    env = dict(os.environ)
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "/bin/false", "GCM_INTERACTIVE": "never"})

    def git(*args):
        cmd = ["git", "-C", str(root), *args]
        cp = runner(cmd, env=env, capture_output=True, text=True,
                    check=False, timeout=timeout)
        if cp.returncode != 0:
            raise ValueError("Git inspection failed: " + " ".join(args))
        return cp.stdout.strip("\n")

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            raise ValueError("checkout is not the root of its Git worktree")
        if git("remote", "get-url", "origin") != EXPECTED_URL:
            raise ValueError("origin URL does not match locked repository")
        if git("rev-parse", "HEAD") != EXPECTED_COMMIT:
            raise ValueError("HEAD does not match locked commit")
        if git("cat-file", "-t", EXPECTED_COMMIT) != "commit":
            raise ValueError("locked object is not a Git commit")
        if git("status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"):
            raise ValueError("checkout or submodules have local modifications")
        if not (root / ".gitmodules").is_file():
            raise ValueError("expected submodule manifest missing")
        status = git("submodule", "status", "--recursive").splitlines()
        if not status:
            raise ValueError("recursive submodules absent")
        for line in status:
            if not re.match(r"^ [0-9a-f]{40} [^\s]+(?: .*)?$", line):
                raise ValueError("recursive submodule missing, mismatched or conflicted")
    except (ValueError, subprocess.TimeoutExpired, OSError) as exc:
        result["reason"] = ("local Git inspection timed out" if isinstance(exc, subprocess.TimeoutExpired)
                            else str(exc))
        return result
    result.update({
        "status": "verified", "locked_commit_verified": True,
        "recursive_submodules_verified": True,
        "submodule_count": len(status),
        "reason": "exact locked commit, origin, clean tree and recursive Git submodules verified; no native gameplay executed",
    })
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", type=Path, default=Path("my_test/UPSTREAM.lock"))
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    report = verify(args.lock, args.checkout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "verified" else 2


if __name__ == "__main__":
    sys.exit(main())
