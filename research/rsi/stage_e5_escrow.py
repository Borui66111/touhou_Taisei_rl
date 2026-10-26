#!/usr/bin/env python3
"""Verify and stage permanently escrowed E3/E4 evidence for native E5 replay.

Source-only operation: no downloads, training, Taisei execution or gameplay claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from escrow_native_artifacts import (
    ARTIFACTS, BRANCH, EXPECTED, EXPECTED_COUNTS, LOCKED_COMMIT,
)

SCHEMA = "taisei_rsi_escrow_v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_escrow(root: Path) -> dict:
    """Validate the 33 original files, complete manifest and pinned identities."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Missing or unsafe escrow directory")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Missing or unsafe escrow manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or manifest.get("branch") != BRANCH
            or manifest.get("pinned_native_commit") != LOCKED_COMMIT
            or manifest.get("native_game_executed") is not False
            or manifest.get("source_checkout_recovered") is not False
            or manifest.get("gameplay_improvement_claimed") is not False):
        raise ValueError("Escrow provenance mismatch")
    if set(manifest.get("artifacts", {})) != set(ARTIFACTS):
        raise ValueError("Escrow experiment set mismatch")
    if {p.name for p in root.iterdir()} != {"manifest.json", *ARTIFACTS}:
        raise ValueError("Unexpected escrow root contents")
    total_bytes = 0
    for tag, pinned in ARTIFACTS.items():
        folder = root / tag
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"Missing or unsafe {tag} escrow directory")
        artifact = manifest["artifacts"][tag]
        if any(artifact.get(k) != v for k, v in pinned.items()):
            raise ValueError(f"{tag} run/artifact identity mismatch")
        expected = artifact.get("files")
        if not isinstance(expected, dict) or len(expected) != EXPECTED_COUNTS[tag]:
            raise ValueError(f"{tag} manifest file count mismatch")
        if not set(EXPECTED[tag]).issubset(expected):
            raise ValueError(f"{tag} missing critical E5 evidence")
        if {p.name for p in folder.iterdir()} != set(expected):
            raise ValueError(f"{tag} escrow file set mismatch")
        for name, record in expected.items():
            if not isinstance(name, str) or not BASENAME.fullmatch(name) or name in {".", ".."}:
                raise ValueError("Unsafe escrow filename")
            path = folder / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Unsafe or missing escrow file: {tag}/{name}")
            if (not isinstance(record, dict)
                    or type(record.get("bytes")) is not int or record["bytes"] <= 0):
                raise ValueError(f"Invalid escrow file size: {tag}/{name}")
            expected_hash = record.get("sha256")
            if not isinstance(expected_hash, str) or not HEX64.fullmatch(expected_hash):
                raise ValueError(f"Invalid escrow digest: {tag}/{name}")
            if name in EXPECTED[tag] and expected_hash != EXPECTED[tag][name]:
                raise ValueError(f"Critical E5 identity mismatch: {tag}/{name}")
            if path.stat().st_size != record["bytes"] or sha256(path) != expected_hash:
                raise ValueError(f"Escrow file SHA256/size mismatch: {tag}/{name}")
            total_bytes += record["bytes"]
    return {
        "schema": "taisei_rsi_e5_escrow_stage_v1",
        "source": "permanent_git_escrow",
        "branch": BRANCH,
        "pinned_native_commit": LOCKED_COMMIT,
        "source_run_ids": {tag: info["run_id"] for tag, info in ARTIFACTS.items()},
        "verified_files": sum(EXPECTED_COUNTS.values()),
        "verified_bytes": total_bytes,
        "source_checkout_recovered": False,
        "native_game_executed": False,
        "native_replay_verified": False,
        "gameplay_improvement_claimed": False,
    }


def stage(root: Path, results: Path) -> dict:
    """Verify before copying; reject preexisting targets and copy corruption."""
    report = verify_escrow(root)
    if results.is_symlink():
        raise ValueError("Unsafe results directory")
    for tag in ARTIFACTS:
        target = results / f"{tag}_artifact"
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite E5 evidence: {target}")
    results.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".e5-escrow-", dir=results))
    created = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for tag in ARTIFACTS:
            source = root / tag
            dest = temporary / tag
            dest.mkdir()
            for name, info in manifest["artifacts"][tag]["files"].items():
                shutil.copyfile(source / name, dest / name, follow_symlinks=False)
                if sha256(dest / name) != info["sha256"]:
                    raise ValueError(f"Copied E5 evidence mismatch: {tag}/{name}")
        for tag in ARTIFACTS:
            target = results / f"{tag}_artifact"
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Concurrent E5 stage target exists: {target}")
            os.rename(temporary / tag, target)
            created.append(target)
    except BaseException:
        for target in created:
            shutil.rmtree(target)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--escrow", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and (args.report.exists() or args.report.is_symlink()):
        raise FileExistsError("Refusing to overwrite stage report")
    result = stage(args.escrow, args.results_root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
