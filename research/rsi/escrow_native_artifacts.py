#!/usr/bin/env python3
"""Fail-closed, source-only preservation of expiring E3/E4 native evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

BRANCH = "research/rsi-automation-v1"
LOCKED_COMMIT = "789200ff255c681b3ed4230a3f26f0b8c12c78fe"
EXPECTED = {
    "e3": {
        "teacher_demonstrations_12000.npz": "02dd11feb60e0dcb5710ae8af3cf33dca7e7e77756d6467f822ad94d0f5a0d8d",
        "teacher_bc_seed48030.pt": "87b43918d0181355435f105e3fcec44d7fd314c12aec6978b63ec33e9955cf54",
        "teacher_bc_ppo_seed48020.pt": "dd110638e81e2479e13591ad1a5cfb0ab28e535d7b889e25cdefd287f55b17d9",
        "e3_manifest.json": "d62212ae98c4f554dd7f81d8552c12d8fb7a86f311493ca04f490b175e740d0b",
    },
    "e4": {
        "teacher_bc_seed48030.pt": "87b43918d0181355435f105e3fcec44d7fd314c12aec6978b63ec33e9955cf54",
        "teacher_bc_rebalanced_seed48030.pt": "06c344d1c9415d1277e89c7d7c1a4519f274952f159e1cc5095d7fecf696e6fc",
        "teacher_bc_seed48030_native.jsonl": "8e050bc8cf983fe569bcab328adbfabfbcfa8d5f3d06d6355c9ed80a83bbf71e",
        "teacher_bc_rebalanced_seed48030_native.jsonl": "cefd841c784ac14accb39681070d6ec40ce12c52a13afd6f541d42bc247d8663",
        "risk_teacher_native.jsonl": "e66b08af96a341651288256f42d93d90604e904c51921d4364f8b49ffa51f0e5",
    },
}
ARTIFACTS = {
    "e3": {"run_id": 40219788151, "artifact_id": 12609092648,
           "zip_sha256": "ecb660decee9706dd62c8a98023d0fe0979b578c24eb6a037c089f1014b169d1",
           "expires_at": "2026-11-15T18:12:07Z"},
    "e4": {"run_id": 40220975924, "artifact_id": 12609314728,
           "zip_sha256": "1dc6ab9ffe7d5ab88d30c162246d0b2c7fe66c29556fb751b54886b4d2430789",
           "expires_at": "2026-11-15T18:28:40Z"},
}
EXPECTED_COUNTS = {"e3": 17, "e4": 16}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def archive(e3, e4, destination):
    """Verify required identities before copying; never overwrite prior escrow."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("immutable escrow already exists")
    sources = {"e3": e3, "e4": e4}
    report = {"schema": "taisei_rsi_escrow_v1", "branch": BRANCH,
              "pinned_native_commit": LOCKED_COMMIT,
              "native_game_executed": False, "source_checkout_recovered": False,
              "gameplay_improvement_claimed": False, "artifacts": {}}
    for tag, folder in sources.items():
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"{tag} artifact directory missing or unsafe")
        entries = list(folder.iterdir())
        if len(entries) != EXPECTED_COUNTS[tag] or any(not x.is_file() or x.is_symlink() for x in entries):
            raise ValueError(f"{tag} artifact file count/type mismatch")
        names = {p.name for p in entries}
        if not set(EXPECTED[tag]).issubset(names):
            raise ValueError(f"{tag} missing required evidence")
        for name, expected in EXPECTED[tag].items():
            if digest(folder / name) != expected:
                raise ValueError(f"{tag}/{name} SHA256 mismatch")
        report["artifacts"][tag] = dict(ARTIFACTS[tag])
        report["artifacts"][tag]["files"] = {
            p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(entries)
        }
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for tag, folder in sources.items():
            target = destination / tag
            target.mkdir()
            for name, info in report["artifacts"][tag]["files"].items():
                shutil.copyfile(folder / name, target / name)
                if digest(target / name) != info["sha256"]:
                    raise ValueError("copied evidence digest mismatch")
        (destination / "manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(destination)
        raise
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--e3", type=Path, required=True)
    p.add_argument("--e4", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    result = archive(a.e3, a.e4, a.destination)
    print(json.dumps({"status": "verified_copy", "counts": {k: len(v["files"])
        for k, v in result["artifacts"].items()}, "native_game_executed": False}))
