#!/usr/bin/env python3
"""Read-only preflight for a rendered Taisei policy file.

The original TPPOv1 format is documented in export_live_ppo.py.
TSKILL2 is reported by the local Windows worker; without a published full
TSKILL2 binary schema, this tool recognizes its family but DOES NOT validate
its complete layout or prove successful inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

LEGACY_MAGIC = b"TPPOv1\x00\x00"
SKILL_V2_MAGIC = b"TSKILL2"
LEGACY_HEADER = struct.Struct("<8sIII32s")
INPUT_DIM, WIDTH, ACTIONS = 528, 128, 18
LEGACY_BYTES = LEGACY_HEADER.size + 4 * (
    WIDTH * INPUT_DIM + WIDTH + WIDTH * WIDTH + WIDTH + ACTIONS * WIDTH + ACTIONS
)


def inspect(path: str | Path, demo: str, expected_sha256: str | None = None) -> dict:
    path = Path(path)
    result = {"path": str(path), "demo": demo, "recognized_format": None,
              "compatible_family": False, "fully_validated": False,
              "launch_recommendation": "do_not_launch"}
    if not path.is_file():
        return {**result, "status": "file_missing"}
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    result.update({"size_bytes": len(blob), "sha256": digest})
    if expected_sha256 is not None and expected_sha256.lower() != digest:
        return {**result, "status": "sha256_mismatch"}
    if blob.startswith(LEGACY_MAGIC):
        result["recognized_format"] = "TPPOv1"
        if len(blob) < LEGACY_HEADER.size:
            return {**result, "status": "truncated_tppo_header"}
        _, n_in, width, actions, original_sha = LEGACY_HEADER.unpack_from(blob)
        result.update({"input_dim": n_in, "width": width, "actions": actions,
                       "source_checkpoint_sha256": original_sha.hex()})
        if (n_in, width, actions) != (INPUT_DIM, WIDTH, ACTIONS) or len(blob) != LEGACY_BYTES:
            return {**result, "status": "invalid_tppo_shape_or_size",
                    "expected_size_bytes": LEGACY_BYTES}
        if demo != "legacy-movement":
            return {**result, "status": "incompatible_demo_expects_TSKILL2",
                    "reason": "528-input movement-only TPPOv1 cannot run as 546-input skill-v2."}
        return {**result, "status": "legacy_format_valid",
                "compatible_family": True, "fully_validated": True,
                "launch_recommendation": "check_rendered_loader_SHA256_allowlist_before_launch",
                "reason": "The legacy renderer may hard-code a binary SHA-256. This test does not verify that allowlist."}
    if blob.startswith(SKILL_V2_MAGIC):
        result["recognized_format"] = "TSKILL2"
        if demo != "skill-v2":
            return {**result, "status": "incompatible_demo_expects_TPPOv1",
                    "reason": "TSKILL2 skill-head binary cannot run as TPPOv1 movement-only."}
        return {**result, "status": "skill_v2_family_recognized_schema_unverified",
                "compatible_family": True,
                "reason": "Full skill-v2 layout, checkpoint SHA policy and model integration must be confirmed on laptop.",
                "launch_recommendation": "verify_local_demo_binary_schema_and_hash_before_launch"}
    return {**result, "status": "unknown_binary_format",
            "first_8_bytes_hex": blob[:8].hex()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("policy", help="Path to the exported .bin, not a PyTorch .pt")
    p.add_argument("--demo", choices=["skill-v2", "legacy-movement"], default="skill-v2")
    p.add_argument("--expected-sha256", help="Optional expected SHA-256 for the exported binary")
    a = p.parse_args(argv)
    result = inspect(a.policy, a.demo, a.expected_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["compatible_family"]:
        return 2
    if not result["fully_validated"]:
        return 3  # recognized family, but cannot safely assert loader acceptance
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
