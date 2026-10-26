#!/usr/bin/env python3
"""Fail-fast, read-only preflight for Taisei rendered .tsr recordings.

This detects header-only or obviously empty compressed recordings without
pretending to decode game events or identify an RL policy. Successful preflight
means "requires actual playback/decoding", NOT proven gameplay.
Requires only Python 3.11+ stdlib; suitable for Windows local workers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct

MAGIC = b"hono\xe2\x9d\xa4umi"
HEADER_BYTES = 21
ZSTD_EMPTY_FRAME = bytes.fromhex("28b52ffd2000010000")


def inspect(path: str | Path) -> dict:
    path = Path(path)
    result = {"file": str(path), "status": None, "playable_verified": False,
              "policy_checkpoint_verified": False, "gameplay_events_verified": False}
    if not path.is_file():
        return {**result, "status": "missing_file"}
    data = path.read_bytes()
    result.update({"file_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if len(data) < HEADER_BYTES:
        return {**result, "status": "truncated_before_replay_header"}
    if data[:10] != MAGIC:
        return {**result, "status": "invalid_taisei_magic"}
    version_raw = struct.unpack_from("<H", data, 10)[0]
    version = version_raw & 0x7fff
    compressed = bool(version_raw & 0x8000)
    game_version = tuple(data[12:17])
    result.update({"replay_struct_version": version,
                   "compressed": compressed,
                   "game_version_raw": list(game_version)})
    if not compressed:
        return {**result, "status": "non_zstd_variant_requires_native_decoder",
                "reason": "This gate analyzes Taisei compressed replay v14 only."}
    if version != 14:
        return {**result, "status": "other_version_requires_native_decoder"}
    events_start = struct.unpack_from("<I", data, 17)[0]
    result["events_offset"] = events_start
    if events_start <= HEADER_BYTES or events_start >= len(data):
        return {**result, "status": "invalid_or_missing_compressed_sections"}
    metadata = data[HEADER_BYTES:events_start]
    events = data[events_start:]
    result.update({"metadata_bytes": len(metadata), "events_bytes": len(events)})
    if metadata.startswith(ZSTD_EMPTY_FRAME):
        return {**result, "status": "invalid_empty_metadata",
                "reason": "Replay metadata starts with a valid zero-content zstd frame. A usable Taisei replay requires player metadata and at least one stage."}
    if events.startswith(ZSTD_EMPTY_FRAME):
        return {**result, "status": "invalid_empty_events",
                "reason": "Replay events start with a valid zero-content zstd frame; gameplay inputs are missing."}
    return {**result, "status": "header_plausible_full_decode_required",
            "reason": "Preflight alone cannot prove replay integrity, stage clear, game behavior or which RL checkpoint controlled it. Verify in Taisei."}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", help="Path to rendered Taisei .tsr file")
    args = p.parse_args(argv)
    result = inspect(args.replay)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "header_plausible_full_decode_required" else 2


if __name__ == "__main__":
    raise SystemExit(main())
