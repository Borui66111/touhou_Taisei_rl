#!/usr/bin/env python3
"""Read-only gate for Taisei TSR replay structural validity.

Only rejects impossible/empty replay captures; a plausible header does not
establish actual playback, policy attribution, or visible gameplay.
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
    p = Path(path)
    result = {"file": str(p), "status": None, "playable_verified": False,
              "gameplay_events_verified": False, "visual_inspection": "not_performed",
              "checkpoint_attribution_verified": False}
    if not p.is_file():
        return {**result, "status": "missing_file"}
    b = p.read_bytes()
    result.update({"file_bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
    if len(b) < HEADER_BYTES:
        return {**result, "status": "truncated_before_header"}
    if b[:10] != MAGIC:
        return {**result, "status": "invalid_magic"}
    version_raw = struct.unpack_from("<H", b, 10)[0]
    version = version_raw & 0x7fff
    compressed = bool(version_raw & 0x8000)
    result.update({"replay_struct_version": version, "compressed": compressed,
                   "game_version_raw": list(b[12:17])})
    if not compressed or version != 14:
        return {**result, "status": "different_variant_requires_game_decoder"}
    events_offset = struct.unpack_from("<I", b, 17)[0]
    result["events_offset"] = events_offset
    if not HEADER_BYTES < events_offset < len(b):
        return {**result, "status": "missing_sections"}
    metadata = b[HEADER_BYTES:events_offset]
    events = b[events_offset:]
    result.update({"metadata_bytes": len(metadata), "events_bytes": len(events)})
    # Empty Zstd frames alone are not meaningful gameplay, even when tiny
    # trailing framing/CRC bytes are present. Check both sections explicitly.
    metadata_empty = metadata.startswith(ZSTD_EMPTY_FRAME)
    events_empty = events.startswith(ZSTD_EMPTY_FRAME)
    result.update({"metadata_empty_frame_prefix": metadata_empty,
                   "events_empty_frame_prefix": events_empty})
    if metadata_empty and events_empty:
        return {**result, "status": "invalid_empty_metadata_and_events"}
    if metadata_empty:
        return {**result, "status": "invalid_empty_metadata"}
    if events_empty:
        return {**result, "status": "invalid_empty_events"}
    return {**result, "status": "header_plausible_full_decode_required"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args(argv)
    result = inspect(args.replay)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "header_plausible_full_decode_required" else 2

if __name__ == "__main__":
    raise SystemExit(main())
