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


HASH_CHUNK_BYTES = 1024 * 1024


def _read_replay(p: Path):
    file_size = p.stat().st_size
    metadata_prefix = b""
    events_prefix = b""

    def read_prefix(stream, offset: int, length: int) -> bytes:
        stream.seek(offset)
        chunks = []
        remaining = length
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    with p.open("rb") as stream:
        opened_size = stream.seek(0, 2)
        header = read_prefix(stream, 0, min(HEADER_BYTES, file_size))
        if len(header) >= HEADER_BYTES and header[:10] == MAGIC:
            events_offset = struct.unpack_from("<I", header, 17)[0]
            if HEADER_BYTES < events_offset < file_size:
                metadata_prefix = read_prefix(
                    stream, HEADER_BYTES, min(len(ZSTD_EMPTY_FRAME), events_offset - HEADER_BYTES)
                )
                events_prefix = read_prefix(
                    stream, events_offset, min(len(ZSTD_EMPTY_FRAME), file_size - events_offset)
                )

        digest = hashlib.sha256()
        bytes_hashed = 0
        stream.seek(0)
        remaining = file_size
        while remaining:
            chunk = stream.read(min(HASH_CHUNK_BYTES, remaining))
            if not chunk:
                break
            digest.update(chunk)
            bytes_hashed += len(chunk)
            remaining -= len(chunk)
        extra_byte = bool(stream.read(1)) if bytes_hashed == file_size else False
        final_open_size = stream.seek(0, 2)

    final_path_size = p.stat().st_size
    changed = (opened_size != file_size or bytes_hashed != file_size or extra_byte or
               final_open_size != file_size or final_path_size != file_size)
    return (file_size, header, metadata_prefix, events_prefix,
            None if changed else digest.hexdigest(), changed)


def inspect(path: str | Path) -> dict:
    p = Path(path)
    result = {"file": str(p), "status": None, "playable_verified": False,
              "gameplay_events_verified": False, "visual_inspection": "not_performed",
              "checkpoint_attribution_verified": False}
    if not p.is_file():
        return {**result, "status": "missing_file"}
    file_size, b, metadata_prefix, events_prefix, sha256, changed = _read_replay(p)
    result["file_bytes"] = file_size
    if changed:
        return {**result, "status": "file_changed_during_inspection"}
    result["sha256"] = sha256
    if file_size < HEADER_BYTES or len(b) < HEADER_BYTES:
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
    if not HEADER_BYTES < events_offset < file_size:
        return {**result, "status": "missing_sections"}
    result.update({"metadata_bytes": events_offset - HEADER_BYTES,
                   "events_bytes": file_size - events_offset})
    # Empty Zstd frames alone are not meaningful gameplay, even when tiny
    # trailing framing/CRC bytes are present. Check both section prefixes explicitly.
    metadata_empty = metadata_prefix.startswith(ZSTD_EMPTY_FRAME)
    events_empty = events_prefix.startswith(ZSTD_EMPTY_FRAME)
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
