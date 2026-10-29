#!/usr/bin/env python3
"""Read-only Taisei compressed-v14 .tsr structural and event forensic audit.

Uses the real Taisei replay binary layout (src/replay/{struct.h,read.c,write.c})
without loading a game. A successful structural check DOES NOT establish that
Taisei can play a replay or that a neural policy controlled the original run.
The optional `zstandard` Python package or the `zstd` CLI enables full decoding.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

MAGIC = b"hono\xe2\x9d\xa4umi"
HEADER_LEN = 21
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
EMPTY_ZSTD_FRAME = bytes.fromhex("28b52ffd2000010000")
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_SECTION_BYTES = 16 * 1024 * 1024
ERROR_SIGNALS = (
    "saving ", "replay", "zstd", "sdl_write", "sdl_close", "io_status",
    "vfs error", "write failed", "error reading", "no stages", "no events",
    "rl policy outcome", "rl run ", "rl decision ", "policy loaded",
)

class ReplayError(ValueError):
    pass

class Cursor:
    def __init__(self, data: bytes):
        self.data = data
        self.i = 0

    def read(self, fmt: str):
        n = struct.calcsize(fmt)
        if self.i + n > len(self.data):
            raise ReplayError(f"truncated decoded metadata at byte {self.i}, need {n}")
        value, = struct.unpack_from(fmt, self.data, self.i)
        self.i += n
        return value

    def skip(self, n: int):
        if self.i + n > len(self.data):
            raise ReplayError("truncated decoded metadata")
        self.i += n


def parse_metadata_v14(meta: bytes):
    c = Cursor(meta)
    name_bytes = c.read("<B")
    c.skip(name_bytes)
    flags = c.read("<I")
    nstages = c.read("<H")
    if not 1 <= nstages <= 16:
        raise ReplayError(f"replay stages count invalid: {nstages}")
    stages=[]
    for _ in range(nstages):
        stage_flags=c.read("<I")
        stage_id=c.read("<H")
        start_time=c.read("<Q")
        seed=c.read("<Q")
        difficulty=c.read("<B")
        skip_frames=c.read("<H")
        start_score=c.read("<Q")
        total_lives_used=c.read("<B")
        total_bombs_used=c.read("<B")
        total_continues=c.read("<B")
        character=c.read("<B")
        shot=c.read("<B")
        x=c.read("<H")
        y=c.read("<H")
        power=c.read("<H")
        lives=c.read("<B")
        life_fragments=c.read("<H")
        bombs=c.read("<B")
        bomb_fragments=c.read("<H")
        inputflags=c.read("<B")
        graze=c.read("<I")
        piv=c.read("<I")
        final_score=c.read("<Q")
        stage_lives_used=c.read("<B")
        stage_bombs_used=c.read("<B")
        stage_continues_used=c.read("<B")
        nevents=c.read("<H")
        checksum=c.read("<I")
        stages.append({"stage_id":stage_id,"seed":seed,"difficulty":difficulty,
                       "stage_flags":stage_flags,"event_count_declared":nevents,
                       "start_time":start_time,"start_score":start_score,
                       "final_score":final_score,"start_x":x,"start_y":y,
                       "starting_lives":lives,"starting_bombs":bombs,
                       "stage_bombs_used":stage_bombs_used,
                       "stage_lives_used":stage_lives_used})
    if c.i != len(meta):
        raise ReplayError(f"trailing bytes in decoded metadata: {len(meta)-c.i}")
    return {"playername":meta[1:1+name_bytes].decode("utf-8",errors="replace"),
            "flags":flags,"stage_count":nstages,"stages":stages}


def parse_events_v14(payload: bytes, stages: list[dict]):
    total = sum(s["event_count_declared"] for s in stages)
    if total <= 0:
        raise ReplayError("all replay stage event counts are zero")
    if len(payload) != total*7:
        raise ReplayError(f"events length mismatch: declared {total} * 7, got {len(payload)}")
    cursor=0
    for s in stages:
        if s["event_count_declared"] == 0:
            raise ReplayError("stage with zero replay events")
        previous=-1
        histogram={}
        for _ in range(s["event_count_declared"]):
            frame,typ,val=struct.unpack_from("<IBH", payload,cursor)
            cursor += 7
            if frame < previous:
                raise ReplayError("events have decreasing frame indices")
            previous=frame
            histogram[str(typ)]=histogram.get(str(typ),0)+1
        s["last_event_frame"]=previous
        s["event_type_counts"]=histogram
    return total


def decompress_zstd(blob: bytes, *, max_output=MAX_SECTION_BYTES) -> bytes:
    if not blob.startswith(ZSTD_MAGIC):
        raise ReplayError("compressed section missing Zstandard magic")
    try:
        import zstandard as zstd
    except ImportError:
        zstd = None
    if zstd is not None:
        try:
            import io
            with zstd.ZstdDecompressor().stream_reader(io.BytesIO(blob)) as r:
                out=r.read(max_output+1)
        except zstd.ZstdError as e:
            raise ReplayError(f"zstandard decode failed: {e}") from e
        if len(out)>max_output:
            raise ReplayError("decoded section exceeds safety size cap")
        return out
    binary=shutil.which("zstd")
    if binary is None:
        raise ReplayError("full decode unavailable: install Python 'zstandard' or zstd CLI")
    try:
        process=subprocess.run([binary,"-dc","--quiet"],input=blob,capture_output=True,timeout=15)
    except subprocess.TimeoutExpired as e:
        raise ReplayError("zstd decompression timed out") from e
    if process.returncode:
        raise ReplayError("zstd decompression failed: "+process.stderr[:250].decode(errors="replace"))
    if len(process.stdout)>max_output:
        raise ReplayError("decoded section exceeds safety size cap")
    return process.stdout


def analyze(path: str | Path, *, log_path: str | Path | None = None) -> dict:
    p = Path(path)
    report={"file":str(p),"status":"unverified", "replay_playability":"not_tested",
            "visual_inspection":"not_performed","policy_attribution":"not_tested"}
    if not p.is_file():
        report["status"]="missing_replay"
        return report
    size=p.stat().st_size
    report["file_bytes"]=size
    if size>MAX_FILE_BYTES:
        report["status"]="too_large_for_bounded_audit"
        return report
    b=p.read_bytes()
    report["sha256"]=hashlib.sha256(b).hexdigest()
    if len(b)<HEADER_LEN or b[:10]!=MAGIC:
        report["status"]="invalid_header"
        return report
    version_raw=struct.unpack_from("<H",b,10)[0]
    version=version_raw & 0x7fff
    compressed=bool(version_raw & 0x8000)
    events_offset=struct.unpack_from("<I",b,17)[0]
    report.update({"replay_struct_version":version,"compressed":compressed,
                   "game_version_bytes":list(b[12:17]),"events_offset":events_offset})
    if version!=14 or not compressed:
        report["status"]="unsupported_replay_variant"
        return report
    if not HEADER_LEN<events_offset<len(b):
        report["status"]="invalid_events_offset"
        return report
    # Taisei src/replay/write.c leaves a 4-byte gap after closing metadata
    # and before the events stream (`end_of_meta + 4`). Do not treat that gap
    # as a third game event or a zero-content compressed frame.
    gap=b[events_offset-4:events_offset]
    have_gap=gap==b"\0\0\0\0" and events_offset-4>=HEADER_LEN
    meta_end=events_offset-4 if have_gap else events_offset
    metaz=b[HEADER_LEN:meta_end]
    eventsz=b[events_offset:-1]
    report.update({"metadata_compressed_bytes":len(metaz),
                   "events_compressed_bytes":len(eventsz),
                   "metadata_alignment_gap_bytes":4 if have_gap else 0,
                   "trailing_sentinel_byte":b[-1]})
    if metaz==EMPTY_ZSTD_FRAME and eventsz==EMPTY_ZSTD_FRAME:
        report["status"]="invalid_empty_metadata_and_events"
        report["root_cause_conclusion"]="serialization output empty; cannot distinguish missing encoder input from failed compressed writes without instrumented native logs"
    elif metaz==EMPTY_ZSTD_FRAME:
        report["status"]="invalid_empty_metadata"
    elif eventsz==EMPTY_ZSTD_FRAME:
        report["status"]="invalid_empty_events"
    else:
        try:
            meta=decompress_zstd(metaz)
            ev=decompress_zstd(eventsz)
            report["metadata_decompressed_bytes"]=len(meta)
            report["events_decompressed_bytes"]=len(ev)
            metadata=parse_metadata_v14(meta)
            count=parse_events_v14(ev,metadata["stages"])
            report["decoded_stage_count"]=metadata["stage_count"]
            report["decoded_event_count"]=count
            report["stages"]=metadata["stages"]
            report["status"]="structurally_decoded_requires_game_playback"
        except ReplayError as e:
            report["status"]=("decoder_unavailable_requires_zstandard" if str(e).startswith("full decode unavailable:")
                              else "invalid_or_unverified_decoded_payload")
            report["decode_error"]=str(e)
    if log_path is not None:
        log=Path(log_path)
        if not log.is_file():
            report["matching_log"]={"status":"not_found","path":str(log)}
        else:
            interesting=[]
            for line in log.read_text(encoding="utf-8",errors="replace").splitlines():
                if any(signal in line.lower() for signal in ERROR_SIGNALS):
                    interesting.append(line[:1200])
            report["matching_log"]={"status":"read","path":str(log),
                                     "matching_lines":interesting[-150:]}
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("replay",type=Path)
    ap.add_argument("--log",type=Path,help="matching Taisei log (read-only)")
    ap.add_argument("--out",type=Path,help="write JSON result to a new unique path")
    a=ap.parse_args(argv)
    result=analyze(a.replay,log_path=a.log)
    print(json.dumps(result,indent=2,sort_keys=True))
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open("x",encoding="utf-8") as f:
            json.dump(result,f,indent=2,sort_keys=True)
            f.write("\n")
    return 0 if result["status"]=="structurally_decoded_requires_game_playback" else 2

if __name__=="__main__":
    raise SystemExit(main())
