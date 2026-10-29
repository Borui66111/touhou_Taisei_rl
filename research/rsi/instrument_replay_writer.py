#!/usr/bin/env python3
"""Opt-in, exact-context Taisei replay-save C instrumentation for a local worker.

Not an engine replacement. Only accepts the matching upstream `src/replay/replay.c`
function body and makes a backup on --apply; no changes without --apply.
Compile the user's existing matching game source after applying and review logs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

NEEDLE = (
    "\tbool result = replay_write(rpy, file, REPLAY_STRUCT_VERSION_WRITE);\n"
    "\tSDL_CloseIO(file);\n"
    "\tvfs_sync(VFS_SYNC_STORE, NO_CALLCHAIN);\n"
    "\treturn result;"
)
MARKER="RSI_REPLAY_WRITE_BEGIN"
REPLACEMENT = r'''    // RSI DEBUG, source provenance: manual instrumentation, not replay repair.
    // Stage and input counts distinguish a lost recording from a writer failure.
    size_t rsi_events = 0;
    dynarray_foreach_elem(&rpy->stages, ReplayStage *rsi_stg, {
        rsi_events += rsi_stg->events.num_elements;
    });
    log_info("RSI_REPLAY_WRITE_BEGIN stages=%zu events=%zu",
             (size_t)rpy->stages.num_elements, rsi_events);
    bool result = replay_write(rpy, file, REPLAY_STRUCT_VERSION_WRITE);
    int64_t rsi_bytes = SDL_TellIO(file);
    // Do not silently ignore an unsuccessful stream close.
    bool rsi_close_ok = SDL_CloseIO(file);
    vfs_sync(VFS_SYNC_STORE, NO_CALLCHAIN);
    log_info("RSI_REPLAY_WRITE_END stages=%zu events=%zu bytes=%" PRIi64
             " write_ok=%d close_ok=%d",
             (size_t)rpy->stages.num_elements, rsi_events,
             rsi_bytes, (int)result, (int)rsi_close_ok);
    if(!result || !rsi_close_ok || rsi_bytes <= 44) {
        log_error("RSI_REPLAY_WRITE_INVALID: stages=%zu events=%zu bytes=%" PRIi64,
                  (size_t)rpy->stages.num_elements, rsi_events, rsi_bytes);
    }
    return result && rsi_close_ok && rsi_bytes > 44;'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(source: Path):
    source=Path(source)
    if not source.is_file():
        return {"status":"missing_source","source":str(source)}
    try:
        text=source.read_text(encoding="utf-8")
    except UnicodeError:
        return {"status":"non_utf8_source","source":str(source)}
    if MARKER in text:
        return {"status":"already_instrumented","source":str(source),"sha256":sha256(source)}
    if text.count(NEEDLE)!=1 or not text.count("bool replay_save(Replay *rpy, const char *name)") == 1:
        return {"status":"source_signature_mismatch","source":str(source),"sha256":sha256(source),
                "reason":"Expected replay_save body not found exactly once. Do not force-apply to a different game revision."}
    return {"status":"ready","source":str(source),"sha256":sha256(source),
            "backup":str(source.with_name(source.name+'.rsi-preinstrument-backup'))}


def apply(source: Path):
    source=Path(source)
    result=inspect(source)
    if result["status"]!="ready":
        return result
    backup=source.with_name(source.name+'.rsi-preinstrument-backup')
    if backup.exists():
        return {**result,"status":"backup_already_exists_refuse_overwrite"}
    old=source.read_text(encoding="utf-8")
    new=old.replace(NEEDLE, REPLACEMENT)
    assert new!=old and new.count(MARKER)==1
    # Create exclusive backup first and never overwrite a prior original.
    with backup.open('x',encoding='utf-8',newline='') as out:
        out.write(old)
    try:
        source.write_text(new,encoding='utf-8',newline='')
    except OSError:
        # Original backup remains; source must be repaired manually if write failed.
        raise
    return {"status":"applied_rebuild_required","source":str(source),
            "source_original_sha256":hashlib.sha256(old.encode()).hexdigest(),
            "source_instrumented_sha256":sha256(source),
            "backup":str(backup),"backup_sha256":sha256(backup),
            "runtime_verified":False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path,help='Exact src/replay/replay.c in local matching Taisei source checkout')
    p.add_argument('--apply',action='store_true',help='Opt-in: patch source after exclusive backup; rebuild game manually')
    args=p.parse_args(argv)
    result=apply(args.source) if args.apply else inspect(args.source)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result['status'] in {'ready','already_instrumented','applied_rebuild_required'} else 2

if __name__=='__main__':raise SystemExit(main())
