# Taisei RSI debug: replay serializer + E1 action observability (2026-10-29)

**Canonical branch:** `research/rsi-automation-v1`; **single rendered control ABI:** `rl_skill_v2` / `TSKILL2` / 546 features, 18 movement + 3 skill actions, mask `taisei_skill_mask_v2_respawn`.

## Reproducible evidence and new diagnosis

The two user-provided real-game `.tsr` files (16:34 and 17:33 on 2026-10-28) are SHA-256-identical (`3eb7427f4c3372d997c3718c699803c16bb42fead5a0f644ff800bed48958943`) at 44 bytes. Both Zstd frame bodies decompress to 0 bytes. Version v14's normal serializer always emits player-name length, global flags and stage count, even if stages have accidentally been omitted. Two **empty compressed streams** therefore implicate the serialization/writer path (or an unwritten compression stream), not just an empty list of gameplay events. The exact defect remains unproven until writer state is observed.

This was cross-checked against **public** `taisei-project/taisei` 2026-10-29 master source `src/replay/{write.c,replay.c,read.c,struct.h,rw_common.c}` and `src/rwops/rwops_zstd.c`. This is a source-reference analysis, **not** a verified rebuild of the user's local fork. `replay_save()` logs `Saving ...` before calling `replay_write()`. That code path did not validate whether the final compressed streams held actual metadata/events; the source's `SDL_CloseIO` result was unchecked. An empty `.tsr` plus `Saving ...` is not a successful save.

### Tests performed now

- Added `research/rsi/replay_forensics.py`: read-only binary v14 Zstd decoder, stage/event count and order verification with bounded size, and optional native log filtering. It returns a *distinct* status for unavailable decoder; full structural decode is not proof of playback.
- Added `research/rsi/instrument_replay_writer.py`: **opt-in and exact-context** backup-producing patcher for an existing `src/replay/replay.c` checkout. Emits `RSI_REPLAY_WRITE_BEGIN` and `RSI_REPLAY_WRITE_END` with stage/event counts, bytes written, and write/close success; logs `RSI_REPLAY_WRITE_INVALID` on obviously invalid results. It does not repair the compression implementation.
- Extended `my_test/scripts/w2ctrl.py` source-level audit to separate **logged RL control** from **replay-writer correctness**. If a writer log reports invalid or incomplete output, the RSI evidence status is `failed_replay` even if the controller made accepted decisions.
- On current local Python environment, existing W2CTRL suite **78 cases: 76 passed, 2 optional skipped**; new replay forensic/instrumentation suite **23/23 passed**. Synthetic fixtures used for positive replay decoder tests are **not actual game recordings**.

### Windows local-worker next diagnostic commands

After pulling the canonical research branch, from the **repository root**:

```powershell
py -3.9 research/rsi/replay_forensics.py 'C:\actual\recording.tsr' --log "$env:APPDATA\taisei\log.txt"
```

For fully compressed nonempty future replays, install `py -3.9 -m pip install zstandard` or ensure `zstd` CLI is available. The known 44-byte empty replay needs neither.

Inspect the exact game **source path** of the currently built `rl_skill_v2` game before instrumenting. Do not apply to an unrelated checkout. Do not guess the source directory.

```powershell
# Source path is an example: replace with the REAL source checkout used by rl_skill_v2
py -3.9 research/rsi/instrument_replay_writer.py --source 'C:\actual\taisei\src\replay\replay.c'
# Only after inspecting dry-run report and matching game source:
py -3.9 research/rsi/instrument_replay_writer.py --source 'C:\actual\taisei\src\replay\replay.c' --apply
```

The instrumentation copies the exact original to `replay.c.rsi-preinstrument-backup` and refuses to overwrite it. Rebuild the existing matching `rl_skill_v2` executable through its **existing** Meson build; then run a single **development** stage (no sealed evaluation seeds) with E1. Keep `%APPDATA%\taisei\log.txt`, new `%LOCALAPPDATA%\hermes\cache\scratch\taisei-live-ppo-*.json`, the `.tsr`, and independently captured real rendered video/frames. If the patcher rejects an unknown source signature, do not force it.

**Interpretation of new writer diagnostics:**

| Writer log | Meaning | Next diagnosis |
|---|---|---|
| `stages=0` (even when gameplay is seen) | Replay state was not populated | Inspect stage-record initialization and lifecycle |
| `stages>=1 events=0` | Inputs were not recorded | Inspect stage input event hooks |
| `stages>=1 events>0 bytes=44` | Replay inputs existed, but output streams are empty | Focus on Zstd stream writer, SDL I/O callbacks, write result and flush/close |
| `write_ok=0` or `close_ok=0` | Writer reported failure | Preserve SDL/Zstd error logs and debug I/O layer |
| `bytes>44` | Some output written; not yet a valid replay | Run full decoder, then playback in genuine Taisei and compare frame traces |

### RSI: E1 dodge and collection

The real native environment exposes item positions through `obs['items']`, but `skill_features` drops them from E1's 546 input floats; the native reward has no pickup term. An actual E1 checkpoint experiment confirms that changes to item positions alone cause no output changes. Consequently, do **not** extend blind PPO or claim autonomous collection is trainable under the current interface. A versioned item-aware 546-feature layout and synchronized C-rendered/native encoders, plus a pickup event reward guarded by survival constraints, are prerequisites for E7. Never reinterpret existing E1 weights under a changed layout.

The user's visual observation of poor dodging is real feedback, but without a fresh real-game frame/action trace, native↔rendered feature parity, and matched-seed evaluation, the cause (training quality, observation mismatch, policy stochasticity or action wiring) remains unknown. No new native Taisei training or rendered video replay was run in this research iteration because the exact pinned simulator source and the local Windows binary are unavailable in the assistant environment. The canonical research and one live-control ABI remain unchanged.

## Independent, limited offline E1/E4 movement response probe

`offline_policy_diagnostic.json` was produced by applying archived E1/E4 actor weights to the archived **E3 12k demonstrations**. E1 requires 546 features; E3 provides 528, and 18 dummy zero suffixes were appended. This is deliberately **out of distribution** for E1, cannot stand in for live observations, and its teacher-action agreement is **not** a measure of real dodging. It is intended only to motivate a native observation/actor trace, not a new model promotion.

**Next acceptance gates:** exact C game source + writer logs; nonempty v14 replay event and stage payload; genuine playback/visual review; C-vs-native 546-feature parity; item feature/schema migration; bounded E7 matched development-seed evaluation. No scheduled automation or source test is a substitute for these.
