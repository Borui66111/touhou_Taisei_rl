# Local neural-survival replay provenance and failed recording — 2026-10-26

## User-provided claim and evidence

- User identifies the local RL result/model name as `taisei_best_verified_neural_survival_20261025`.
- Uploaded file: `taisei_20261026_19-42-50-365_stg1_reimuA_easy.tsr`.
- Exact file bytes: **44 bytes**, SHA-256 `3eb7427f4c3372d997c3718c699803c16bb42fead5a0f644ff800bed48958943`.
- Filename suggests Stage 1 / Reimu A / Easy, but there is **no usable stage metadata inside this file to verify any stage, player, outcome or duration**. File name and user-reported policy label are not model provenance receipts.

## Binary check

The first ten bytes match Taisei's replay magic `hono❤umi`. The next two encode compressed replay struct version 14 (`0x800e`); the 5-byte stored game version indicates 1.5.0. Header's event-data offset is 34. Metadata bytes are offsets 21..33 (13 bytes), while the remainder is 10 bytes.

Both sections **begin with the exact zstd compressed empty frame** `28b52ffd2000010000`. A native `libzstd` decompression test independently verified that this 9-byte frame expands to **zero bytes** in each section. Extra trailing bytes in the metadata/events slices do not furnish readable player/stage or input-event records.

Taisei's official `src/replay/read.c` requires a player name, flags, a nonzero number of stage structures, and then event records. The uploaded file does not contain the required decoded stage metadata. It is **header-only/empty, not an auditable gameplay replay**. Unlike the prior Stage 1 Easy challenger capture (17,701 bytes), its survival, deaths, inputs, Bombs, Power Surge, score, clear flag and RNG seed are **all unknown**, not zero.

Source: https://github.com/taisei-project/taisei/blob/master/src/replay/read.c ; https://github.com/taisei-project/taisei/blob/master/src/replay/struct.h

## Conclusions and attribution limits

1. The 44-byte file is **not evidence of gameplay failure or success**. It is a replay-capture/export deficiency (potentially a header-only file), and the exact local cause is not observable in the bytes alone.
2. The local label `taisei_best_verified_neural_survival_20261025` is **user-reported**; this file does not embed a model SHA-256, `TSKILL2` acceptance, controller action logs, or checkpoint lineage. Do not equate it with any GitHub neural checkpoint without matching a manifest and the local loader log.
3. A complete replay is necessary for input-sequence analysis, but a replay alone generally does not certify the controlling neural checkpoint. Record both the replay and unrotated log / file SHA.
4. No native Taisei training, new checkpoint, champion promotion or final-seed evaluation was performed during this file audit.

## Local worker recovery instructions

- Exit the recording/playback session normally and ensure the Taisei process has flushed and closed the replay file before copying it; preserve game and launcher logs before their rotation.
- Verify replay using `python research/rsi/inspect_rendered_tsr.py path/to/capture.tsr`. Exit code 2 rejects obvious empty/invalid files; exit code 0 is **only a preliminary header check**, not proof of complete playable events.
- Load the completed `.tsr` in the same rendered Taisei build to confirm there is a recorded stage and actionable inputs. Capture video or position/hazard telemetry when feasible.
- Save local model binary SHA256, accepted loader family (e.g. `TSKILL2`), policy identifier/manifest, recorded seed/stage, actual AI decision count and replay SHA256 together. Prevent human controls or unknown fallback from masquerading as a neural run.
- Reupload a replay with valid stage metadata and events; only then compare against previous runs or update learning conclusions.

## Engineering mitigation

`research/rsi/inspect_rendered_tsr.py` is a read-only, Windows-capable standard-library preflight that rejects this exact 44-byte empty-stage replay, and `research/rsi/test_inspect_rendered_tsr.py` tests it. The validator intentionally does not assert gameplay success even for a plausible header.

**Decision:** Classify as `invalid_replay_capture`, not RL performance. Preserve the model label as a user-reported provenance hint and request a complete replay plus loader identity.
