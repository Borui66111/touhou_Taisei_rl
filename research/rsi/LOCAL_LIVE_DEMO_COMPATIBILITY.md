# Local Taisei live-demo policy compatibility and challenger replay audit

**Date:** 2026-10-25. **Canonical research branch:** `research/rsi-automation-v1`.

## Reported local worker evidence (not independently replayed)

- The current local rendered launcher is **`rl_skill_v2`**, expecting a **`TSKILL2`** policy family, 546 input features, separate movement + skill heads.
- `survival_first_seed48020.bin` is **`TPPOv1`**, 528 inputs, 18 movement actions. Worker reported SHA-256 `d317303dfd02f76ce646c84646b812bd1d490eb81abc6f90fc48e86a21167e0b`.
- Worker observed process exit status 0, but launcher classification `invalid_or_unknown_format`; game log rolled over, so the execution did not establish that the policy ran.
- Worker identified local `challenger_seed48020_live_v2.bin` as the matching export for the skill-v2 demo. That binary and the local loader's source have **not** been uploaded/verified here.
- `vswhere.exe` warning was also present in a successful dry run; do not attribute the failure to that warning.

**Decision:** TPPOv1 and TSKILL2 are not interchangeable. Use the skill-v2 export only with the skill-v2 demo, after verifying its identity/hash and seeing explicit loader acceptance and nonzero native action counts. Legacy TPPOv1 requires a matching legacy loader (or a separately validated adapter/rebuild). Do not rename the binary to bypass the guard.

## Executable preflight

`python research/rsi/policy_format_preflight.py PATH_TO_POLICY.bin --demo skill-v2`

Read-only guard outputs actual file SHA-256 and recognized format. TPPOv1 shape/byte count is validated. TSKILL2 is **family-recognized only** because the full v2 binary specification is not available in this repo; it must not claim successful live load or inference.

`python research/rsi/policy_format_preflight.py PATH_TO_POLICY.bin --demo legacy-movement --expected-sha256 SOME_SHA256`

For legacy policy, the prior native renderer hardcodes an accepted policy-binary SHA-256. This checker deliberately does not claim that the renderer's allowlist has been updated.

## To validate a challenger rendered replay

1. Obtain the actual new `.tsr` replay file (or captured video) and unrotated launch logs. At the time of this integration, no **new** challenger replay is attached to the conversation.
2. Verify the logged policy binary SHA-256, recognized magic, loader acceptance, run seed, stage/difficulty/shot, `RL decision` lines and outcome. The replay alone may not identify the checkpoint.
3. Parse replay metadata and the exact input sequence; quantify survival duration, movement/actions, skill presses, action-change entropy, wall pinning proxies and outcome.
4. Assess dodge/collect competence only with synchronized native bullet/item/player trajectories or video, not direction changes alone.
5. Compare only identical stage/spell, difficulty, initial lives/Bombs, engine version, controller, and seeds. Keep final 43000–43099 sealed.
6. Append the conclusion (including negative evidence) to `research/rsi/trajectory/events.jsonl` after the actual file is available.

This document is an integration and diagnostic plan, **not a validated live-game performance claim**.
