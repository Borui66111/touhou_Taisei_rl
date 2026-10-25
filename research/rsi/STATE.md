# Taisei RSI automation: pilot state

## Purpose
A bounded GitHub Actions pilot verifies that the genuine pinned Taisei simulator can be built, executes a 300-frame authentic-native smoke test, runs 1,024 PPO training decisions for a skill-aware model, and evaluates that checkpoint against random and geometric-heuristic policies on the same four **development** seeds (42100-42103).

## Scope and safeguards
- Research branch: `research/rsi-automation-v1`; base snapshot: `d463a2fd`.
- Workflow: `.github/workflows/taisei-rsi-pilot.yml`.
- Entry point: `research/rsi/run_pilot.sh`; result validation: `research/rsi/summarize_pilot.py`.
- Actual Taisei native library is mandatory; the job aborts on build/load/smoke failure. No toy fallback.
- Native simulator pinned via `my_test/UPSTREAM.lock`.
- This is **NOT** autonomous parameter search, a promotion decision, or proof of RSI improvement. It establishes working automation and comparable evidence.
- Final sealed seeds **43000-43099 are never evaluated or trained on**.
- The small 4-seed evaluation is diagnostic only; it cannot support a robust improvement claim.
- No secrets, tokens, or write access required by the workflow; GitHub Action has `contents: read`.
- Artifacts retained 14 days; copy meaningful results to permanent versioned evidence if research requires retention.
- The current skill-action mask is known to exclude early discharge of an already active Power Surge; fixing that behavior and reward-scale calibration should precede *scientific* RSI model comparisons.
- Do not automatically promote any pilot model to the champion.

## Operational steps
1. In GitHub Actions, open *Taisei RSI pilot (native, fail closed)* on the research branch.
2. Inspect failures in `native_build.log`, `native_smoke.log`, `pilot_training.stdout.log` in that order.
3. Verify `summary.json` and the generated checkpoint hash before making any claims.
4. Once infrastructure passes, add controlled one-factor experiments with matched seeds, multiple independent training seeds, explicit model identity, and champion non-regression checks.
5. Keep all RSI candidate comparisons on a development branch and do not touch `main` or final seeds.

## Current state
- Prior native smoke failed at SDL3 configuration because X11/Wayland development headers were missing. The research pilot now installs those development packages, but **successful build and training remain unverified until Actions completes**.
- The current agent-side container cannot directly compile the engine; the GitHub runner is the execution backend.

## Build troubleshooting 2026-10-25

- GitHub Actions run 40122026719 reached SDL3 CMake but failed due missing XTEST development package (`libxtst-dev`). No smoke or PPO run completed.
- Targeted workflow dependency fix: commit `48493153b9dc7dc5f2d6911c31b740fa9c611160` on `research/rsi-automation-v1`; subsequent run still needs native build/smoke/train verification.
- New independent competence research branch `research/rsi-competence-v1` contains the skill-oriented design and descriptive comparison evaluator. It must not overwrite this runtime branch or reuse the sealed final seeds.

## SDL3 static/PIC native build follow-up — 2026-10-25

- Repaired XTEST dependency run 40122420549 proceeded past SDL3 CMake and failed when Meson refused non-PIC `SDL3_static` while linking `libtaisei_sim.so` (pinned wrapper forces static SDL).
- Added `research/rsi/build_native_with_shared_sdl3.sh`: builds the SDL source tag named in Taisei's wrapper as an external **shared** library in the temporary CI runner, without modifying the pinned game source checkout. `run_pilot.sh` now supplies the package path and runtime loader path.
- Run 40122713708 is the first attempt of the shared-SDL workaround. Its native smoke, training and results are **unverified** until its job actually completes.

## 2026-10-25 — E1 preregistered: learning defensive movement without skills

- **Trigger:** real Taisei 1024-step skill PPO pilot run 40185526694 survived mean 3455 frames with 9 Bombs per episode; paired same-checkpoint native skill-off run 40186512931 survived only 792.25 frames, versus 1630.75 for the heuristic. All policy variants cleared 0/4 and lost 3 lives. The v2 checkpoint is resource-dependent; evidence does not establish competent dodging.
- **E1 question:** does a survival/clear-oriented reward formulation improve PPO **movement-only** survival when native Bomb/Surge actions are unavailable?
- **Two arms, only reward differs:** original control vs survival_first (alive +0.002, death -30, loss -30, clear +100, graze 0, boss damage 0.001). Same native Easy Stage 1, 18-action movement-only ActorCritic, learning rate 3e-4, gamma .995, rollout 512, training seed 48020, exactly 20k native game interactions each.
- **Evaluation:** 16 identical development seeds 42100-42115 for each policy, plus no-skill geometric heuristic. Native deterministic replay check, same game build, exact episode statistics, paired bootstrap interval over evaluation seeds (does not cover multi-training-seed uncertainty). Sealed final seeds 43000-43099 remain untouched.
- **Status:** experiment source and executable pipeline prepared on `research/rsi-movement-reward-e1`; no training evidence exists until new GitHub Actions run passes. No champion promotion authorized.
- **Next:** review native build, replay determinism, each training checkpoint and manifest, actual deaths/first-hit/clear, paired baseline outcomes. If the calibrated reward fails to improve movement, prioritize threat representation and teacher-guided curriculum rather than repeated reward-only tuning.
