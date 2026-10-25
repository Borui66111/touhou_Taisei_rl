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

## 2026-10-25 — E1 verified negative; E2 native movement diagnostics started

- E1 authentic native Easy Stage 1 workflow [40187280987](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40187280987) **passed** build, 300-frame smoke, two 600-frame deterministic native trajectories (identical game digests), source tests, and two movement-only 20,000-step PPO arms, with 16 paired development-seed evaluations. Checkpoint hashes baseline `a2a0ce0d4716945b228641844f65620ed1548a4342f7f891539e98303c1eed7b`, survival-first `5526812cb58988849165389db39eec50add78d0dfc41840f5eba1ecc491b93b8`.
- Results on seeds `42100–42115`: geometric heuristic mean alive 1659.875 frames; legacy 18-movement original-reward PPO 1036; survival-first bundle PPO 901.0625. All 0 clears, 3 deaths, zero Bombs. Paired change survival-first minus original = **−134.9375 alive frames**, eval-seed bootstrap interval [−176.375,−105.8125], excludes between-training seed variance. Neither PPO arm demonstrates authentic dodging; this one reward-bundle experiment is negative, not proof reward engineering never works. The single training seed 48020 covers only ~7 full episodes per arm.
- Evidence committed at `research/rsi/evidence/native_movement_reward_e1_40187280987.json` on the meta-controller branch; official 14-day CI artifacts expire 2026-11-08.
- Higher-level RSI decision: do **not** scale reward-only PPO or promote checkpoint; perform E2 cheap native trajectory diagnosis first, inspecting actual movement action frequencies, border-push proxy, immediate projectile closest-approach estimates, and first death increment. These proxies do NOT constitute verified dodges.
- E2 isolated branch `research/rsi-movement-trace-e2`: download verified E1 checkpoint artifacts by fixed hash from run 40187280987, compile pinned Taisei (no toy), native-smoke, capture 4-seed movement+hazard traces for both PPO arms and geometric heuristic, summarize results without training or promotion. Current CI run [40188054001](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40188054001) was started; results pending.
- Sealed final seeds `43000–43099` untouched. Next action after E2: decide whether to test teacher-distillation/safety-filtered control, a versioned time-to-impact feature, action persistence or short history, **one factor at a time**.
