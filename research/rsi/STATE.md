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

## User-directed canonical branch policy — 2026-10-25

- **Default for ALL future Taisei RSI development:** `research/rsi-automation-v1` (the branch containing this file).
- Standing instructions are in `research/rsi/BRANCH_POLICY.md`: **do not create an experimental branch** absent a verified technical reason documented in the trajectory, with an integration plan and review where consequential.
- Prior work in `research/rsi-easy-curriculum-fix`, `research/rsi-competence-v1`, `research/rsi-meta-controller-v1` and other experiment branches is historical source/evidence, **not automatically merged** into this currently older branch. Before reusing fixes, carefully review and integrate compatible, verified changes into this selected branch without rewriting history.
- Daily autonomous RSI research and hourly milestone monitor have both been updated to target this canonical branch. Check GitHub Actions branch triggers after each workflow integration.

## 2026-10-26 — E2 prediction-based defensive movement research (pending native comparison)

- **Research hypothesis:** published no-skill PPO policies exhibit strong boundary-pushing proxies and limited responsiveness; a transparent native-observation-only risk teacher with explicit projected collision and wall penalties may produce better native survival than the current geometric heuristic. No gain has yet been measured.
- **Source:** `research/rsi/risk_aware_teacher.py`; source-only guard tests: `test_risk_aware_teacher.py`, `test_e2_teacher_compare.py`. Tests passed in workflow [40212565935](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40212565935). This is not gameplay validation.
- **Native protocol:** `research/rsi/run_e2_teacher.sh`, `research/rsi/e2_evaluate_teacher.py`, `research/rsi/e2_compare_teacher.py`, workflow `.github/workflows/rsi-e2-teacher.yml`. Genuine pinned Taisei Stage1 Easy, Reimu A, 16 development seeds 42100-42115, equal initial conditions, no neural skill head or Bomb/Special actions, fixed controller coefficients, 60-minute job budget, no auto-promotion or sealed final seeds.
- **Active gate:** [native pilot 40212525844](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40212525844) was still in progress at latest inspection. Do **not** start the E2 native comparison while another native job is active.
- **CI failure diagnosed:** previous native pilot [40211694671](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40211694671) failed resource packing because installing `zstandard` did not satisfy Taisei's `backports.zstd` import. The canonical pilot workflow now installs/tests `backports.zstd` and **only responds to an explicit `PILOT_RUN_REQUEST.md`**, avoiding native builds after ordinary research edits. The ongoing older job predates this repair and can still fail.
- **Next:** after any active native job finishes, submit `E2_TEACHER_RUN_REQUEST.md` on this same branch to launch the reproducible E2 native evaluation, check the result artifacts and append observed outcomes. If teacher is better, study controlled distillation/auxiliary policy training as a separate L2/L0 challenger. If worse, falsify/revise the predictive movement objective before PPO HPO.
- **Live replay reference:** `research/rsi/evidence/rendered_challenger_replay_20261026.md` identifies 1016 uninterrupted Left input frames, 7 Bomb uses and no clear, but does not prove actual wall contact or checkpoint identity.
