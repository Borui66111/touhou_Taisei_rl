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

## 2026-10-25 — Competence-first RSI design

- Independent research branch: `research/rsi-competence-v1`.
- Full ability specification: `research/rsi/COMPETENCE_SPEC.md`; requires actual dodging, successful survival, confirmed native collection, and strict promotion gates.
- Executable descriptive evaluator: `research/rsi/competence_eval.py` + nine unit tests in `research/rsi/test_competence_eval.py`.
- Benchmark uses existing native JSONL records (42000–42015); invalid/incomplete provenance forbids promotion, sealed final seeds rejected.
- **Blocked:** new native training and validated collection events still need actual compiled simulator + native instrumentation. Original repair pipeline run 40122026719 failed during SDL3 configuration because XTEST dev package absent; a targeted fix is applied on `research/rsi-automation-v1`.
- Next: validate evaluator on GitHub Actions, resolve native build, run baseline E0 and first reward-only E1 challenger on matched real-game seeds.

### E1 reward-only challenger prepared

- `train_reward_ablation.py` defines current reward (`control`) vs calibrated survival/clear reward (`challenger`) with the *same* architecture, mask, PPO settings and native game config. It refuses existing checkpoint/log paths. The launcher is **not yet executed on native Taisei**; `--preflight-only` validates configuration without inventing gameplay.
- `test_reward_ablation.py` adds four regression tests. `rsi-competence-checks.yml` runs all 13 unit tests plus a descriptive comparison of published 16-seed native evaluations.
- Native build follow-up after adding `libxtst-dev` is still being verified on the separate automation branch. Do not launch resource-heavy E1 training or claim results before it passes.

## 2026-10-25 — Meta-controller and experiment memory

- Research-method specification: `research/rsi/RESEARCH_OS.md` (L0 policy through L4 RSI-governance recursion; marginal return and human-review rules).
- Evidence ledger: `research/rsi/trajectory/events.jsonl`, currently 11 linked events including retrospective toy-to-native pivot and daily continuation setup; source refs and limitations are explicit. Latest event IDs `T-20261025-04` and `T-20261025-05` are retrospective/operational, not evidence of native gameplay learning.
- Validator: `research/rsi/trajectory/ledger.py` plus `test_ledger.py`; CI `.github/workflows/rsi-meta-audit.yml` checks byte-prefix history and validates linked events. Initial run 40126793965 passed all 10 tests; appended events require follow-up CI confirmation.
- Daily safe RSI research continuation enabled; hourly status watcher remains separate. Future execution/actual contribution is unverified until the first scheduled run.
- Current bottleneck: latest native workflow run 40122731438 failed in SDL3 configuration before smoke/training. Continue targeted native build investigation before another training experiment; no sealed final seeds and no champion promotion.
- UKS persistent synthesis: `/Knowledge/05_Methods/RSI_Multi_Level_Research_Operating_System_and_Trajectory_2026-10-25.md` with live Git evidence pointers.

## 2026-10-25 — Follow-up: real native build and post-build runtime

- Native Easy pilot run 40128137506 **successfully completed all 618/618 build tasks**, including `src/sim/libtaisei_sim.so` and game resource packages. It then failed because `meson devenv -C ... -- python` was interpreted as trying to execute `--`; no real gameplay/training occurred.
- Next run 40185187240 removed `--`; it again compiled but `meson devenv` changed the working directory to `build/`, so relative `scripts/smoke_native.py` could not be found; still no native gameplay.
- Isolated branch `research/rsi-easy-curriculum-fix` now uses absolute workbench script paths, with preflight assertion to catch regression. Native pilot run 40185526694 is the latest run to verify. Do not treat compile success as proven ABI or gameplay.
- Separate agent-strategy branch `research/rsi-surge-mask-v3` corrects Power Surge early-discharge eligibility without reinterpreting historical v2 checkpoints. GitHub Actions run 40185726369 passed 11 unit tests. This is a code correctness result, not native gameplay success.
- Current research gates: actual reset/step smoke; deterministic replay check; 1,024-step training and checkpoint hash; same Easy difficulty 1/development seeds; E0 baselines then E1 reward-only challenger. Sealed 43000–43099 untouched.
