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
