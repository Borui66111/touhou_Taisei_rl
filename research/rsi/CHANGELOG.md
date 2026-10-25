# RSI Automation Changelog

## 2026-10-25 — Initial native pilot

**Issue:** Previous GitHub Actions smoke job failed before Taisei compiled: SDL3 CMake configuration could not find X11 or Wayland development libraries.

**Reason:** The isolated GitHub-hosted Ubuntu runner did not have the necessary Linux windowing headers for the SDL3 fallback dependency to configure, even though a headless simulator was the target.

**Fix:** Created a new isolated `research/rsi-automation-v1` branch and a fail-closed pilot workflow with Linux windowing build dependencies. Added a 300-frame real-native smoke gate, bounded skill-aware PPO training (1,024 decisions), a four-seed matched evaluation against random and heuristic baselines, and evidence artifacts.

**Validation required:** Build and native gameplay must complete in GitHub Actions. The training/evaluation stage has not yet been confirmed successful. No champion promotion or final-seed testing permitted.

**Known research issues:** Survival reward scale, Power Surge discharge mask behavior, and deployment parity need independent resolution before full RSI experiments.

## 2026-10-25 — SDL3 XTEST dependency repair

**Issue:** The full native pilot run 40122026719 stopped at SDL3 CMake: dependency package for XTEST was not found.

**Reason:** The Ubuntu GitHub runner had X11 headers but not the XTest development package; native library was not built and no RL experiment ran.

**Fix:** Include `libxtst-dev` and `libxrender-dev` in the workflow's apt installation. Only the workflow dependency list was edited; no RL/policy code changed.

**Validation:** Await the new Actions run for actual compiled native engine and genuine gameplay/training outputs.

## 2026-10-25 — PIC-safe SDL3 shared dependency for actual native simulation

**Issue:** Taisei native build run 40122420549 failed at `src/sim/meson.build`: non-PIC `SDL3_static` cannot be linked into `taisei_sim` shared library.

**Reason:** The pinned `subprojects/sdl3-cmake-wrapper/meson.build` forces `SDL_SHARED=false`, `SDL_STATIC=true`; linking the imported fallback into a shared target fails Meson's PIC validation.

**Fix:** On the disposable GitHub Actions runner, independently build tagged `taisei-project/SDL` as a shared library; expose pkg-config and loader paths so unchanged pinned Taisei resolves a compatible shared SDL3 dependency. Avoid rewriting upstream gameplay sources.

**Validation:** Await run 40122713708; no native training success is claimed yet.

## 2026-10-25 — E1 movement-only native reward ablation initiated

**Issue:** Native same-checkpoint 4-seed skill ablation showed PPO survival collapses from 3455 to 792.25 mean alive frames when Bomb/Surge are forcibly disabled; thus the current skill PPO's apparent advantage is mainly associated with resource use, not verified dodging.

**Reason:** Existing full-game PPO controls both movement and consumables, while per-alive-frame reward is +1 and death costs -10. No controlled movement-only reward study yet distinguishes flawed reward incentives from deficient observations or learning method.

**Fix:** Created `e1_train_movement.py` with two 20k interaction movement-only PPO reward arms, `e1_summarize.py` with matched 16-seed descriptive evaluation, `run_e1_movement.sh` with real Taisei build/smoke/determinism gate, and `test_e1_plan.py` with no-skill, task, budget and sealed-seed safety assertions. No auto promotion and no fabricated native results. Revisit hypotheses after CI evidence appears.
