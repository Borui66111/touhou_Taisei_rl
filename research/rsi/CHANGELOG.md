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

## 2026-10-26 — E2 native teacher and budget-safe workflow routing

**Issue:** Real native movement E1 traces show near-constant movements and boundary-pushing proxies; skill-aware replay also records 1016 consecutive Left frames. Previous reward changes failed to demonstrate reliable dodging. Separately, the pilot's broad `research/rsi/**` push trigger started costly native builds after research-only changes; its recent runner failed because `zstandard` did not provide the `backports.zstd` import required by pinned Taisei resource packaging.

**Reason:** The current PPO policy lacks explicit candidate-action threat clearance and boundary constraints, so reward changes alone may not yield tactical movement. CI triggers were too broad and the Python Zstandard dependency did not match the upstream packer imports.

**Fix:** Added a deterministic predictive-risk and wall-aware baseline controller in `research/rsi/risk_aware_teacher.py`, eight source-unit tests, a 16-development-seed native Easy comparison protocol with old heuristic, a fail-closed paired evaluator and a separate opt-in CI workflow. Changed the native pilot to explicit request-file triggering, installed and import-checked `backports.zstd`, and retained a 60-minute limit. Canonical branch unchanged.

**Verification:** Source-only teacher/comparison tests passed in GitHub Actions run 40212565935. Genuine native E2 evaluation is **not yet executed** because prior native pilot run 40212525844 remains active. Success in source tests is not evidence of learned dodging, improvement, or live render parity. No checkpoint was replaced or promoted.

**Next:** Run the isolated E2 native comparison when the active native runner is finished, inspect metadata/episode records and decide whether to research teacher-distilled PPO or revise the hazard model.

## 2026-10-26 — Cross-workflow native concurrency guard

**Observation:** At the 00:40 SGT check, native pilot run `40212525844` was still `in_progress` at the build/smoke/train/evaluate step; source-level diagnostics had passed. GitHub job logs for the active job were temporarily unavailable (`BlobNotFound`), so no gameplay or training outcome is inferred. The earlier pilot `40211694671` had failed during the native step; the repository records the `backports.zstd` resource-packer diagnosis and a dependency repair in the newer pilot workflow.

**Issue:** The pilot and the opt-in E2 teacher workflow used different Actions `concurrency.group` values (`taisei-rsi-pilot` vs `taisei-native-serial`), so CI concurrency did not actually serialize native work *across* these workflows.

**Fix:** Commit `a17bf744f49523ab25448f4c6edb3aed72ad6bfe` changes the pilot group to `taisei-native-serial`, matching the E2 teacher workflow. This protects **future** workflow runs; it does not retroactively change the concurrency group of the already-running pilot.

**Next gate:** Recheck `40212525844` completion and artifacts; do not submit an E2 run request while the older pilot remains active. No new native run was dispatched, no checkpoint promoted, and sealed final seeds remain excluded.

## 2026-10-26 — E2 positive native controller result and E3 imitation-learning pipeline

**Issue:** The prior PPO policies mostly push toward a boundary and fail to clear Stage1 Easy; reward ablations alone did not fix navigation. Even a higher-performing scripted controller would not establish a *learned RL policy* improvement.

**Reason:** The existing 528-feature MLP actor lacks an explicit predictive-risk decision rule. Before changing the architecture, it is worth testing whether a learned actor can imitate a source-verified, much stronger handcrafted movement controller.

**Fix:** E2 passed full native build/smoke/determinism and 16 paired evaluations, gaining +5,938.3125 mean alive frames with zero skill uses, but still 0 clears and 2 truncations. Implemented E3 native training pipeline with exactly 12k teacher demonstration decisions and 8k PPO fine-tuning decisions, source budget guards, dataset/model hashes, matched 16-seed evaluations, fail-closed no-skill checks and explicit launch trigger. Original model remains intact.

**Verification:** E2 authentic-game run 40213025937 succeeded. E3 has been written and awaits its own source preflight and actual Taisei execution. No E3 learned-policy results, native clears, or champion promotion are claimed yet.

**Next:** Run E3 with bounded CI, validate training/evaluation artifacts, inspect learned movement behavior and register a controlled E4 improvement only if justified by evidence.
