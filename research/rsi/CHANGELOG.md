# RSI Automation Changelog

## 2026-10-25 — Initial native pilot

**Issue:** Previous GitHub Actions smoke job failed before Taisei compiled: SDL3 CMake configuration could not find X11 or Wayland development libraries.

**Reason:** The isolated GitHub-hosted Ubuntu runner did not have the necessary Linux windowing headers for the SDL3 fallback dependency to configure, even though a headless simulator was the target.

**Fix:** Created a new isolated `research/rsi-automation-v1` branch and a fail-closed pilot workflow with Linux windowing build dependencies. Added a 300-frame real-native smoke gate, bounded skill-aware PPO training (1,024 decisions), a four-seed matched evaluation against random and heuristic baselines, and evidence artifacts.

**Validation required:** Build and native gameplay must complete in GitHub Actions. The training/evaluation stage has not yet been confirmed successful. No champion promotion or final-seed testing permitted.

**Known research issues:** Survival reward scale, Power Surge discharge mask behavior, and deployment parity need independent resolution before full RSI experiments.

## 2026-10-25 — Competence-first RSI design and offline evaluator

**Issue:** Reward improvements and apparent action diversity did not establish useful Taisei survival/dodging/collection. Native item state was omitted from PPO features and item pickup effects were not verified.

**Reason:** Existing reward/reporting tracks generic survival and combat but not robust competence; earlier 48 evaluation runs had no clear and were not fully comparable by checkpoint provenance.

**Fix:** Add `COMPETENCE_SPEC.md`, `competence_eval.py`, and nine automated regression tests. Formalize capability milestones, a safe research ladder, development/final seed separation, exact telemetry requirements and fail-closed comparison rules. This is scientific infrastructure, not evidence of successful training.

**Verification pending:** GitHub Actions evaluator tests; native simulator build and real training on the remote runner.

## 2026-10-25 — First native-only challenger launcher

**Issue:** Research goals were described but not executable as a controlled PPO challenger.

**Reason:** No isolated training arm and no output/provenance guard existed for a reward-only comparison.

**Fix:** Added `train_reward_ablation.py` with an unmodified control arm and a survival/clear-weight challenger arm; all other PPO/game choices are fixed. Added four tests, CLI preflight, immutable output safeguards and expanded CI. No fake native dynamics or automatic champion promotion.

**Verification:** Source-only GitHub Actions evaluator succeeded for the prior nine-test revision. The full 13-test revision and real native E1 training require separate verification.

## 2026-10-25 — Experiment trajectory and self-improving research governance

**Issue:** Sequential RL training changes were not connected by an immutable, provenance-aware scientific process history; early failures and pivots were difficult to reconstruct and a CI fix could be mistaken for gameplay learning.

**Reason:** Training-only recursion and unstructured logs do not capture hypotheses, experiment choices, failure mechanisms, decisions, or why higher-level strategy changed.

**Fix:** Added `RESEARCH_OS.md` with L0-L4 recursion, bounded default-continuation, marginal-gain evaluation and human gates. Seeded and corrected 11 evidence-labeled trajectory events, appended retroactive context without treating it as reexecuted data; implemented stdlib ledger validator, 10 regression tests, immutable byte-prefix CI, and machine-readable research digest. Enabled daily safe continuation separately from an existing hourly status watcher. Persisted a compact cross-project synthesis to the UKS.

**Verification:** CI run 40126793965 passed the initial 9-event/10-test trajectory audit. Newly appended 2 events and this changelog update must be checked by follow-up CI; native training still unverified due to SDL3 build failure.

## 2026-10-25 — Native build complete; two post-build command errors repaired

**Issue:** Easy native pilot 40128137506 passed all 618 Taisei build tasks but stopped before executing gameplay; the next attempt 40185187240 also stopped before smoke.

**Reason:** First `meson devenv` interpreted `--` as executable name; after removing the token Meson executed in the build directory, where relative `scripts/*.py` was absent.

**Fix:** On isolated `research/rsi-easy-curriculum-fix`, remove the stray separator, replace all four runtime script paths with absolute `$PWD/scripts/...` paths expanded by the shell, and add a source-level preflight assertion. Pending native integration verdict from GitHub run 40185526694. No PPO architecture, code semantics, or champion touched.

## 2026-10-25 — Power Surge early-discharge action-mask candidate

**Issue:** Published v2 action mask disallowed SPECIAL whenever Surge was already active, despite Taisei supporting deliberate early discharge.

**Reason:** It models SPECIAL only as activation, omitting its discharge transition.

**Fix:** Added separate mask v3 `taisei_skill_mask_v3_discharge` that permits SPECIAL while alive and surged (excluding Bomb-active state), retained an explicit historical v2 function and checkpoint-driven inference branch. Eleven numerical / checkpoint tests passed in GitHub Actions run 40185726369. Native effectiveness remains unproven; candidate is not merged into the active training path.
