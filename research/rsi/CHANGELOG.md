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
