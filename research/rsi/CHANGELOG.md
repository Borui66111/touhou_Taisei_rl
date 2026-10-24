# RSI Automation Changelog

## 2026-10-25 — Initial native pilot

**Issue:** Previous GitHub Actions smoke job failed before Taisei compiled: SDL3 CMake configuration could not find X11 or Wayland development libraries.

**Reason:** The isolated GitHub-hosted Ubuntu runner did not have the necessary Linux windowing headers for the SDL3 fallback dependency to configure, even though a headless simulator was the target.

**Fix:** Created a new isolated `research/rsi-automation-v1` branch and a fail-closed pilot workflow with Linux windowing build dependencies. Added a 300-frame real-native smoke gate, bounded skill-aware PPO training (1,024 decisions), a four-seed matched evaluation against random and heuristic baselines, and evidence artifacts.

**Validation required:** Build and native gameplay must complete in GitHub Actions. The training/evaluation stage has not yet been confirmed successful. No champion promotion or final-seed testing permitted.

**Known research issues:** Survival reward scale, Power Surge discharge mask behavior, and deployment parity need independent resolution before full RSI experiments.
