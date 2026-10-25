# Explicit native E2 experiment request

- Experiment ID: E2_NATIVE_RISK_TEACHER_V1
- Requested: 2026-10-26, Asia/Singapore
- Canonical writable branch: research/rsi-automation-v1
- Preregistered code: research/rsi/risk_aware_teacher.py
- Baseline: existing native_rl.planner.heuristic_action
- Candidate: research/rsi/risk_aware_teacher.risk_aware_action
- Game engine: actual pinned Taisei native simulator from my_test/UPSTREAM.lock; toy fallback forbidden
- Task: Stage 1, Easy (native D_Easy=1), Reimu A, default full-stage conditions
- Development game RNG seeds: 42100 through 42115 inclusive; identical order/starting state
- Actions: movement only with auto-shoot, zero Bomb and zero Special use
- Workload: no PPO training, up to 16 × 2 native full-stage evaluations capped at 10,000 frames each
- Runner budget: 60-minute GitHub Actions maximum
- Native smoke/deterministic-reset gates: required before evaluation
- Compare: exact alive frames, deaths, stage clears, boss damage; paired seed bootstrap intervals
- Next decision: if safe teacher beats heuristic reproducibly, consider an **independent** behavior-cloning/teacher-guided PPO experiment. If not, revise trajectory/collision representation before further resource-intensive RL training.
- Safeguards: never use sealed seeds 43000–43099; never replace the champion; never infer learned dodging from a handcrafted controller or unit tests.
- Trigger: this file's creation triggers .github/workflows/rsi-e2-teacher.yml on the canonical branch.
