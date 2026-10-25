# Taisei RL: Windows-ready integration + PyTorch PPO (Phase 2)

**Windows user:** START HERE: [README_WINDOWS.md](README_WINDOWS.md).

**Status:** native game NOT built in this assistant environment, so real-game RL training has not occurred. See [EVIDENCE.md](EVIDENCE.md) and [STATE.md](STATE.md).

This update adds a PowerShell/WSL2 installer, pure PyTorch PPO, proper time-limit bootstrapping, mask-aware bullet observations, seeded evaluation, baseline planner, native spell discovery, reproducibility and replay scripts. All real-game work calls the native Taisei C API; there is no physics substitute.

---

# Native Touhou-style RL: Taisei integration (Phase 1)

Remote headless PPO quickstart and handoff: [my_test/README_REMOTE.md](my_test/README_REMOTE.md).

**Project objective:** RL evaluation and training on the actual, open-source Taisei game logic, not a handmade physics approximation. Target: full bullet-hell difficulty, boss offense, native grazing, collision safety, and reproducible evaluation.

## Status (2026-10-18)

- Upstream source/API inspected: `touhourl/taisei-sim` (`src/sim/taisei_sim.h`, `sim.c`, `runtime.c`) and `taisei-project/taisei`.
- First Python `ctypes` wrapper and an optional Gymnasium adapter implemented, with ABI size validation against the native library when loaded.
- Four interface tests passed using a **stub native interface**; **these do not validate gameplay**.
- Engine **not compiled or executed in the current environment**: outbound GitHub DNS resolution blocked; source not mounted. No claim of native game fidelity or RL outcomes has yet been made.
- Old toy PPO models **cannot simply be transferred** to this new game's state/action representation; retraining required.

## Setup on GNU/Linux (or WSL2, dependencies may vary)

Upstream's `taisei-sim` README currently documents `git clone --recurse-submodules` and `make`. Current source requires Meson >=1.8, Python >=3.7, SDL3 and other dependencies. Consult upstream `doc/BUILD.rst` for the OS-specific dependencies. The upstream project supports GNU/Linux; Windows native builds are not assumed to work.

```bash
# 0. From this project's directory; have git, C compiler, meson, ninja, Python available.
python scripts/doctor.py --check-dns
bash scripts/build_taisei_game.sh
bash scripts/build_taisei_sim.sh

# 1. Locate compiled shared library
find upstream/taisei-sim/build -name 'libtaisei_sim.so*'

# 2. ACTUAL native smoke test — replace the path with the one printed above
python scripts/smoke_native.py \
  --library "$PWD/upstream/taisei-sim/build/src/sim/libtaisei_sim.so" \
  --stage 1 --difficulty 2 --frames 600

# 3. Optional PPO: real Taisei game, NOT the toy simulator
python -m pip install gymnasium numpy stable-baselines3
python scripts/train_sb3.py \
  --library "$PWD/upstream/taisei-sim/build/src/sim/libtaisei_sim.so" \
  --stage 1 --difficulty 2 --steps 100000
```

**Note:** library path shown is a plausible Meson layout, not a verified binary path; use `find` output. Runtime may need resource data path; pass `--resource-path` to the smoke test or use `TAISEI_RES_PATH`. Incompatible runtime dependencies must be resolved before proceeding. The provided build helper may need platform-specific changes; it's not yet validated on a successful Taisei build.

## What is implemented

- `native_rl/taisei_ffi.py`: native ABI v1 structures, functions, runtime size guards, exact game state, projectiles, lasers, seeded stage reset, native step, native replay export.
- `native_rl/env.py`: explicit Gym-style reset/step API; action mapping (18 combinations of directions+focus) with auto shooting; per-frame damage, deaths, grazing, win metrics; closest-approach projectile ranking and laser samples as observation features.
- `native_rl/gym_adapter.py`: optional Stable-Baselines3-compatible `gymnasium.Env` and flattened observations.
- `scripts/build_taisei_sim.sh`: build-helper for upstream native game logic; upstream's source is fetched separately, not copied into this bundle.
- `scripts/build_taisei_game.sh`: builds the original rendered Taisei game separately for fidelity and visualization.
- `scripts/doctor.py`: readiness audit of local dependencies and native-library load.
- `scripts/smoke_native.py`: fail-fast REAL Taisei episode verification, with frame-level native observations.
- `scripts/train_sb3.py`: optional SB3 PPO example for **authentic native simulation once compiled**. Not yet executed.
- `tests/test_native_interface.py`: tests only adapter logic and fail-fast behavior, not native physics.

## Integrity / milestone gates

1. **Game exists:** successful compilation, screenshots/replay demonstrating the official rendered Taisei engine. **NOT MET**.
2. **Native simulator works:** loads ABI, starts actual stage, observes projectiles, returns consistent frames and native statistics. **NOT MET**.
3. **Fidelity cross-check:** repeated input sequences with matched game mechanics against the rendered engine, including collision/graze. **NOT MET**.
4. **Baseline real-game controllers:** random, geometric planner, human/playback on identical genuine stages. **NOT MET**.
5. **RL on real-game engine:** train and evaluate in authentic stages; separately track deaths, boss damage and native graze; held-out stages/spells. **NOT MET**.

**Prototype-exit contract:** This Python wrapper is integration code, not the game, and it is only a useful interim deliverable. No RL experiment counts as a real-game result until gates 1–3 are met. We should not repeat the earlier failure of treating the prototype as the project's destination.

## Observations / limitations

- Our initial Stage 1 episode includes stage content before a boss, so boss damage can remain zero for a long time. Spell practice is the appropriate next target after enumerating stage IDs from the real game.
- `frames_per_action=1` prioritizes responsive dodging but reduces training throughput; benchmarking will guide possible frame skipping.
- States hold only the highest-risk 64 projectiles and 16 laser samples, while the full bullet/laser counts remain visible. Dense-pattern truncation and collision-risk priority should be studied as explicit observation ablations.
- Native graze count is used rather than the old custom graze counter. **Perfect dodge** as an extra research metric must be explicitly defined separately.
- Strong policy performance in the toy game is not evidence of actual Touhou competence.
- Raw C ABI depends on the upstream header; if the header changes, current code fails its size checks rather than trying to use mismatched structures.
- Upstream library documents only one active simulation per process; vectorized training would require multiprocessing, not many native instances in one process.
- Headless simulation may differ from a rendered/full game; fidelity requires experiments, not assumption.

## Primary external sources

- Native fork: https://github.com/touhourl/taisei-sim
- Original full game: https://github.com/taisei-project/taisei
- Upstream headless API: https://github.com/touhourl/taisei-sim/blob/master/src/sim/taisei_sim.h
- Building: https://github.com/taisei-project/taisei/blob/master/doc/BUILD.rst
