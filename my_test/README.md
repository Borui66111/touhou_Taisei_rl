# Taisei RL: native integration + PyTorch PPO

The collaboration sources contain the Python/native interface, training and evaluation scripts, tests, and an optional live-controller integration patch. This local `my_test` workbench also has Windows rendered-game builds and a PPO policy export; those local artifacts are not automatically present in a fresh Git clone or remote collaboration copy.

**Pinned base:** [`touhourl/taisei-sim`](https://github.com/touhourl/taisei-sim) at `789200ff255c681b3ed4230a3f26f0b8c12c78fe` (`UPSTREAM.lock`). Apply [`patches/live-ppo.patch`](patches/live-ppo.patch) from that source tree's root. The patch is controller integration only; it contains no trained policy weights, checkpoint, or exported model. [`patches/COPYING.txt`](patches/COPYING.txt) carries the upstream notice.

**Fresh-clone scope:** upstream/game source, assets, toolchains, local build trees, virtual environments, native binaries, results/checkpoints, policy exports, replays, and logs are not automatically included in a fresh Git clone or remote collaboration copy. In this local workbench, `.venv-winbuild/Scripts/meson.exe`, `wb3/src/taisei.exe`, `wb5_agent_demo/src/taisei.exe`, and `results/ppo_live_export.bin` are present. Their presence does not establish gameplay parity or learned performance.

For Windows/WSL2 notes and the limits of the source bundle, see [README_WINDOWS.md](README_WINDOWS.md).

## Local Windows rendered-game runs

These are local workbench builds, not upstream Windows headless support. `wb3` is the human-controlled build; `wb5_agent_demo` enables opt-in live PPO control. The menus remain human-controlled until gameplay; `--rl-policy` supplies the gameplay policy.

Run the PPO command from the local `my_test` directory; its policy path is relative to that directory.

Human/manual testing with `wb3`:

```text
./.venv-winbuild/Scripts/meson.exe devenv -C ./wb3 ./wb3/src/taisei.exe --renderer gl33
```

Live PPO control with `wb5_agent_demo`:

```text
./.venv-winbuild/Scripts/meson.exe devenv -C ./wb5_agent_demo ./wb5_agent_demo/src/taisei.exe --renderer gl33 --rl-policy ./results/ppo_live_export.bin
```

From `my_test` in the moved workbench, this direct launcher exited 1 silently; stale Meson resource/toolchain paths were found. The one tested fallback reached Stage 1 then Game Over with resource warnings. The concise command above is unverified and does not establish PPO inference or performance. The manual `wb3` command above was not independently executed here.

---

# Native Touhou-style RL: Taisei integration (Phase 1)

**Project objective:** RL evaluation and training on the actual, open-source Taisei game logic, not a handmade physics approximation. Target: full bullet-hell difficulty, boss offense, native grazing, collision safety, and reproducible evaluation.

## Status

- Upstream source/API inspected: `touhourl/taisei-sim` (`src/sim/taisei_sim.h`, `sim.c`, `runtime.c`) and `taisei-project/taisei`.
- First Python `ctypes` wrapper and an optional Gymnasium adapter implemented, with ABI size validation against the native library when loaded.
- Historical stub/unit tests and source-level checks do not validate gameplay. The local workbench has rendered builds and a policy export, but native gameplay, replay/render parity, and learned performance require separate reproducible validation.
- Old toy PPO models **cannot simply be transferred** to this new game's state/action representation; retraining required.

## Setup on GNU/Linux (or WSL2, dependencies may vary)

Upstream's `taisei-sim` README currently documents `git clone --recurse-submodules` and `make`. Current source requires Meson >=1.8, Python >=3.7, SDL3 and other dependencies. Consult upstream `doc/BUILD.rst` for OS-specific dependencies. The upstream headless project documents GNU/Linux support. The local rendered Windows builds are experimental; they do not establish official Windows headless support, parity, or performance.

```bash
# 0. From this project's directory; have git, C compiler, meson, ninja, Python available.
python scripts/doctor.py --check-dns
bash scripts/build_taisei_game.sh
bash scripts/build_taisei_sim.sh

# 1. Locate compiled shared library
find upstream/taisei-sim-pinned/build -name 'libtaisei_sim.so*'

# 2. ACTUAL native smoke test — replace the path with the one printed above
python scripts/smoke_native.py \
  --library "$PWD/upstream/taisei-sim-pinned/build/src/sim/libtaisei_sim.so" \
  --stage 1 --difficulty 2 --frames 600

# 3. Optional PPO: real Taisei game, NOT the toy simulator
python -m pip install gymnasium numpy stable-baselines3
python scripts/train_sb3.py \
  --library "$PWD/upstream/taisei-sim-pinned/build/src/sim/libtaisei_sim.so" \
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

## Validation scope

The included adapter/unit tests and source checks do not establish native gameplay, rendered-game parity, or trained-policy performance. The presence of local rendered builds and a policy export does not establish those results either. A separate run must verify:

1. Native library loads and its ABI matches the Python structures.
2. A real native episode advances and produces expected state changes.
3. Deterministic input/seed traces and replay loading behave as intended.
4. Headless behavior matches the corresponding rendered game where claimed.
5. Policies are evaluated on fixed, held-out seeds against baselines.

The live-controller patch is integration code only. Local game binaries and the policy export listed above are workbench-only artifacts and are not automatically included in a fresh Git clone or remote collaboration copy; checkpoint and evaluation outputs are not established by their presence.


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
