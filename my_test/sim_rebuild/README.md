# Rebuilt `libtaisei_sim` (public Taisei + reconstruction patch)

The original simulator, `touhourl/taisei-sim@789200ff` (`my_test/UPSTREAM.lock`), has been
unreachable since 2026-10-26, including with the owner's authenticated GitHub account
(ledger T-20261126-62). This directory rebuilds the same simulator ABI from
**public Taisei** and pins everything needed to reproduce it.

| Pinned input | Value |
|---|---|
| Upstream | `https://github.com/taisei-project/taisei.git` @ `5668c41b41f7227d21fc80cf56997d4d30356ce7` (2026-10-15) |
| Submodules | basis_universal `380ef938`, gamecontrollerdb `7988b5e8`, koishi `5701d88c`, openlibm `8f5d88c2`: identical to the original build log |
| Patch | `0001-libtaisei_sim-taisei-sim-ABI-v1-reconstruction.patch` (sha256 in `UPSTREAM_PUBLIC.lock`) |
| SDL3 | `taisei-project/SDL` tag `taisei-3.4.12`, built shared by `research/rsi/build_native_with_shared_sdl3.sh` |

```bash
bash research/rsi/build_native_with_shared_sdl3.sh            # once; or any SDL 3.4.12 shared build
export PKG_CONFIG_PATH=<sdl prefix>/lib/pkgconfig
bash my_test/sim_rebuild/build_rebuilt_sim.sh                 # -> my_test/upstream/taisei-sim-rebuild
source my_test/upstream/taisei-sim-rebuild/rebuilt_sim.env    # TAISEI_SIM_LIBRARY / TAISEI_SIM_RESOURCES
python research/rsi/validate_rebuilt_sim.py --library "$TAISEI_SIM_LIBRARY" \
    --resource-path "$TAISEI_SIM_RESOURCES" \
    --taisei-exe my_test/upstream/taisei-sim-rebuild/build/src/taisei
```
Build dependencies: meson >= 1.8, ninja, gcc/clang, gettext (`msgfmt`), libpng, freetype,
libwebp, zstd, cglm, libunibreak, opusfile, openssl, and `backports.zstd` on Python < 3.14.
`native_rl.taisei_ffi`, `env.py` and every experiment script work unchanged. Pass the paths
above wherever they used the old `libtaisei_sim.so` and `resources`.

## What the patch does
* `src/sim/` adds `libtaisei_sim.so` with the exact ABI v1 of `native_rl/taisei_ffi.py`:
  14 structs with size checks, plus init/create/reset/step/get_state/save_replay. Each
  `step` runs N real stage logic frames; rendering never runs (as in `--verify-replay`).
* `src/sim_hooks.{c,h}` are inert unless the library enables them:
  * a virtual keyboard: press/release edges go through Taisei's own gameplay input handler
    and are recorded into the replay by the game itself;
  * an RNG seed override;
  * game over ends the episode instead of opening the game-over menu;
  * instant dialogue waits.
* `main.c` reads `TAISEI_SIM_SKIP_DIALOG_WAITS=1` so the normal executable can play back
  and verify replays recorded by the sim.

## Reverse-engineered conventions of the original
Each one was found where the archived trajectories first diverged, then confirmed on the
full archive:

| Behaviour | Original convention | Evidence |
|---|---|---|
| reset | runs the stage's first (input-free) logic frame; `logical_frame` = 1 | smoke log frame 1 = 3 projectiles; E3 progress log |
| projectile `velocity` | movement-parameter velocity (non-zero on the spawn frame) | demo step 269 |
| boss `phase_type` | `AttackType + 1` (0 = no current attack) | demo step 2699 |
| boss `remaining_timeout_frames` | `starttime + timeout - frame` (includes the start delay) | demo steps 2699, 4162 |
| dialogue | skipped instantly; the boss attack begins on the next frame | demo step 7440 |
| game over | game freezes for 14 steps (status RUNNING, `logical_frame` frozen), LOST on the 15th | demo steps 7842–7856; E4 `logical_frame` |
| upstream base | before the 2026-10-20 scoring batch ("convert full damage to score", etc.) | E4 per-episode `score` |

## Validation against the original simulator's archived evidence
`research/rsi/evidence/rebuilt_sim_validation_20261127.json`, built from a fresh clone with
this script:

* smoke (E3 and E4 logs), all fields: identical
* determinism (E3 and E4): identical (600 frames, max 26 projectiles)
* **12,000 archived E3 teacher decisions replayed open-loop: all 528-d observation vectors
  bit-identical** (max |Δ| = 0.0, including a game over and reset)
* E4 risk-aware teacher, closed loop, 16 seeds: **0 mismatches across 20 per-episode metrics**
* E2 heuristic + teacher, 2 × 16 seeds: all 8 published aggregates identical
* replay round trip: the sim's `.tsr` of the 7,857-step teacher episode passes the real
  game's `--verify-replay`, and desyncs without the dialogue rule (negative control)

## Limits
* The gameplay digest is a new FNV-1a hash; original digests cannot be reproduced and are
  never compared.
* Not covered by archived evidence: Bomb/Special input paths, laser geometry export,
  item/enemy fields, stages 2–7, characters other than Reimu A, difficulties other than
  Easy. Replay round trips across stages 1–7, all 6 player modes, all difficulties, with
  bombs and surge, verify cleanly in the real game. That is self-consistency, not equality
  with the original.
* Laser points export the quantized collision segments (`laser_sample_step <= 0`); the
  original's sampling rule is unknown.
* One simulation per process; not thread-safe.
