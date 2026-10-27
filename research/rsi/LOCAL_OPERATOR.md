# Local-first Taisei RSI — executable operator (2026-10-28)

**Research branch:** `research/rsi-automation-v1` (read `BRANCH_POLICY.md` first).  
**Status:** Archived E3/E4 native-result auditing and E6 offline BC work are reproducible locally; fresh native Taisei gameplay is **BLOCKED** until exact pinned source and toolchain are available. No GitHub Actions is required or invoked by the local operator.

## Recover the saved self-contained offline bundle

For a fresh ephemeral ChatGPT execution environment, locate and materialize `Taisei_RSI_Local_Automation_v2_2026-10-28.zip` from the user's Library folder `/AI/Taisei_RSI_Local_Automation_2026-10-28/`. Extract its single `taisei_local/` root safely; validate `MANIFEST_SHA256.json` before trusting the archive. The existing directory is not persistent across all scheduled runs. Do not claim a background simulator is running when it is not. The branch contains the operator source and source-only tests, but **not** the complete native simulator.

## Tested commands

```bash
cd taisei_local
PYTHONPATH=research/rsi python -m unittest discover -s research/rsi -p 'test_local_native_operator.py' -v
PYTHONPATH=my_test python -m unittest discover -s research/rsi -p 'test_e6_episode_holdout.py' -v
python research/rsi/local_native_operator.py doctor --root "$PWD"                # exit 2 if native prerequisites unavailable
python research/rsi/local_native_operator.py audit --root "$PWD"                 # exit 0 for SHA-checked archived E3/E4 native episode evidence
python research/rsi/local_native_operator.py run --root "$PWD" \
  --out "$PWD/results/local_operator/unique_experiment_id" --attempt-native      # safe E5 native replay if ALL gates are ready, otherwise exit 2 with JSON diagnosis
python research/rsi/local_native_operator.py replay-preflight --input /real/game/replay.tsr
python research/rsi/local_native_operator.py video-preflight --input /real/gameplay.mp4 \
  --out "$PWD/results/visual_review_unique_run"                                 # validates decoded video and extracts true video frames, does NOT make a visual judgement
```

`doctor` requires the exact pinned Git checkout, original Git origin, exact SHA-1 commit, clean working tree and recursively initialized submodules. A mere ZIP of game files is not sufficient to authenticate original revision. When a full canonical repo is present, it also checks the branch, original Python native FFI (not offline shim), native library, Meson and game resources.

The `run` action does **not** blindly retrain E3, which already completed and showed a matched-seed PPO regression. Instead it audits preserved native E3/E4 results and, when feasible, compiles/loads the pinned engine and runs the preregistered E5 two-process no-training replay on development seeds `42132` and `42133`. It uses single-run lock files, exclusive output directories, a 60-minute overall deadline and a persistent success registry to avoid repeat evaluation, and it never changes the champion. No experiment can claim rendered visual evidence without an actual playable native replay/video and timestamped image inspection.

## Native source recovery gate

Required checkout: `my_test/upstream/taisei-sim-pinned` at `789200ff255c681b3ed4230a3f26f0b8c12c78fe` from `https://github.com/touhourl/taisei-sim.git`, with all original submodules/resources and source history. In this local container, `github.com` DNS resolution and authenticated fork access remain unavailable. `taisei-project/taisei` has a different ABI and must not substitute for the pinned simulator. Original upstream cannot be reconstructed from checkpoints or recorded episode summaries. A complete authorized Git copy or bundle of **all nested submodules** is necessary; avoid setting a fake remote URL just to pass the origin check.

When the full repo is available locally, install Meson >=1.8, Ninja, CMake, a shared SDL3 package exposed through pkg-config, game build headers/libs and `backports.zstd` from an authorized local system package/wheel cache. The operator uses the existing canonical `my_test/scripts/build_taisei_sim.sh` after checking exact source identity and the local SDL3 dependency. It does not invoke or fall back to GitHub Actions. Build failures are kept in a versioned local log and never interpreted as a policy failure.

## Post-E5 research loop

1. Verify E5 complete action/state determinism and terminal events on matched development seeds.
2. Integrate an authentic rendered Taisei policy loader/replay capture; retain checkpoint and loader hashes and **actual** video or PNG frames. Validate non-empty TSR and video, decode/extract frames, have the agent visually inspect collisions, bullet paths, boundary pinning and item pickups; see `REPLAY_VERIFICATION.md`. A TSR containing only inputs is not a video.
3. Use E3/E4/E6 analysis to plan a *new* E7 intervention focused on multiple independent teacher episodes and rare hazard-context movements. Existing E3 12k demonstration frames span only one full and one partial episode; more class-weight-only sweeps have low marginal return.
4. Train one bounded authentic challenger (<=20,000 **training** native decisions, <=60 minutes), preserve new checkpoint SHA-256, run matched native seeds with baseline and capture real game footage. Analyze uncertainty and failure clips; no automatic promotion.
5. Record hypothesis → intervention → evidence → visual observations (or `NOT_PERFORMED`) → interpretation → decision → next action in append-only trajectory. If source or capture genuinely cannot be recovered, report the exact blocker instead of repeating failed clones or requesting the human to play the game.

**Verification distinctions:** archived native evidence arithmetic passed ≠ new native gameplay; offline E6 CPU learning passed ≠ improved game competence; valid TSR header ≠ playable replay; extracted video frames ≠ visual inspection; verified game footage ≠ statistical gameplay improvement. The operator is source-tested; the restored-source native branch is **not tested in this session** and cannot be promised bug-free.
