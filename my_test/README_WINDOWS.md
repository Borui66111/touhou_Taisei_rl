# Taisei native RL — Windows / WSL2

**Game choice:** open-source [Taisei](https://github.com/taisei-project/taisei) and the `taisei-sim` headless version of its native game logic; this is not a hand-made toy game.

**Platform note:** the upstream headless build documents GNU/Linux support; WSL2 Ubuntu is the optional route for headless training below. Separately, this local `my_test` workbench contains experimental Windows rendered-game builds and a PPO export. They are not automatically present in a fresh Git clone or remote collaboration copy, and do not establish official Windows headless support, parity, or performance.

**Fresh-clone scope:** a fresh Git clone or remote collaboration copy does not automatically include prebuilt game/simulator binaries, local build trees, virtual environments, assets, results, checkpoints, policy exports, replays, or logs. In this local workbench, `.venv-winbuild/Scripts/meson.exe`, `wb3/src/taisei.exe`, `wb5_agent_demo/src/taisei.exe`, and `results/ppo_live_export.bin` are present. The optional `patches/live-ppo.patch` is controller integration only; local artifacts are separate from that patch.

## Local Windows rendered-game runs (this workbench only)

Use these commands from the local `my_test` directory; the PPO policy path is relative to it. `wb3` is the human-controlled build. `wb5_agent_demo` provides opt-in live PPO control: menus remain human-controlled until gameplay, when `--rl-policy` supplies the gameplay policy.

Human/manual testing with `wb3`:

```text
./.venv-winbuild/Scripts/meson.exe devenv -C ./wb3 ./wb3/src/taisei.exe --renderer gl33
```

Live PPO control with `wb5_agent_demo`:

```text
./.venv-winbuild/Scripts/meson.exe devenv -C ./wb5_agent_demo ./wb5_agent_demo/src/taisei.exe --renderer gl33 --rl-policy ./results/ppo_live_export.bin
```

The PPO command uses a relative path per user instruction; it is equivalent to the user-reported absolute-path command and was not independently rerun for this documentation update. These experimental rendered builds are local workbench artifacts, not a claim of upstream Windows headless support or rendered/headless parity.

## 1. Install WSL2 Ubuntu (optional)

This route requires an available WSL2 Ubuntu installation. It builds the headless simulator from the pinned source and is separate from any rendered Windows GUI build.

In **Administrator PowerShell**:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if requested. Open **Ubuntu** from the Start menu and finish creating your Linux username/password.

## 2. Run the prepared installer

Unzip the bundle on your Windows machine. From a normal PowerShell window opened at the extracted `my_test` folder (the bundle root):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_windows.ps1
```

This copies the project to `~/touhou-taisei-rl` inside WSL Ubuntu (faster than training under `/mnt/c`), installs the build dependencies, clones the entire original Taisei-sim repository with submodules, builds the actual native simulation library, and executes a **real native gameplay smoke test**. If a dependency or source build fails, stop and resolve that error; **do not substitute a toy simulator**.

The source build may need extra dependencies on your particular Ubuntu release, especially SDL3. See the upstream [`doc/BUILD.rst`](https://github.com/touhourl/taisei-sim/blob/master/doc/BUILD.rst). The installer is prepared but has *not been executed on an actual WSL2 machine in this chat*.

## 3. Inspect actual rendered gameplay

For a separately obtained official Windows release, download Taisei from:

https://github.com/taisei-project/taisei/releases

Run it on Windows and observe a boss fight, bullet density and native graze. Use the same game-version lineage wherever possible. Headless simulation may differ from normal rendering; verify gameplay and replay compatibility before claiming equivalence.

## 4. Train PPO on the real game

In the **Ubuntu** terminal:

```bash
cd ~/touhou-taisei-rl
source .venv/bin/activate
# CPU-only PyTorch for the starter run; CUDA acceleration can be enabled later.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
source .native_paths
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/train_torch_ppo.py \
    --library "$TAISEI_NATIVE_LIBRARY" \
    --resource-path "$TAISEI_NATIVE_RESOURCES" \
    --stage 1 --difficulty 2 --steps 200000 \
    --out results/stage1_ppo.pt
```

**Find a native boss spell-practice stage (recommended for first offense/dodging study):**

```bash
cd ~/touhou-taisei-rl
source .venv/bin/activate
source .native_paths
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/probe_spell_ids.py --library "$TAISEI_NATIVE_LIBRARY" \
  --resource-path "$TAISEI_NATIVE_RESOURCES" --origin-stage 1 --difficulty 2
# Replace --stage 1 above with a verified id=... from this output.
# Keep practice mode enabled for every run using that probed spell ID.
```

**Important:** Stage 1 is a complete stage and may spend many frames before the boss appears. For research targeting boss fights, enumerate verified spell-practice IDs before training. The model script refuses to train if the native library is absent. Training creates `results/stage1_ppo.pt` and JSONL learning logs, but **200,000 steps is not a promise of successful play**.

**Practice-mode consistency:** A probed spell ID requires `--practice` for every command that uses it; the flag is explicit and defaults to false. For example, set `SPELL_ID` to one of the IDs printed by the probe, then include `--practice` consistently:

```bash
SPELL_ID='REPLACE_WITH_VERIFIED_ID'
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/train_torch_ppo.py --library "$TAISEI_NATIVE_LIBRARY" \
    --resource-path "$TAISEI_NATIVE_RESOURCES" --stage "$SPELL_ID" --practice \
    --out results/spell_ppo.pt
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/evaluate_native.py --library "$TAISEI_NATIVE_LIBRARY" \
    --resource-path "$TAISEI_NATIVE_RESOURCES" --stage "$SPELL_ID" --practice \
    --policy ppo --model results/spell_ppo.pt
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/check_native_determinism.py --library "$TAISEI_NATIVE_LIBRARY" \
    --resource-path "$TAISEI_NATIVE_RESOURCES" --stage "$SPELL_ID" --practice
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/record_native_replay.py --library "$TAISEI_NATIVE_LIBRARY" \
    --resource-path "$TAISEI_NATIVE_RESOURCES" --stage "$SPELL_ID" --practice \
    --out results/spell_replay.tsr
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/train_sb3.py --library "$TAISEI_NATIVE_LIBRARY" \
    --resource-path "$TAISEI_NATIVE_RESOURCES" --stage "$SPELL_ID" --practice \
    --out results/spell_sb3.zip
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/smoke_native.py --library "$TAISEI_NATIVE_LIBRARY" \
    --resource-path "$TAISEI_NATIVE_RESOURCES" --stage "$SPELL_ID" --practice
```

Use a verified ID printed by the probe above; replace the placeholder before running. `--practice` defaults to false, so ordinary story-stage commands (including the Stage 1 example above) omit it. Every command using a probed spell ID includes `--practice`; do not mix modes for the same ID.

## 5. Evaluate fairly

```bash
cd ~/touhou-taisei-rl
source .venv/bin/activate
source .native_paths
for POLICY in heuristic random; do
  meson devenv -C upstream/taisei-sim-pinned/build -- \
    python scripts/evaluate_native.py \
      --library "$TAISEI_NATIVE_LIBRARY" --resource-path "$TAISEI_NATIVE_RESOURCES" \
      --policy "$POLICY" --episodes 20 --out "results/${POLICY}.jsonl"
done
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/evaluate_native.py \
    --library "$TAISEI_NATIVE_LIBRARY" --resource-path "$TAISEI_NATIVE_RESOURCES" \
    --policy ppo --model results/stage1_ppo.pt --episodes 20 \
    --out results/ppo_greedy.jsonl
```

All comparisons should use *identical seeds, character/shot mode, stage and difficulty*. Report victory, deaths, grazing, boss damage/phase clear, and survival, not only total reward. The current first-pass baseline planner is intentionally simple and not guaranteed strong.

## 6. Validation gates

1. Real native `.so` loaded with C/Python struct sizes matching.
2. Full native episode advances actual time, includes real projectiles and changes `gameplay_digest`.
3. Same input sequence and seed reproduces the headless state digests.
4. Actual-game rendered replay/control parity is measured, not assumed.
5. Train and evaluate PPO using native Taisei only, then compare to geometric baseline on unseen stage seeds.

**Bundle status:** A fresh Git clone includes code and workflow instructions but does not automatically receive this local workbench's native/game builds, policy export, checkpoints, or evaluation outputs. These local artifacts do not establish that gameplay, live control, or rendered parity has been validated.

## 7. Verify determinism and save a native replay

After the first successful native gameplay smoke test, run:

```bash
source ~/touhou-taisei-rl/.venv/bin/activate
cd ~/touhou-taisei-rl
source .native_paths
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/check_native_determinism.py \
    --library "$TAISEI_NATIVE_LIBRARY" --resource-path "$TAISEI_NATIVE_RESOURCES" \
    --stage 1 --frames 600 --seed 1402
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/record_native_replay.py \
    --library "$TAISEI_NATIVE_LIBRARY" --resource-path "$TAISEI_NATIVE_RESOURCES" \
    --stage 1 --seed 1402 --out results/native_agent_replay.tsr
```

For a probed spell ID, replace `--stage 1` with `--stage "$SPELL_ID"` and add `--practice` to **both** commands above; story-stage runs omit the flag.

If replay export succeeds, open the replay through the matching rendered Taisei build after verifying replay compatibility. The replay script deliberately requires native episode termination and does **not** fabricate video from a toy simulator.

## Optional live-controller source tests

Run only the project's test suite from this project directory with `PYTHONPATH=. pytest tests -q -ra -p no:cacheprovider`. The live-controller integration checks inspect a patched Taisei source tree and default to the local-only `windows-build/wlive_agent_demo/src` directory when it exists. That tree is excluded from a source-only clone. To use another patched source tree, set `TAISEI_PATCHED_SOURCE` to its `src` directory; relative paths are resolved from this project directory. If the required source files are absent, those source checks report as skipped with the missing prerequisite. The C/Python inference-parity check additionally requires an installed C compiler (or the optional local toolchain) and skips when none is available. These checks cover source contracts and inference numerics only; they do not validate native gameplay, rendered parity, or policy performance.
