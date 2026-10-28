# Taisei native RL — Windows / WSL2

## Live demo from a trusted skill-policy `.pt` (w2ctrl)

**Single-control design:** There is only **one** supported rendered RL policy runtime: `rl_skill_v2` with `TSKILL2`, 546 input features, 18 movement actions + 3 skill actions, and `taisei_skill_mask_v2_respawn`. The 528-feature historical E4 model is **adapted to this format** without altering its trained movement behavior; it always selects skill `None`. New models trained with `scripts/train_torch_ppo.py` now use the full 546-feature skill-v2 design by default. See [LIVE_POLICY_CONTRACT.md](LIVE_POLICY_CONTRACT.md).

Update your existing **Windows workbench** checkout first. The prebuilt `rl_skill_v2/src/taisei.exe`, Meson environment and game resources must already exist locally; `git pull` does **not** download or rebuild them.

```powershell
# At the repository root
git switch feat/native-rl-workbench
git pull --ff-only origin feat/native-rl-workbench
cd my_test

# Install dependencies into the Python used for exporting the model (once)
py -3.9 -m pip install torch numpy

# Convert a trusted compatible experiment checkpoint, launch Taisei, and audit new logs
py -3.9 .\scripts\w2ctrl.py demo "C:\path\to\new-experiment.pt" --renderer gl33

# Optional: conversion and validation ONLY, without launching the game
py -3.9 .\scripts\w2ctrl.py prepare "C:\path\to\new-experiment.pt" .\results\new-experiment_TSKILL2.bin

# Optional: run the local source tests (no gameplay is performed)
py -3.9 -m unittest discover -s .\tests -p test_w2ctrl.py -v
```

This invokes the branch's canonical `export_live_skill_ppo_v2.py`, validates both movement and skill heads, exports a SHA-bound `TSKILL2` binary under `results/w2ctrl/`, launches the existing Windows Taisei executable, and audits the fresh game log. **Enter a stage via the game menu**. A game log establishes controller acceptance only; it does not establish a playable replay, visual inspection, or improved gameplay competence. Results and audit files are kept under `results/w2ctrl/` by default. The workflow does not compile or install the game.

**Compatibility gate:** Native 546-input/18+3 skill-v2 checkpoints export directly. Historical **528-input E4 movement-only `.pt`** checkpoints are accepted only through a verified 528→546 zero-padding adapter that fixes Bomb/Special to `None`, with source and derived checkpoint hashes recorded. The game always loads `TSKILL2`: there is **no second live controller**. Unsupported architectures or mask versions fail closed. The original E4 checkpoint is unchanged. For optional provenance, use `--expected-sha256 EXPECTED_DIGEST`. Never load an untrusted `.pt` (PyTorch serialization can execute code).

See [W2CTRL.md](W2CTRL.md) for export and live-log audit, [LIVE_POLICY_CONTRACT.md](LIVE_POLICY_CONTRACT.md) for the single controller contract, and [RESOURCE_RECOVERY.md](RESOURCE_RECOVERY.md) for missing renderer atlas diagnostics.

**Replay integrity (observed failure on 2026-10-28):** Do not treat a `Saving ...tsr` log line or a distinct replay filename as proof of recorded gameplay. Two separately named files from local runs were **identical 44-byte TSR headers with empty compressed metadata and events**. `w2ctrl.py` now checks any saved replay attached to its fresh launcher manifest independently of the policy-loaded/decision log. If replay integrity fails, the CLI returns a nonzero status even when logged movement control was attributed. Test any exported file explicitly:

```powershell
py -3.9 .\scripts\w2ctrl.py replay-check 'C:\path\to\recording.tsr'
```

`invalid_empty_metadata_and_events` is a **recording failure**, not proof of bad gameplay. Save `%APPDATA%\taisei\log.txt` and the `taisei-live-ppo-*.json` manifest under `%LOCALAPPDATA%\hermes\cache\scratch` for debugging; preserve real MP4/PNG footage as a separate observable gameplay source. **User-observed E1 issue:** dodging and item collection are weak. Source inspection shows native item positions exist in the environment but are omitted from the 546-feature model vector, with no explicit item-pickup reward; see the research-branch failure diagnosis. A future item-aware experiment must version its feature semantics and update native/rendered encoders together while retaining the one `TSKILL2` live controller. Do **not** silently reinterpret existing E1 weights.

**Your two local checkpoint paths — same command, same renderer:**

```powershell
# E1 skill-aware checkpoint (direct TSKILL2 export)
py -3.9 .\scripts\w2ctrl.py demo '.\res\ckpt\Taisei_E4_2026-10-28\models\e1_challenger_seed48020.pt' --renderer gl33

# E4 historical movement-only checkpoint (adapted to skill-v2; skills disabled)
py -3.9 .\scripts\w2ctrl.py demo '.\res\ckpt\Taisei_E4_2026-10-28\models\e4_weighted_bc_seed48030.pt' --renderer gl33
```

**If Taisei reports `Required atlas portraits could not be loaded`:** the error is a game **resource initialization failure**, not a model architecture failure. The launcher now checks the atlas before starting. Run `py -3.9 .\scripts\taisei_resource_doctor.py` to locate a valid compiled resource root; if necessary, supply `--resources 'C:\actual\matching\game\resources'` (replace with a verified real path). If none contains the atlas, rebuild/regenerate matching game assets first. The separate `vswhere.exe` warning did not cause the observed fatal atlas error. Do not copy assets from an unrelated game revision.

**Game choice:** open-source [Taisei](https://github.com/taisei-project/taisei) and the `taisei-sim` headless version of its native game logic; this is not a hand-made toy game.

**Platform note:** the upstream headless build documents GNU/Linux support; WSL2 Ubuntu is the optional route for headless training below. Separately, this local `my_test` workbench contains experimental Windows rendered-game builds and a PPO export. They are not automatically present in a fresh Git clone or remote collaboration copy, and do not establish official Windows headless support, parity, or performance.

**Fresh-clone scope:** a fresh Git clone or remote collaboration copy does not automatically include prebuilt game/simulator binaries, local build trees, virtual environments, assets, results, checkpoints, policy exports, replays, or logs. In this local workbench, `.venv-winbuild/Scripts/meson.exe`, `wb3/src/taisei.exe`, legacy `wb5_agent_demo/src/taisei.exe`, isolated `rl_skill_v2/src/taisei.exe`, and legacy and skill-v2 policy exports are present. The optional `patches/live-ppo.patch` is controller integration only; local artifacts are separate from that patch.

## Local Windows rendered-game runs (this workbench only)

Use these commands from the local `my_test` workbench. `wb3` is human-controlled, `wb5_agent_demo` is legacy movement-only, and `rl_skill_v2` is the current skill-capable demo build. Menus remain human-controlled until gameplay.

Human/manual testing with `wb3`:

```text
./.venv-winbuild/Scripts/meson.exe devenv -C ./wb3 ./wb3/src/taisei.exe --renderer gl33
```

Skill-v2 PPO control with `rl_skill_v2`:

```powershell
.\.venv-winbuild\Scripts\python.exe .\scripts\run_live_ppo_windows.py --renderer gl33 --rl-policy .\results\ppo_skill_survival1_maskv2_200k_seed48000_20261023_live.bin
```

**Policy selection:** `--rl-policy` is required and replaceable with another valid canonical `TSKILL2` mask-v2 export path. The legacy `results/ppo_live_export.bin` is a 528-input, 18-movement policy and is incompatible with this runtime.

**Verification:** the local GL33 log confirms checkpoint SHA `540ef892f52e8edd15a131e3c0f540e02db9f45a495cffa0aa9099b187068d21` and 5,461 in-game `RL decision` entries across 8 run records. Four decisions selected skills (3 Bomb, 1 Special), all accepted. This verifies policy loading and rendered control, not performance: the candidate recorded 0 wins in 16 development seeds and is not promoted. The local resource bundle still logs missing sound and postprocessing files; they did not prevent gameplay. The earlier direct `meson.exe` launcher exited silently, so this helper uses Meson's Python entry point and current runtime paths. These Windows builds remain local experiments, not upstream Windows headless support.

**Convert and run the local challenger checkpoint (this workbench only):** Run this from the local `my_test` directory in Windows Command Prompt. It exports the skill-aware `.pt` checkpoint to a canonical v2 `TSKILL2` policy, then starts the existing GL33 demo. These checkpoint/build files are local-only and are absent from a fresh clone.

```cmd
py -3.9 .\scripts\export_live_skill_ppo_v2.py .\res\Taisei_RL_Checkpoints_Local_Demo_2026-10-25\challenger_seed48020.pt .\res\Taisei_RL_Checkpoints_Local_Demo_2026-10-25\challenger_seed48020_live_v2.bin && .\.venv-winbuild\Scripts\python.exe .\scripts\run_live_ppo_windows.py --renderer gl33 --rl-policy .\res\Taisei_RL_Checkpoints_Local_Demo_2026-10-25\challenger_seed48020_live_v2.bin
```

The converter needs Python 3.9 with NumPy and PyTorch; `py -3.9` is the verified local interpreter. It replaces only the derived `.bin` export and leaves the source checkpoint unchanged. This 20,000-step Easy Stage 1 candidate is experimental and had 0/16 clears; this is not evidence of competent dodging or improved performance.

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
