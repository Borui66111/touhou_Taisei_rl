# Taisei W2CTRL: one canonical live-control policy, version 2

**Working branch:** `feat/native-rl-workbench` (local Windows live-demo integration). **Research authority:** `research/rsi-automation-v1`. **Date:** 2026-10-28.

## One runtime format, not two controllers

All local rendered Taisei demonstrations use **`rl_skill_v2` and `TSKILL2`**. The only live inference signature is:

| Field | Canonical value |
| --- | --- |
| Model family | `SkillActorCritic` |
| Observation | 546 floats = original 528 movement/threat features + 18 timing/skill features |
| Hidden layers | 128, Tanh, 128, Tanh |
| Heads | 18 movement logits and 3 skill logits (`None`, `Bomb`, `Special`) |
| Mask version | `taisei_skill_mask_v2_respawn` |
| On-disk live export | `TSKILL2\0`, canonical metadata + checksum + ordered `float32` weights |
| Game integration | Existing **rendered** Windows `rl_skill_v2/src/taisei.exe` with a matching skill-v2 loader |

The old 528-feature `TPPOv1` loader and its binary are *not* a second supported live-control design. Historical checkpoints are **adapted into this same TSKILL2 ABI** as explained below. No renaming, format-guessing, or substituting a toy simulator is allowed.

## Historical E4 528-feature checkpoint: exact movement embedding

Original E4 class-weighted behavior-cloning `.pt` is `ActorCritic` with 528 inputs, 128 hidden width and 18 movement actions. It **does not have skill training**. `w2ctrl.py demo` or `prepare` now recognizes this exact historical family and creates an **immutable derived** 546-feature checkpoint under `results/w2ctrl/adapted_checkpoints/`:

1. Copy the entire E4 movement/hidden/critic network, with the original 128×528 first-layer weights retained as the first 528 input columns.
2. Append **18 zero columns** to that first-layer weight matrix. The E4 movement logits therefore ignore the new timing/skill input features and remain equivalent for the same first 528 observations (verified by paired forward inference).
3. Add a fixed skill head with zero weights and biases `[0, -1000000, -1000000]`: it **always selects `None`**, even when the renderer allows Bomb/Special.
4. Mark the derived checkpoint with `adapter_kind=legacy528_prefix_zero_pad_skill_none_v1`, full source-checkpoint SHA-256 and a visible warning. The canonical skill-v2 exporter serializes this derived checkpoint into an ordinary valid `TSKILL2` file. The exported binary references the **derived** checkpoint hash; the W2CTRL conversion report includes both original and derived SHA-256 values.
5. Re-export is idempotent; an existing different binary or modified adapter is rejected. The original E4 `.pt` is never edited.

**Interpretation:** E4 now uses the same *game controller*, but it is still a movement-only **historical model**. This wrapper cannot create new skills or improve dodging by itself. The E1 skill-aware model exports directly and retains its original exact binary hash.

## Future experiments: enforce one trainable policy architecture

Run `my_test/scripts/train_torch_ppo.py` for **all new local native PPO models**. Its default and only accepted policy is `skill`, with the canonical mask-v2. The script no longer passes an unsupported `skill_action_mask_version` keyword to the pinned `torch_ppo.train` API. Inputs 546, 18 movement+3 skills, and mask identity are checked at export.

You may evaluate archival legacy PTs offline, but **do not train new legacy 528 policy experiments as if they were candidates for the current live-demo ABI**. A checkpoint made with incompatible model shape or mask version fails closed. Historical model adaptation is for demonstration/comparison only, not for resetting training, claiming skill learning, or promoting champion.

## Windows quick start (from `my_test`)

```powershell
# 1. Source updates. Does not download Windows game build/assets.
git switch feat/native-rl-workbench
git pull --ff-only origin feat/native-rl-workbench
cd my_test

# 2. Verify where this exact game build's atlas is (read-only).
py -3.9 .\scripts\taisei_resource_doctor.py

# 3. E1 full skill model, or E4 movement-only historical model. SAME command:
py -3.9 .\scripts\w2ctrl.py demo '.\res\ckpt\Taisei_E4_2026-10-28\models\e1_challenger_seed48020.pt' --renderer gl33
py -3.9 .\scripts\w2ctrl.py demo '.\res\ckpt\Taisei_E4_2026-10-28\models\e4_weighted_bc_seed48030.pt' --renderer gl33

# If the atlas exists in a known matching resource tree but is not auto-selected:
py -3.9 .\scripts\w2ctrl.py demo '.\res\ckpt\Taisei_E4_2026-10-28\models\e4_weighted_bc_seed48030.pt' --resources 'C:\actual\matching\game\resources' --renderer gl33

# 4. Export only (no Windows game needed), including for E4:
py -3.9 .\scripts\w2ctrl.py prepare '.\res\ckpt\Taisei_E4_2026-10-28\models\e4_weighted_bc_seed48030.pt' '.\results\w2ctrl\E4_TSKILL2.bin'
```

### Fatal `portraits.tsatlas.zst` initialization error

The `2026-10-28` user screenshot showed a **missing required atlas** in the resource root and **zero RL decisions**. The PyTorch checkpoint and E1 conversion were valid; Taisei did not initialize far enough to load the policy. The patched launcher now checks known candidate game resource roots **before launching**; `--resources` accepts a complete, matching compiled resource root when necessary. See [RESOURCE_RECOVERY.md](RESOURCE_RECOVERY.md). The `vswhere.exe` warning did not explain the atlas error. If all matching roots are missing the atlas, regenerate resources in the **same** Taisei build; do not copy unverified assets from another revision.

### Evidence required for a successful live demo

After the resource issue is resolved, require actual game logs proving policy-loaded and nonzero accepted decisions, a complete replay, successful replay loading in the same rendered game, and actual visual review of frames/behavior. The source-test and binary-export results **do not** claim those stages have passed.

### Verification commands

```powershell
py -3.9 -m unittest discover -s .\tests -p test_w2ctrl.py -v
py -3.9 -m unittest discover -s .\tests -p test_legacy_to_skillv2.py -v
py -3.9 -m unittest discover -s .\tests -p test_train_live_contract.py -v
py -3.9 -m unittest discover -s .\tests -p test_taisei_resource_doctor.py -v
py -3.9 -m unittest discover -s .\tests -p test_w2ctrl_asset_preflight.py -v
```

**Validation performed in assistant CPU container:** 68 source and archived-checkpoint tests passed. Actual Windows renderer/game not available here; no authentic replay or visually verified competence obtained.
