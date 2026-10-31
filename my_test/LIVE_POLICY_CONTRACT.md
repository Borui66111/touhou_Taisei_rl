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

The old 528-feature `TPPOv1` loader and its binary are *not* a second supported live-control design. Only the specifically allowlisted `survival_first` source may be adapted into this same TSKILL2 ABI as explained below. No renaming, format-guessing, or substituting a toy simulator is allowed.

## Allowlisted 528-feature legacy checkpoint

Live legacy adaptation is restricted to `survival_first_seed48020.pt`, SHA-256 `5526812cb58988849165389db9eec50add78d0dfc41840f5eba1ecc491b93b8`, with sibling `survival_first_seed48020_manifest.json` matching that checkpoint digest and code SHA `07ad672a595093107277b2e1fa398e65c36b6b64`. The approved source code contract is `features()` for the 528-feature prefix, followed by `skill_features()` appending 18 timing+skill features (`native_rl/torch_ppo.py:61-71`). W2CTRL records that provenance in its conversion report, preserves movement weights with an 18-column zero extension, and fixes the skill head to `None`.

The E4 source SHA-256 `06c344d1c9415d1277e89c7d7c1a4519f274952f159e1cc5095d7fecf696e6fc` is blocked for live use until feature-order provenance and observation parity are proven. Do not treat an arbitrary 528-input checkpoint as equivalent or live-ready.

## Future experiments: enforce one trainable policy architecture

Run `my_test/scripts/train_torch_ppo.py` for **all new local native PPO models**. Its default and only accepted policy is `skill`, with the canonical mask-v2. The script no longer passes an unsupported `skill_action_mask_version` keyword to the pinned `torch_ppo.train` API. Inputs 546, 18 movement+3 skills, and mask identity are checked at export.

You may evaluate archival legacy PTs offline, but **do not train new legacy 528 policy experiments as live candidates**. A checkpoint made with incompatible model shape or mask version fails closed. The single manifest-verified `survival_first` adaptation is for demonstration/comparison only, not for resetting training, claiming skill learning, or promoting a champion.

## Windows quick start (from `my_test`)

`w2ctrl.py` is the only public live-control entrypoint. It produces canonical `TSKILL2` and invokes the existing `rl_skill_v2` runtime internally. The resource root configured for that build is retained when valid; if it lacks the required atlas, pass a verified matching root explicitly with `--resources`.

```powershell
# Native E1 skill-v2 checkpoint
py -3.9 scripts/w2ctrl.py demo res/ckpt/Taisei_E4_2026-10-28/models/e1_challenger_seed48020.pt --renderer gl33

# The single allowlisted legacy checkpoint; requires its matching sibling manifest
py -3.9 scripts/w2ctrl.py demo res/Taisei_RL_Checkpoints_Local_Demo_2026-10-25/survival_first_seed48020.pt --renderer gl33

# If the configured root lacks the atlas, choose a verified matching root
py -3.9 scripts/w2ctrl.py demo res/ckpt/Taisei_E4_2026-10-28/models/e1_challenger_seed48020.pt --resources C:/actual/matching/game/resources --renderer gl33
```

`train_torch_ppo.py` is a separate model-training tool, not a live-control command. W2CTRL refuses E4 pending feature-order provenance and observation parity.

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
