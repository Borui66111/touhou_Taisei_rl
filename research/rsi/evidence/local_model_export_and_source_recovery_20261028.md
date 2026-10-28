# Local model delivery and authentic Taisei source-recovery diagnosis — 2026-10-28

**Branch:** `research/rsi-automation-v1`. **Evidence tier:** model serialization/export and local-source diagnosis **only**. There was **no fresh native Taisei gameplay, no replay playback, no rendered video, no retraining, and no model promotion in this session.**

## Recovered immutable PT checkpoints

| File | SHA-256 | Architecture | Provenance |
| --- | --- | --- | --- |
| `e1_challenger_seed48020.pt` | `be320e7764b236b6d2b8c42eb26524fdd57345fe2182cc916ca53f0cdf8d4478` | skill, 546 inputs, 18 movement + 3 skills, width 128, 20,000 PPO decisions | E1 archived native Actions artifact |
| `e1_control_seed48020.pt` | `5cf7eb84b6a8c56825c25527162691179c057850963e38178d4dbe3fe51d995f` | same skill family | E1 control |
| `e4_weighted_bc_seed48030.pt` | `06c344d1c9415d1277e89c7d7c1a4519f274952f159e1cc5095d7fecf696e6fc` | legacy, 528 inputs, 18 movement actions, width 128 | E4 archived native Actions artifact |
| `e4_plain_bc_seed48030.pt` | `87b43918d0181355435f105e3fcec44d7fd314c12aec6978b63ec33e9955cf54` | same legacy family | E4 original BC |

All four archived checkpoint file hashes were verified. The E1 skill checkpoint loads with PyTorch's constrained `weights_only=True` with explicit NumPy RNG-state dtype globals, and inference outputs are finite. The E4 legacy file loads with `weights_only=True` and strict dimensions.

## Exported movement-only live C policy

`e4_weighted_bc_seed48030_TPPOv1.bin`: SHA-256 `af96b9f2e1fef76f9bc6a0a617cf296841fbf3861870d9b8a6d4b02a05ef3b13`, 346,236 bytes, exact `TPPOv1\x00\x00` format and embedded E4 source PT SHA-256. Exported 528→128→128→18 actor layers in legacy C layout. PyTorch versus exported binary inference agreed on all 10 input-vector checks (max absolute logit deviation under 1e-6 in the explicit sample run). **This is not TSKILL2 and cannot be used in the skill-v2 rendered launcher.**

The existing `my_test/patches/live-ppo.patch` hardcodes both the legacy accepted binary SHA and checkpoint identity; the user-facing package supplies `patch_legacy_loader_pin.py` to preview / opt in to replacing exactly those two pins *on an existing compatible, authentic patched C source* with automatic backup and a rebuild requirement. Neither a source patch nor export establishes gameplay or loader acceptance until the actual rendered game logs `RL policy loaded` and nonzero decisions.

## Exact simulator diagnosis

- Expected origin: `https://github.com/touhourl/taisei-sim.git`.
- Locked commit: `789200ff255c681b3ed4230a3f26f0b8c12c78fe`.
- In the temporary assistant container, `git ls-remote` failed with `Could not resolve host: github.com`, exit 128; no complete local checkout, recursive submodule tree, or compiled shared library was found.
- The GitHub connector's direct repository lookup returned 404 / inaccessible; cached search indexes still mention the upstream fork as formerly public. Do **not** infer that the repo is definitely deleted or intentionally private; access is unresolved.
- Native reproducibility requires a complete exact-commit Git checkout with origin identity, recursive submodules/resources and compatible shared SDL3/Meson build. `taisei-project/taisei` original rendered game does not replace the `touhourl/taisei-sim` native C-ABI fork for historic E1–E6 comparability.
- A verified Windows rendered executable may be available **only on the user's local workbench**, not in this container. Original E1 `.pt` family is skill-v2 but the `TSKILL2` export specification is absent from the canonical repository, so an export cannot be synthesized or declared compatible.

## Local reproducibility package

`Taisei_PyTorch_Models_Local_Test_2026-10-28.zip`, SHA-256 `4f77612e0ff66cd57931e6a3e84f6d070f7c77312587a9b6ff986f6d8f65cf76`, 2,837,535 bytes, local sandbox artifact. Includes four original PT checkpoints, legacy TPPOv1 binary, `checkpoint_tools.py`, `simulator_doctor.py`, `patch_legacy_loader_pin.py`, `test_local_checkpoint_package.py`, readme, reports and SHA manifest. **16/16 tests passed** after independent ZIP extraction, all 13 content hashes verified. Simulator doctor correctly returned `missing_checkout` and `dns_or_network_unavailable` with exit 2, *without running a game or falsely reporting replay success*.

## Historical measurements are not fresh gains

E1 challenger 4,477 vs control 3,454 mean alive frames on 16 Easy development seeds; neither cleared, and comparability is expressly `DESCRIPTIVE_ONLY`, with frequent Bomb usage near nine. E4 weighted BC vs standard BC +81.9375 mean alive frames on matched 16 seeds with confidence interval crossing zero, no clears. Neither establishes improved neural dodging or championship.

## Next bounded authentic gate

Recover authorized original Git checkout from a source the worker can access (including submodules), validate source pin, compile/load genuine `libtaisei_sim.so`, run local 300-frame smoke and two-pass development-seed replay, check checkpoint format for actual native loader, record genuine rendered Taisei output using matching version/seed and validate live action logs and replay/video; inspect actual frames for dodging and wall pinning. Never touch sealed seeds 43000–43099 or auto-promote models. Do not launch GitHub Actions to bypass missing local dependencies.
