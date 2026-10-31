# W2CTRL — verified skill-policy export and guarded Windows live control

**Integration:** This copy lives on `feat/native-rl-workbench`, explicitly requested for the existing local Windows workbench. The public entrypoint is only `w2ctrl.py`; it internally reuses the existing `TSKILL2` exporter and Windows launcher. No second runtime or toy engine is introduced.

The W2CTRL workflow internally uses the branch’s canonical exporter and launcher. This preserves the actual `TSKILL2` schema and rendered game runtime without a guessed format or toy engine.

## Verified contract

| Item | Source / result |
| --- | --- |
| Model checkpoint | Trusted E1 challenger `e1_challenger_seed48020.pt`, SHA-256 `be320e7764b236b6d2b8c42eb26524fdd57345fe2182cc916ca53f0cdf8d4478` |
| Architecture | SkillActorCritic, **546** inputs, **128** hidden width, **18 movement + 3 skill** logits |
| Skill mask | `taisei_skill_mask_v2_respawn` |
| Binary layout | `TSKILL2\0`, metadata byte count, payload byte count, 32-byte metadata SHA-256, canonical JSON metadata, float32 ordered actor/skill tensors |
| Exported challenger | `e1_challenger_seed48020_TSKILL2.bin`, **357,347 bytes**, SHA-256 `63e1acc2b27aa74bb766f079061b2f1b3e0e13097a4f885fcfc8396c14c53423` |
| Exported E1 control | `e1_control_seed48020_TSKILL2.bin`, SHA-256 `c7cc6d0e9238960f0a32a4a7ee190a117713d056918bdf484abc080e781100a2` |
| Numerical validation | 16 observation vectors per model, BOTH logits heads, argmax parity, max absolute difference under `5e-7`; **not C native loader parity** |
| Legacy compatibility | Only the allowlisted `survival_first` SHA plus matching sibling manifest may be adapted; historical E4 is blocked for live use pending feature-order provenance and observation parity |

## How to run on your Windows Taisei workbench

Use your working local `my_test` directory, where `rl_skill_v2/src/taisei.exe`, resources, `.venv-winbuild`, and toolchains are actually present. A fresh clone **does not include** game build/assets.

After updating the existing `feat/native-rl-workbench` checkout, the scripts are available in `my_test/scripts/`. Save the new trusted `.pt` checkpoint wherever convenient; no manual binary conversion, copied scripts, model-specific C patch, or hard-coded checkpoint SHA edit is required for the existing `rl_skill_v2` loader.

**Recommended: one command for any trusted compatible E1-style checkpoint** (from `my_test`):

```powershell
py -3.9 .\scripts\w2ctrl.py demo "C:\path\to\new_experiment.pt" --renderer gl33
```

This automatically verifies and exports the checkpoint into a digest-versioned `results/w2ctrl/` binary, launches the *existing* `rl_skill_v2` game, and writes a new audit JSON. Enter a game stage in the real game’s menu. W2CTRL accepts only **546-input, 18 movement + 3 skill actions, mask-v2** native checkpoints, plus the specifically allowlisted manifest-verified legacy source described below. It does not install or compile Taisei.

```powershell
# From my_test: use W2CTRL for every public live-control operation
py -3.9 scripts/w2ctrl.py demo models/e1_challenger_seed48020.pt --renderer gl33

# Optional conversion/validation only; no game launch
py -3.9 scripts/w2ctrl.py prepare models/e1_challenger_seed48020.pt results/e1_TSKILL2.bin --expected-sha256 be320e7764b236b6d2b8c42eb26524fdd57345fe2182cc916ca53f0cdf8d4478

# Launch and audit a previously prepared canonical policy through W2CTRL
py -3.9 scripts/w2ctrl.py run results/e1_TSKILL2.bin --renderer gl33
```

The launcher opens the real game; **enter a stage through the menu**. `w2ctrl.py run` reads only the new manifest created by the real Windows launcher. It distinguishes *policy loaded and made decisions* from *process exit 0*. If logs rotate, no fresh decisions are attributed, SHA identities differ, or a sealed evaluation seed appears, it reports **not verified**.

## One live controller and manifest-gated legacy adaptation

All rendered-game policies use one ABI: `rl_skill_v2` / `TSKILL2`, 546 floats, 18 movement + 3 skill logits, and mask `taisei_skill_mask_v2_respawn`. Native skill-v2 checkpoints export directly. Live legacy adaptation is restricted to `survival_first_seed48020.pt` with source SHA `5526812cb58988849165389db9eec50add78d0dfc41840f5eba1ecc491b93b8`, plus its sibling `survival_first_seed48020_manifest.json` matching that digest and code SHA `07ad672a595093107277b2e1fa398e65c36b6b64`. Provenance records that `features()` defines the 528-feature prefix and `skill_features()` appends 18 timing+skill features (`native_rl/torch_ppo.py:61-71`). Movement weights are embedded with zero padding and the skill head is fixed to `None`; the original model gains no skill competence.

E4 SHA `06c344d1c9415d1277e89c7d7c1a4519f274952f159e1cc5095d7fecf696e6fc` is blocked for live W2CTRL use until feature-order provenance and observation parity are proven. `train_torch_ppo.py` is a separate model-training tool; it is not a live-control entrypoint.

```powershell
# Native E1 skill-v2 checkpoint
py -3.9 scripts/w2ctrl.py demo res/ckpt/Taisei_E4_2026-10-28/models/e1_challenger_seed48020.pt --renderer gl33

# The only legacy checkpoint eligible for manifest-gated adaptation
py -3.9 scripts/w2ctrl.py demo res/Taisei_RL_Checkpoints_Local_Demo_2026-10-25/survival_first_seed48020.pt --renderer gl33
```

**Before launching:** the required atlas `portraits.tsatlas.zst` must exist in the configured resource root. If it does not, W2CTRL reports known candidates but will not select an alternate automatically. Choose a matching compiled resource root explicitly with `--resources PATH`; see [RESOURCE_RECOVERY.md](RESOURCE_RECOVERY.md). The observed fatal screenshot showed **0 loaded policies and 0 decisions**, a game startup failure, not a checkpoint failure. If no matching atlas is available, regenerate same-revision resources and do not claim the game ran.

The current `main` Windows launcher recognizes a **single** run sequence for its `attribution = matched` status. Its README documents **8** run records and **5,461** decisions in a local session. W2CTRL checks multiple independently ordered `RL run` / `RL decision` / `RL policy outcome` segments with a matching checkpoint SHA and avoids using historic log entries. Its tests use **synthetic log fixtures only** (including out-of-order loads, missing seeds, rejected decisions, and nonmonotonic frames); they do not prove the actual game ran here.

### Evidence levels and remaining gates

1. **Verified:** original `main` Python exporter format, checkpoint architecture and identity, byte-accurate tensor layout, both-head numerical parity (PyTorch vs NumPy), log-audit tests, safe refusal on invalid files.
2. **Worker-reported but not independently replayed here:** Windows `rl_skill_v2` rendered build logged accepted policy and actual decisions in a previous main-branch local session.
3. **Still required on your Windows machine:** actual live controller acceptance and nonzero decisions for *these new* E1 exports; authentic complete `.tsr`; rendered video or screenshots; inspection of player/bullet/skill behavior and comparison on matched **development** seeds.
4. **Not established:** C `TSKILL2` loader inference parity, visual replay inspection, and learned neural gameplay improvement. Replay-save log alone is not proof that the replay is complete/playable; opened visuals are distinct evidence.

The pinned `touhourl/taisei-sim` headless source is a blocker for *new headless training here*, but **not necessarily** a blocker for running the already-built local Windows rendered `rl_skill_v2` launcher. Do not substitute the old legacy `TPPOv1` loader patch for the skill-aware build.

**Safety:** do not use development scripts to access sealed final evaluation seeds `43000–43099`; never overwrite a champion; do not use a `--dry-run` as gameplay evidence. All experiments should preserve checkpoint hash, renderer game hash, fresh log/manifest, and any replay/video artifacts. `w2ctrl.py` does not create or alter gameplay physics, generate video, or launch GitHub Actions.

## Tests

From a fresh source checkout (CPU PyTorch, NumPy):

```powershell
py -3.9 -m unittest discover -s .\tests -p test_w2ctrl.py -v
```

The portable tests generate their own **synthetic model fixture**. For the *actual* archived E1 checkpoint test, set `TAISEI_E1_PT` to the trusted local checkpoint path first; this adds one integration test with the exact expected binary SHA (30/30 passed in the local full test).

Source tests do not establish Windows game integration; do not mark the full game pipeline complete until an actual run and visual verification have been inspected.
