# W2CTRL — verified skill-policy export and guarded Windows live control

**Source comparison:** `Borui66111/touhou_Taisei_rl` `main` at `fd0376f6ca2e63071bb1d0ef07780f349b670614` (2026-10-28), integrated into canonical `research/rsi-automation-v1` without editing `main`.

The actual `main` branch includes `my_test/scripts/export_live_skill_ppo_v2.py` (Git blob `c54158611b5e0a145620733fc24e99612b3e5329`) and `my_test/scripts/run_live_ppo_windows.py` (Git blob `9280be78511a389e407c71684ce34efce63193a0`). These supersede the earlier assumption that the `TSKILL2` binary schema was unknown. W2CTRL **uses the same exporter and launcher**, not a guessed format or toy engine.

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
| Existing E4 model | 528-input legacy `TPPOv1`; **rejected** by skill-v2 exporter and should not be loaded in `rl_skill_v2` |

## How to run on your Windows Taisei workbench

Use your working local `my_test` directory, where `rl_skill_v2/src/taisei.exe`, resources, `.venv-winbuild`, and toolchains are actually present. A fresh clone **does not include** game build/assets.

Copy the included trusted checkpoint and scripts into the corresponding paths, or use them from the source branch after the new scripts are committed.

```powershell
# From the repository's my_test directory. Use the Python installation with torch + numpy.
py -3.9 -m pip install torch numpy

# Verify model SHA, export with the actual main exporter, and check both heads.
py -3.9 .\scripts\w2ctrl.py prepare ".\models\e1_challenger_seed48020.pt" ".\results\e1_challenger_seed48020_TSKILL2.bin" --expected-sha256 be320e7764b236b6d2b8c42eb26524fdd57345fe2182cc916ca53f0cdf8d4478

# Optional dry-run of the existing main-branch game launcher: does NOT play the game.
py -3.9 .\scripts\run_live_ppo_windows.py --renderer gl33 --rl-policy ".\results\e1_challenger_seed48020_TSKILL2.bin" --dry-run

# ACTUAL rendered game launch + independent fresh-log attribution check:
py -3.9 .\scripts\w2ctrl.py run ".\results\e1_challenger_seed48020_TSKILL2.bin" --renderer gl33

# Or use a single PowerShell helper (export only unless -Launch is specified):
.\scripts\RUN_W2CTRL_WINDOWS.ps1 -Checkpoint ".\models\e1_challenger_seed48020.pt" -Launch
```

The launcher opens the real game; **enter a stage through the menu**. `w2ctrl.py run` reads only the new manifest created by the real Windows launcher. It distinguishes *policy loaded and made decisions* from *process exit 0*. If logs rotate, no fresh decisions are attributed, SHA identities differ, or a sealed evaluation seed appears, it reports **not verified**.

### Why a separate W2CTRL verifier?

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

The 29 portable tests generate their own **synthetic model fixture**. For the *actual* archived E1 checkpoint test, set `TAISEI_E1_PT` to the trusted local checkpoint path first; this adds one integration test with the exact expected binary SHA (30/30 passed in the local full test).

Source tests do not establish Windows game integration; do not mark the full game pipeline complete until an actual run and visual verification have been inspected.
