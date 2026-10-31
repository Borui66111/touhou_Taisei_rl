# Windows skill-v2 Taisei: missing `portraits.tsatlas.zst`

## Observed failure (2026-10-28)

`w2ctrl.py demo` correctly verified a trusted E1 skill checkpoint, exported canonical `TSKILL2` and passed numerical policy parity. Taisei then exited with `Required atlas portraits could not be loaded from ...portraits.tsatlas.zst`. Audit showed **0 policy loads, 0 runs, 0 decisions**. This is **game resource initialization failure**, not a trained-policy or checkpoint conversion error. The separate `vswhere.exe` environment warning is not the fatal asset error.

W2CTRL checks the configured resource root before launching. If it lacks the atlas, it reports known candidate roots but never auto-selects an alternate; choose a matching compiled root with an explicit `--resources` argument.

## Recover on the existing Windows workbench

```powershell
# At your existing repository root
 git switch feat/native-rl-workbench
 git pull --ff-only origin feat/native-rl-workbench
 cd my_test

# Public live-control path: W2CTRL keeps a valid configured root; otherwise
# choose a verified resource root for this exact game build explicitly.
py -3.9 scripts/w2ctrl.py demo res/ckpt/Taisei_E4_2026-10-28/models/e1_challenger_seed48020.pt --resources C:/actual/matching/game/resources --renderer gl33
```

W2CTRL reports known candidate roots when the configured root lacks `portraits.tsatlas.zst`; it never selects an alternate based on atlas presence alone. The explicit path must contain the atlas directly or a valid Taisei ZIP package with that root-level member. The example path is a placeholder, not a confirmed location on your PC. If no matching directory or package has the atlas, fix the same game build’s asset-generation/deployment process. Do not copy an atlas from an arbitrary Taisei version or swap engine builds.

After resolving the resource failure, enter a stage through the real rendered-game menu, allow it to produce fresh `RL skill-v2 policy loaded`, `RL run`, `RL decision`, and outcome records, save an authentic playable `.tsr`, and inspect genuine game footage. A found atlas alone cannot certify complete resources or gameplay.

## Debug commands

```powershell
# Optional policy validation/export only; this does not launch the game.
py -3.9 scripts/w2ctrl.py prepare res/ckpt/Taisei_E4_2026-10-28/models/e1_challenger_seed48020.pt results/w2ctrl/e1_manual_TSKILL2.bin

# Existing Taisei fatal logs
Get-Content "$env:APPDATA/taisei/log.txt" -Tail 120

# Source regression tests
py -3.9 -m unittest discover -s tests -p test_taisei_resource_doctor.py -v
py -3.9 -m unittest discover -s tests -p test_w2ctrl.py -v
```

**Not established here:** we do not have access to the physical Windows workstation and cannot confirm which asset root it contains, rebuild its resource bundle, launch the executable, watch replay, or assert gameplay improvement. This documentation describes W2CTRL’s explicit resource-selection guard; atlas presence alone does not verify a complete game build.
