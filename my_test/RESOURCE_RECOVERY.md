# Windows skill-v2 Taisei: missing `portraits.tsatlas.zst`

## Observed failure (2026-10-28)

`w2ctrl.py demo` correctly verified a trusted E1 skill checkpoint, exported canonical `TSKILL2` and passed numerical policy parity. Taisei then exited with `Required atlas portraits could not be loaded from ...portraits.tsatlas.zst`. Audit showed **0 policy loads, 0 runs, 0 decisions**. This is **game resource initialization failure**, not a trained-policy or checkpoint conversion error. The separate `vswhere.exe` environment warning is not the fatal asset error.

The old Windows launcher always set `TAISEI_RES_PATH` to `my_test/windows-build/wlive_agent_demo/resources`, even if that folder lacked a required generated atlas. The updated launcher checks the selected root before launching and considers a bounded set of known build roots, without mixing assets, downloading packages or changing the game.

## Recover on the existing Windows workbench

```powershell
# At your existing repository root
 git switch feat/native-rl-workbench
 git pull --ff-only origin feat/native-rl-workbench
 cd my_test

# Read-only resource check: look for the compiled portraits atlas or an archive containing it.
py -3.9 .\scripts\taisei_resource_doctor.py

# Search generated files by name, including build output subfolders.
Get-ChildItem .\rl_skill_v2,.\windows-build -Recurse -File -Filter portraits.tsatlas.zst -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty FullName
```

If the doctor finds **one** known matching resource root, the patched launcher selects it automatically. To supply a known matching resource root explicitly:

```powershell
py -3.9 .\scripts\w2ctrl.py demo ".\res\ckpt\Taisei_E4_2026-10-28\models\e1_challenger_seed48020.pt" --resources ".\rl_skill_v2\resources" --renderer gl33
```

Replace `--resources` with the actual directory containing `portraits.tsatlas.zst` at its root, or containing a valid Taisei ZIP package with that root-level member. The above `rl_skill_v2\resources` path is **an example, not a confirmed location on your PC**. If *no* directory or package has the atlas, fix the **same** game build's asset-generation/deployment process (for example rebuild the existing Meson game/resource targets and inspect the build output). Do not copy an atlas from an arbitrary Taisei version or swap engine builds just to bypass the failure. No game rebuild is needed to convert the `.pt` itself.

After resolving the resource failure, enter a stage through the real rendered-game menu, allow it to produce fresh `RL skill-v2 policy loaded`, `RL run`, `RL decision`, and outcome records, save an authentic playable `.tsr`, and inspect genuine game footage. A found atlas alone cannot certify complete resources or gameplay.

## Debug commands

```powershell
# Source-only conversion is independent of GUI assets:
py -3.9 .\scripts\w2ctrl.py prepare ".\res\ckpt\Taisei_E4_2026-10-28\models\e1_challenger_seed48020.pt" ".\results\w2ctrl\e1_manual_TSKILL2.bin"

# Additional read-only resource-root diagnosis:
py -3.9 .\scripts\taisei_resource_doctor.py --resources "C:\actual\matching\game\resources"

# Existing Taisei fatal logs:
Get-Content "$env:APPDATA\taisei\log.txt" -Tail 120

# Run source regression tests:
py -3.9 -m unittest discover -s .\tests -p 'test_taisei_resource_doctor.py' -v
py -3.9 -m unittest discover -s .\tests -p 'test_w2ctrl.py' -v
```

**Not established here:** we do not have access to the physical Windows workstation and cannot confirm which asset root it contains, rebuild its resource bundle, launch the executable, watch replay, or assert gameplay improvement. The updated code only fixes hard-coded wrong-directory selection and upgrades diagnostics.
