# Remote headless Taisei PPO worker guide

This workbench trains and evaluates PPO against the native headless Taisei simulator. It is intended for GNU/Linux or WSL2 Ubuntu; the upstream headless target is not a Windows-native build.

## Fresh clone and build

From a GNU/Linux or WSL2 Ubuntu terminal:

```bash
git clone --branch feat/native-rl-workbench https://github.com/Borui66111/touhou_Taisei_rl.git
cd touhou_Taisei_rl/my_test
bash scripts/setup_ubuntu_wsl.sh
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
source .native_paths
```

The bootstrap installs native build prerequisites, creates `.venv`, and invokes `scripts/build_taisei_sim.sh`. `UPSTREAM.lock` pins `touhourl/taisei-sim` at `789200ff255c681b3ed4230a3f26f0b8c12c78fe`; the helper clones that exact revision with recursive submodules and compiles the headless library locally. Upstream source and build binaries are not vendored. The build and resource roots are recorded in the generated `.native_paths` file. The bootstrap also runs a native smoke check.

## Smoke, tests, train, evaluate

Run from `my_test`, with `.venv` active and the paths loaded:

```bash
source .native_paths
meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/smoke_native.py \
    --library "$TAISEI_NATIVE_LIBRARY" \
    --resource-path "$TAISEI_NATIVE_RESOURCES" \
    --stage 1 --difficulty 2 --frames 600

python -m pytest -q tests

meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/train_torch_ppo.py \
    --library "$TAISEI_NATIVE_LIBRARY" \
    --resource-path "$TAISEI_NATIVE_RESOURCES" \
    --stage 1 --difficulty 2 --steps 200000 --rollout-steps 512 \
    --seed 48000 --device cpu --policy-type skill \
    --out results/remote_worker_skill_200k.pt \
    --log results/remote_worker_skill_200k.jsonl

meson devenv -C upstream/taisei-sim-pinned/build -- \
  python scripts/evaluate_native.py \
    --library "$TAISEI_NATIVE_LIBRARY" \
    --resource-path "$TAISEI_NATIVE_RESOURCES" \
    --stage 1 --difficulty 2 --episodes 16 --seed 50000 \
    --policy ppo --model models/ppo_skill_survival1_unmasked_200k_seed48000_portable.pt \
    --out results/remote_worker_unmasked_seed50000_n16.jsonl
```

Choose new output paths before each run; training resumes require both the checkpoint output and JSONL path to be unused. To evaluate the mask-trained candidate instead, pass `--model results/ppo_skill_survival1_maskv2_200k_seed48000_20261023.pt` and a different fresh `--out` path. `evaluate_native.py` exposes `--stage`, `--difficulty`, `--practice`, `--max-frames`, `--model`, `--episodes`, `--seed`, `--policy`, `--sampled`, `--sampling-seed`, `--out`, and `--trace-out`; character and shot mode use the `EnvConfig` defaults (0/0) in this CLI.

`train_torch_ppo.py --init CHECKPOINT` loads policy weights into a new run only: steps, updates, optimizer, counters, and RNG start fresh. `--resume CHECKPOINT` restores model, optimizer, counters, and saved RNG state, and requires `--steps` greater than the checkpoint's step count. Resume also requires a fresh output checkpoint and log; `--resume-log` is only needed for legacy checkpoints missing cumulative counters. Resume comparison keeps environment semantics strict but ignores only machine-local library/resource/storage paths. `--init` and `--resume` are mutually exclusive.

## Code, checkpoints, evidence

- `native_rl/taisei_ffi.py`: native C API/FFI; `native_rl/env.py`: `EnvConfig`, observations and action mapping; `native_rl/torch_ppo.py`: standalone PyTorch PPO, checkpointing and evaluation.
- `scripts/setup_ubuntu_wsl.sh`, `scripts/build_taisei_sim.sh`, `scripts/smoke_native.py`, `scripts/train_torch_ppo.py`, and `scripts/evaluate_native.py`: setup/build and headless pipeline.
- `tests/`: adapter, PPO, resume and live-policy contract tests. These do not by themselves establish GUI or replay parity.
- Portable unmasked baseline: `models/ppo_skill_survival1_unmasked_200k_seed48000_portable.pt`; provenance: `models/ppo_skill_survival1_unmasked_200k_seed48000_portable.manifest.json`.
- Current mask-trained candidate: `results/ppo_skill_survival1_maskv2_200k_seed48000_20261023.pt`.
- Selected development evidence: `results/ppo_skill_survival1_candidate_seed42000_n4_20261023_resourcefix_resume1.jsonl`, `results/ppo_skill_survival1_candidate_seed42004_n12_20261023_resourcefix_resume2_dllfix1.jsonl`, `results/ppo_skill_survival1_maskv2_candidate_seed42000_n16_20261023.jsonl`, `results/ppo_skill_survival1_unmasked_infermaskv2_seed42000_n16_20261023.jsonl`, and `results/ppo_skill_survival1_unmasked_infermaskv2_seed42016_n16_20261023_retry1.jsonl`. These are selected development artifacts; the sealed holdout seeds 43000–43099 are already consumed and must not be reused for selection.

The mask-trained candidate is experimental and not promoted: on matched development seeds 42000–42015 it had 0 wins and lower mean survival and boss damage than the unmasked baseline. The inference-only mask ablation is a separate exploratory result and does not promote the mask-trained checkpoint. This headless bundle does not claim live GUI integration or rendered-replay parity; those remain separate validation work.
