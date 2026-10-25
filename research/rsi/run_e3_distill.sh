#!/usr/bin/env bash
# E3: authentic Taisei demonstration 12k + PPO 8k; evaluate learned policies.
set -euo pipefail
cd "$(dirname "$0")/../../my_test"
OUT="$PWD/results/rsi_e3_distill_v1"
mkdir -p "$OUT"
{
  echo "experiment=E3_NATIVE_TEACHER_DISTILL_V1"
  echo "code_sha=${GITHUB_SHA:-unknown}"
  echo "run_id=${GITHUB_RUN_ID:-unknown}"
  echo "upstream=$(tr '\n' ' ' < UPSTREAM.lock)"
  echo "stage=1"
  echo "difficulty=1"
  echo "demonstration_native_decisions=12000"
  echo "ppo_native_decisions=8000"
  echo "max_train_native_interactions=20000"
  echo "training_seeds=48030_demo,48020_ppo"
  echo "eval_seed_range=42100-42115"
  echo "sealed_final_seeds=43000-43099_unused"
  echo "skill_space=18_movement_only_no_manual_Bomb_or_Surge"
} > "$OUT/run_manifest.txt"

export TAISEI_CI_SDL_PREFIX="${RUNNER_TEMP:-/tmp}/taisei-sdl3-prefix"
export PKG_CONFIG_PATH="$TAISEI_CI_SDL_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$TAISEI_CI_SDL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash ../research/rsi/build_native_with_shared_sdl3.sh 2>&1 | tee "$OUT/native_build.log"
ROOT="$PWD/upstream/taisei-sim-pinned"
LIB="$(find "$ROOT/build" -type f -name 'libtaisei_sim.so*' | head -n1)"
test -n "$LIB" || { echo "ERROR: authentic native simulator not found"; exit 1; }

meson devenv -C "$ROOT/build" python "$PWD/scripts/smoke_native.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --stage 1 --difficulty 1 --seed 42000 --frames 300 \
  2>&1 | tee "$OUT/native_smoke.log"
meson devenv -C "$ROOT/build" python "$PWD/scripts/check_native_determinism.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --stage 1 --difficulty 1 --seed 42000 --frames 600 \
  2>&1 | tee "$OUT/determinism.log"

meson devenv -C "$ROOT/build" python "$PWD/../research/rsi/e3_distill_native.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --outdir "$OUT" --demo-steps 12000 --ppo-steps 8000 \
  --demo-seed 48030 --ppo-seed 48020 \
  2>&1 | tee "$OUT/e3_training.stdout.log"

for policy in teacher_bc_seed48030 teacher_bc_ppo_seed48020; do
  meson devenv -C "$ROOT/build" python "$PWD/scripts/evaluate_native.py" \
    --library "$LIB" --resource-path "$ROOT/resources" \
    --stage 1 --difficulty 1 --max-frames 10000 \
    --policy ppo --model "$OUT/${policy}.pt" \
    --episodes 16 --seed 42100 --out "$OUT/${policy}_evaluation.jsonl" \
    2>&1 | tee "$OUT/${policy}_evaluation.stdout.log"
done
python ../research/rsi/e3_compare.py \
  --bc "$OUT/teacher_bc_seed48030_evaluation.jsonl" \
  --ppo "$OUT/teacher_bc_ppo_seed48020_evaluation.jsonl" \
  --manifest "$OUT/e3_manifest.json" --out "$OUT/e3_comparison.json" \
  | tee "$OUT/comparison.stdout.log"
