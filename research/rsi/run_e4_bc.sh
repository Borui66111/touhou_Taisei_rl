#!/usr/bin/env bash
# E4: reuse archived genuine 12k-game-decision E3 dataset. Zero new train interactions.
set -euo pipefail
cd "$(dirname "$0")/../../my_test"
SRC="$PWD/results/e3_artifact"
OUT="$PWD/results/rsi_e4_class_aware_bc"
mkdir -p "$OUT"
test -s "$SRC/teacher_demonstrations_12000.npz"
test -s "$SRC/teacher_bc_seed48030.pt"
{
  echo "experiment=E4_NATIVE_REBALANCED_BC"
  echo "github_sha=${GITHUB_SHA:-unknown}"
  echo "github_run_id=${GITHUB_RUN_ID:-unknown}"
  echo "source_artifact_run=40219788151"
  echo "source_data_interactions=12000_reused_no_new_training_interactions"
  echo "native_training_interactions=0"
  echo "dataset_sha256_expected=02dd11feb60e0dcb5710ae8af3cf33dca7e7e77756d6467f822ad94d0f5a0d8d"
  echo "task=Authentic_Taisei_Stage1_Easy"
  echo "development_eval_seeds=42132-42147"
  echo "never_use_final_seeds=43000-43099"
} > "$OUT/e4_run_manifest.txt"

python ../research/rsi/e4_rebalanced_bc.py \
  --dataset "$SRC/teacher_demonstrations_12000.npz" \
  --baseline-bc "$SRC/teacher_bc_seed48030.pt" \
  --outdir "$OUT" 2>&1 | tee "$OUT/e4_offline_training.stdout.log"
cp "$SRC/teacher_bc_seed48030.pt" "$OUT/teacher_bc_seed48030.pt"

export TAISEI_CI_SDL_PREFIX="${RUNNER_TEMP:-/tmp}/taisei-sdl3-prefix"
export PKG_CONFIG_PATH="$TAISEI_CI_SDL_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$TAISEI_CI_SDL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash ../research/rsi/build_native_with_shared_sdl3.sh 2>&1 | tee "$OUT/native_build.log"
ROOT="$PWD/upstream/taisei-sim-pinned"
LIB="$(find "$ROOT/build" -type f -name 'libtaisei_sim.so' | head -n1)"
test -f "$LIB"
meson devenv -C "$ROOT/build" python "$PWD/scripts/smoke_native.py" \
  --library "$LIB" --resource-path "$ROOT/resources" --stage 1 --difficulty 1 \
  --seed 42000 --frames 300 2>&1 | tee "$OUT/native_smoke.log"
meson devenv -C "$ROOT/build" python "$PWD/scripts/check_native_determinism.py" \
  --library "$LIB" --resource-path "$ROOT/resources" --stage 1 --difficulty 1 \
  --seed 42000 --frames 600 2>&1 | tee "$OUT/determinism.log"

for name in teacher_bc_seed48030 teacher_bc_rebalanced_seed48030; do
  meson devenv -C "$ROOT/build" python "$PWD/scripts/evaluate_native.py" \
    --library "$LIB" --resource-path "$ROOT/resources" --stage 1 --difficulty 1 \
    --max-frames 10000 --policy ppo --model "$OUT/${name}.pt" \
    --episodes 16 --seed 42132 --out "$OUT/${name}_native.jsonl" \
    2>&1 | tee "$OUT/${name}_native.stdout.log"
done

meson devenv -C "$ROOT/build" python "$PWD/../research/rsi/e4_eval_teacher.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --out "$OUT/risk_teacher_native.jsonl" \
  2>&1 | tee "$OUT/risk_teacher_native.stdout.log"

python ../research/rsi/e4_compare.py \
  --baseline "$OUT/teacher_bc_seed48030_native.jsonl" \
  --candidate "$OUT/teacher_bc_rebalanced_seed48030_native.jsonl" \
  --teacher "$OUT/risk_teacher_native.jsonl" \
  --offline "$OUT/e4_offline_bc_report.json" \
  --models-dir "$OUT" --out "$OUT/e4_native_comparison.json" \
  2>&1 | tee "$OUT/e4_native_comparison.stdout.log"
