#!/usr/bin/env bash
# E1: actual native Easy Stage 1, controlled movement-only reward comparison.
# No toy fallback, no checkpoint promotion, no final test seeds.
set -euo pipefail
cd "$(dirname "$0")/../../my_test"
OUT="$PWD/results/rsi_e1"
mkdir -p "$OUT"
{
  echo "github_sha=${GITHUB_SHA:-unknown}"
  echo "experiment=E1_movement_reward_ablation"
  echo "task=Stage 1 Easy (D_Easy=1)"
  echo "native_upstream=$(tr '\n' ' ' < UPSTREAM.lock)"
  echo "training_seed=48020"
  echo "evaluation_seeds=42100-42115"
  echo "sealed_final_seeds=43000-43099_unused"
} > "$OUT/execution_manifest.txt"

export TAISEI_CI_SDL_PREFIX="${RUNNER_TEMP:-/tmp}/taisei-sdl3-prefix"
export PKG_CONFIG_PATH="$TAISEI_CI_SDL_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$TAISEI_CI_SDL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash ../research/rsi/build_native_with_shared_sdl3.sh 2>&1 | tee "$OUT/native_build.log"
ROOT="$PWD/upstream/taisei-sim-pinned"
LIB="$(find "$ROOT/build" -type f -name 'libtaisei_sim.so*' | head -n1)"
test -n "$LIB" || { echo "Missing native Taisei shared library" >&2; exit 1; }

meson devenv -C "$ROOT/build" python "$PWD/scripts/smoke_native.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --stage 1 --difficulty 1 --seed 42000 --frames 300 \
  2>&1 | tee "$OUT/native_smoke.log"

# Same action trace, reset twice in one real Taisei runtime process.
meson devenv -C "$ROOT/build" python "$PWD/scripts/check_native_determinism.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --stage 1 --difficulty 1 --seed 42000 --frames 600 \
  2>&1 | tee "$OUT/determinism.log"

for arm in baseline survival_first; do
  meson devenv -C "$ROOT/build" python "$PWD/../research/rsi/e1_train_movement.py" \
    --arm "$arm" --library "$LIB" --resource-path "$ROOT/resources" \
    --stage 1 --difficulty 1 --steps 20000 --seed 48020 --outdir "$OUT" \
    2>&1 | tee "$OUT/${arm}_training.stdout.log"
  meson devenv -C "$ROOT/build" python "$PWD/scripts/evaluate_native.py" \
    --library "$LIB" --resource-path "$ROOT/resources" \
    --stage 1 --difficulty 1 --max-frames 10000 \
    --policy ppo --model "$OUT/${arm}_seed48020.pt" \
    --episodes 16 --seed 42100 --out "$OUT/${arm}_evaluation.jsonl" \
    2>&1 | tee "$OUT/${arm}_evaluation.stdout.log"
done

meson devenv -C "$ROOT/build" python "$PWD/scripts/evaluate_native.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --stage 1 --difficulty 1 --max-frames 10000 \
  --policy heuristic --episodes 16 --seed 42100 \
  --out "$OUT/heuristic_evaluation.jsonl" \
  2>&1 | tee "$OUT/heuristic_evaluation.stdout.log"

python ../research/rsi/e1_summarize.py "$OUT" | tee "$OUT/summary.stdout.log"
