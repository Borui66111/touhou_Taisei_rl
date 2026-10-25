#!/usr/bin/env bash
# E2: low-cost authentic Taisei baseline comparison, not RL training.
# Source preflight and native build are mandatory; no toy fallback.
set -euo pipefail
cd "$(dirname "$0")/../../my_test"
OUT="$PWD/results/rsi_e2_teacher_v1"
mkdir -p "$OUT"
{
  echo "experiment=E2_NATIVE_RISK_TEACHER_V1"
  echo "github_sha=${GITHUB_SHA:-unknown}"
  echo "github_run_id=${GITHUB_RUN_ID:-unknown}"
  echo "upstream=$(tr '\n' ' ' < UPSTREAM.lock)"
  echo "task=Taisei genuine Stage1 Easy D_Easy=1 Reimu A"
  echo "evaluation_seeds=42100-42115"
  echo "sealed_final_seeds=43000-43099_unused"
  echo "policies=existing_heuristic,risk_teacher_v1"
  echo "bomb_and_surge_actions=disabled"
} > "$OUT/manifest.txt"

export TAISEI_CI_SDL_PREFIX="${RUNNER_TEMP:-/tmp}/taisei-sdl3-prefix"
export PKG_CONFIG_PATH="$TAISEI_CI_SDL_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$TAISEI_CI_SDL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash ../research/rsi/build_native_with_shared_sdl3.sh 2>&1 | tee "$OUT/native_build.log"
ROOT="$PWD/upstream/taisei-sim-pinned"
LIB="$(find "$ROOT/build" -type f -name 'libtaisei_sim.so*' | head -n1)"
test -n "$LIB" || { echo "Missing compiled authentic native simulator" >&2; exit 1; }

meson devenv -C "$ROOT/build" python "$PWD/scripts/smoke_native.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --stage 1 --difficulty 1 --seed 42000 --frames 300 \
  2>&1 | tee "$OUT/native_smoke.log"
meson devenv -C "$ROOT/build" python "$PWD/scripts/check_native_determinism.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --stage 1 --difficulty 1 --seed 42000 --frames 600 \
  2>&1 | tee "$OUT/determinism.log"

for policy in existing_heuristic risk_teacher_v1; do
  meson devenv -C "$ROOT/build" python "$PWD/../research/rsi/e2_evaluate_teacher.py" \
    --library "$LIB" --resource-path "$ROOT/resources" \
    --policy "$policy" --stage 1 --difficulty 1 \
    --episodes 16 --seed 42100 --max-frames 10000 \
    --out "$OUT/${policy}.jsonl" \
    2>&1 | tee "$OUT/${policy}.stdout.log"
done
python ../research/rsi/e2_compare_teacher.py \
  --baseline "$OUT/existing_heuristic.jsonl" \
  --teacher "$OUT/risk_teacher_v1.jsonl" \
  --out "$OUT/e2_teacher_comparison.json" \
  | tee "$OUT/comparison.stdout.log"
