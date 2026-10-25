#!/usr/bin/env bash
# Fail-closed real Taisei pilot; NO baseline promotion or final-test evaluation.
set -euo pipefail
cd "$(dirname "$0")/../../my_test"
OUT="$PWD/results/rsi_pilot"
mkdir -p "$OUT"
# Task identities are pinned to native src/difficulty.h: D_Easy = 1, D_Normal = 2.
EASY_DIFFICULTY=1
{
  echo "github_sha=${GITHUB_SHA:-unknown}"
  echo "stage_id=1"
  echo "difficulty_id=$EASY_DIFFICULTY"
  echo "difficulty_label=Easy (native enum D_Easy=1)"
  echo "github_run_id=${GITHUB_RUN_ID:-unknown}"
  echo "sim_source=$(cat UPSTREAM.lock | tr '\n' ' ')"
  echo "python=$(python --version)"
  echo "meson=$(meson --version)"
} > "$OUT/manifest.txt"

# A real engine is mandatory. Any failure stops the pipeline.
export TAISEI_CI_SDL_PREFIX="${RUNNER_TEMP:-/tmp}/taisei-sdl3-prefix"
export PKG_CONFIG_PATH="$TAISEI_CI_SDL_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$TAISEI_CI_SDL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash ../research/rsi/build_native_with_shared_sdl3.sh 2>&1 | tee "$OUT/native_build.log"
ROOT="$PWD/upstream/taisei-sim-pinned"
LIB="$(find "$ROOT/build" -type f -name 'libtaisei_sim.so*' | head -n1)"
test -n "$LIB" || { echo "BLOCKED: native simulator library missing" >&2; exit 1; }
printf 'native_library=%s\n' "$LIB" >> "$OUT/manifest.txt"

meson devenv -C "$ROOT/build" python "$PWD/scripts/smoke_native.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --stage 1 --difficulty "$EASY_DIFFICULTY" --seed 42000 --frames 300 \
  2>&1 | tee "$OUT/native_smoke.log"

# Bounded pipeline verification: a fresh SKILL policy, NOT RSI proof.
# Do not use sealed 43000-43099 evaluation seeds anywhere.
meson devenv -C "$ROOT/build" python "$PWD/scripts/train_torch_ppo.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --stage 1 --difficulty "$EASY_DIFFICULTY" --policy-type skill \
  --skill-action-mask --seed 48010 --steps 1024 --rollout-steps 256 \
  --out "$OUT/pilot_skill.pt" --log "$OUT/pilot_training.jsonl" \
  2>&1 | tee "$OUT/pilot_training.stdout.log"

# Same fixed initial game seeds for baseline and pilot.
for policy in heuristic random; do
  meson devenv -C "$ROOT/build" python "$PWD/scripts/evaluate_native.py" \
    --library "$LIB" --resource-path "$ROOT/resources" \
    --stage 1 --difficulty "$EASY_DIFFICULTY" --policy "$policy" \
    --episodes 4 --seed 42100 --out "$OUT/${policy}.jsonl" \
    2>&1 | tee "$OUT/${policy}.stdout.log"
done
meson devenv -C "$ROOT/build" python "$PWD/scripts/evaluate_native.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --stage 1 --difficulty "$EASY_DIFFICULTY" --policy ppo --model "$OUT/pilot_skill.pt" \
  --episodes 4 --seed 42100 --out "$OUT/pilot_skill_evaluation.jsonl" \
  2>&1 | tee "$OUT/pilot_skill_evaluation.stdout.log"

python ../research/rsi/summarize_pilot.py "$OUT"
