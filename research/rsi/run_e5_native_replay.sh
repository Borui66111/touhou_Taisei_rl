#!/usr/bin/env bash
# E5: capped native policy replays of existing immutable E3/E4 checkpoints.
# No training, no final seeds, no champion promotion, no toy fallback.
set -euo pipefail
cd "$(dirname "$0")/../../my_test"
OUT="$PWD/results/rsi_e5_native_replay_v1"
E3="$PWD/results/e3_artifact"
E4="$PWD/results/e4_artifact"
mkdir -p "$OUT"
for p in "$E3/teacher_demonstrations_12000.npz" "$E3/teacher_bc_seed48030.pt" \
         "$E4/teacher_bc_seed48030.pt" "$E4/teacher_bc_rebalanced_seed48030.pt"; do
  test -s "$p" || { echo "BLOCKED: missing authentic Taisei evidence $p" >&2; exit 1; }
done
{
  echo "experiment=E5_TWO_PROCESS_NATIVE_POLICY_REPLAY_V1"
  echo "source_sha=${GITHUB_SHA:-unknown}"
  echo "github_run_id=${GITHUB_RUN_ID:-unknown}"
  echo "native=Taisei_Stage1_Easy"
  echo "seeds=42132,42133"
  echo "max_frames_per_episode=3000"
  echo "per_policy_max_decisions=12000"
  echo "training_decisions=0"
  echo "sealed_final_seeds=43000-43099_UNUSED"
  echo "upstream=$(tr '\\n' ' ' < UPSTREAM.lock)"
} > "$OUT/manifest.txt"
python ../research/rsi/e5_audit_replay.py audit \
  --e3 "$E3" --e4 "$E4" --out "$OUT/archived_native_audit.json" \
  2>&1 | tee "$OUT/audit.stdout.log"
export TAISEI_CI_SDL_PREFIX="${RUNNER_TEMP:-/tmp}/taisei-sdl3-prefix"
export PKG_CONFIG_PATH="$TAISEI_CI_SDL_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$TAISEI_CI_SDL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash ../research/rsi/build_native_with_shared_sdl3.sh 2>&1 | tee "$OUT/native_build.log"
ROOT="$PWD/upstream/taisei-sim-pinned"
LIB="$(find "$ROOT/build" -type f -name 'libtaisei_sim.so' | head -n1)"
test -f "$LIB" || { echo "BLOCKED: genuine native simulator missing" >&2; exit 1; }
meson devenv -C "$ROOT/build" python "$PWD/scripts/smoke_native.py" \
  --library "$LIB" --resource-path "$ROOT/resources" --stage 1 --difficulty 1 \
  --seed 42000 --frames 300 2>&1 | tee "$OUT/native_smoke.log"
for name in original_bc rebalanced_bc; do
  case "$name" in
    original_bc) MODEL="$E4/teacher_bc_seed48030.pt" ;;
    rebalanced_bc) MODEL="$E4/teacher_bc_rebalanced_seed48030.pt" ;;
  esac
  # Fresh OS process per pass: Taisei native lifecycle is not reliably reentrant.
  for pass in first second; do
    meson devenv -C "$ROOT/build" python "$PWD/scripts/evaluate_native.py" \
      --library "$LIB" --resource-path "$ROOT/resources" \
      --stage 1 --difficulty 1 --max-frames 3000 \
      --policy ppo --model "$MODEL" --episodes 2 --seed 42132 \
      --out "$OUT/${name}_${pass}.jsonl" \
      --trace-out "$OUT/${name}_${pass}.trace.jsonl" \
      2>&1 | tee "$OUT/${name}_${pass}.stdout.log"
  done
  python ../research/rsi/e5_audit_replay.py replay \
    --first-summary "$OUT/${name}_first.jsonl" \
    --second-summary "$OUT/${name}_second.jsonl" \
    --first-trace "$OUT/${name}_first.trace.jsonl" \
    --second-trace "$OUT/${name}_second.trace.jsonl" \
    --out "$OUT/${name}_native_replay_verification.json" \
    2>&1 | tee "$OUT/${name}_native_replay_verification.stdout.log"
done
