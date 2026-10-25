#!/usr/bin/env bash
# E2 observation diagnostics with actual native Taisei and archived E1 models.
# Never retrain, never use toy gameplay, never access sealed final test seeds.
set -euo pipefail
cd "$(dirname "$0")/../../my_test"
OUT="$PWD/results/rsi_e2"
ARTIFACTS="$PWD/results/e1_download"
mkdir -p "$OUT"
for name in baseline survival_first; do
  mapfile -t FOUND < <(find "$ARTIFACTS" -type f -name "${name}_seed48020.pt")
  test "${#FOUND[@]}" -eq 1 || { echo "Missing/duplicated E1 checkpoint ${name}"; exit 1; }
  sha256sum "${FOUND[0]}" | tee "$OUT/${name}_checkpoint_sha256.txt"
  case "$name" in
    baseline) EXPECTED=a2a0ce0d4716945b228641844f65620ed1548a4342f7f891539e98303c1eed7b ;;
    survival_first) EXPECTED=5526812cb58988849165389db39eec50add78d0dfc41840f5eba1ecc491b93b8 ;;
  esac
  test "$(sha256sum "${FOUND[0]}" | cut -d' ' -f1)" = "$EXPECTED" || {
    echo "E1 checkpoint hash mismatch"; exit 1;
  }
done

export TAISEI_CI_SDL_PREFIX="${RUNNER_TEMP:-/tmp}/taisei-sdl3-prefix"
export PKG_CONFIG_PATH="$TAISEI_CI_SDL_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$TAISEI_CI_SDL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash ../research/rsi/build_native_with_shared_sdl3.sh 2>&1 | tee "$OUT/native_build.log"
ROOT="$PWD/upstream/taisei-sim-pinned"
LIB="$(find "$ROOT/build" -type f -name 'libtaisei_sim.so*' | head -n1)"
test -n "$LIB" || exit 1
meson devenv -C "$ROOT/build" python "$PWD/scripts/smoke_native.py" \
  --library "$LIB" --resource-path "$ROOT/resources" \
  --stage 1 --difficulty 1 --seed 42000 --frames 300 \
  2>&1 | tee "$OUT/native_smoke.log"

for name in baseline survival_first heuristic; do
  if [[ "$name" != heuristic ]]; then
    mapfile -t CKPT < <(find "$ARTIFACTS" -type f -name "${name}_seed48020.pt")
  fi
  if [[ "$name" == heuristic ]]; then
    meson devenv -C "$ROOT/build" python "$PWD/scripts/evaluate_native.py" \
      --library "$LIB" --resource-path "$ROOT/resources" \
      --stage 1 --difficulty 1 --max-frames 10000 --episodes 4 --seed 42100 \
      --policy heuristic --out "$OUT/${name}_evaluation.jsonl" \
      --trace-out "$OUT/${name}_trace.jsonl" \
      2>&1 | tee "$OUT/${name}_evaluation.stdout.log"
  else
    meson devenv -C "$ROOT/build" python "$PWD/scripts/evaluate_native.py" \
      --library "$LIB" --resource-path "$ROOT/resources" \
      --stage 1 --difficulty 1 --max-frames 10000 --episodes 4 --seed 42100 \
      --policy ppo --model "${CKPT[0]}" \
      --out "$OUT/${name}_evaluation.jsonl" \
      --trace-out "$OUT/${name}_trace.jsonl" \
      2>&1 | tee "$OUT/${name}_evaluation.stdout.log"
  fi
done
python ../research/rsi/e2_trace_diagnostics.py \
  --trace baseline "$OUT/baseline_trace.jsonl" \
  --trace survival_first "$OUT/survival_first_trace.jsonl" \
  --trace heuristic "$OUT/heuristic_trace.jsonl" \
  --out "$OUT/e2_diagnostics.json" \
  2>&1 | tee "$OUT/diagnostics.stdout.log"
