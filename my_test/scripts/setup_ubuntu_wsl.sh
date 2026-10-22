#!/usr/bin/env bash
# Run inside Ubuntu on WSL2. Installs dependencies and builds the REAL native Taisei.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$(uname -s)" != Linux ]]; then echo 'Requires GNU/Linux (WSL2 Ubuntu supported).' >&2; exit 2; fi
printf '\n== Ubuntu / WSL2: build requirements ==\n'
sudo apt-get update
sudo apt-get install -y git build-essential cmake ninja-build pkg-config python3 python3-dev python3-venv python3-pip \
  gettext libfreetype-dev libwebp-dev libzstd-dev zlib1g-dev libpng-dev libssl-dev \
  libopusfile-dev libunibreak-dev
# SDL3 >= 3.2, cglm and other optional development packages may be unavailable
# in older Ubuntu repos; Taisei's Meson wrap fallback can fetch them when online.
for pkg in libsdl3-dev libcglm-dev libgl1-mesa-dev libegl1-mesa-dev; do
  if apt-cache show "$pkg" >/dev/null 2>&1; then sudo apt-get install -y "$pkg"; fi
done
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip 'meson>=1.8' ninja backports.zstd numpy pytest
printf '\n== Clone and compile native game engine ==\n'
bash scripts/build_taisei_sim.sh
printf '\n== Find compiled native simulation ==\n'
LIB="$(find "$ROOT/upstream/taisei-sim-pinned/build" -type f -name 'libtaisei_sim.so*' | head -n1)"
if [[ -z "$LIB" ]]; then echo 'Native library missing after Meson build. Check build logs.' >&2; exit 1; fi
printf 'Library: %s\n' "$LIB"
cat > .native_paths <<EOF
TAISEI_NATIVE_LIBRARY='$LIB'
TAISEI_NATIVE_RESOURCES='$ROOT/upstream/taisei-sim-pinned/resources'
EOF
meson devenv -C "$ROOT/upstream/taisei-sim-pinned/build" -- \
  python scripts/doctor.py --library "$LIB" --resource-path "$ROOT/upstream/taisei-sim-pinned/resources" --check-dns
printf '\n== Actual headless gameplay smoke test ==\n'
# meson devenv sets compiled game's resource/dependency environment.
meson devenv -C "$ROOT/upstream/taisei-sim-pinned/build" -- \
  python scripts/smoke_native.py --library "$LIB" \
  --resource-path "$ROOT/upstream/taisei-sim-pinned/resources" --stage 1 --difficulty 2 --frames 1200
printf '\nNative smoke test finished. For training, see README_WINDOWS.md.\n'
