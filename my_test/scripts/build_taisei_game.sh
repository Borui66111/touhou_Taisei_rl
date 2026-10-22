#!/usr/bin/env bash
set -euo pipefail
# Builds the REAL rendered game, separately from the native headless RL simulation.
ROOT="${1:-$PWD/upstream/taisei-game}"
mkdir -p "$(dirname "$ROOT")"
if [[ ! -d "$ROOT/.git" ]]; then
  git clone --recurse-submodules https://github.com/taisei-project/taisei.git "$ROOT"
fi
cd "$ROOT"
git submodule update --init --recursive
if ! command -v meson >/dev/null; then
  echo 'meson >=1.8 required; see Taisei doc/BUILD.rst' >&2
  exit 2
fi
if [[ ! -d build ]]; then
  meson setup build
else
  meson setup build --reconfigure
fi
meson compile -C build
printf '\nRendered Taisei compiled. Locate executable in %s/build.\n' "$ROOT"
