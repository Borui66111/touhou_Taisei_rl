#!/usr/bin/env bash
# GitHub-hosted Linux native-Taisei dependency builder. Preserves pinned game source.
# The Taisei fallback explicitly builds SDL3_static as non-PIC and cannot link
# into libtaisei_sim.so. Build the exact wrapper-tag SDL3 as a shared system dep.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../my_test" && pwd -P)"
CI_TEMP="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
PREFIX="${TAISEI_CI_SDL_PREFIX:-$CI_TEMP/taisei-sdl3-prefix}"
SOURCE="$CI_TEMP/taisei-sdl3-source"
BUILD="$CI_TEMP/taisei-sdl3-build"
SDL_REPO="https://github.com/taisei-project/SDL.git"
SDL_TAG="taisei-3.4.12"

if [[ -e "$SOURCE" || -e "$BUILD" || -e "$PREFIX" ]]; then
  echo "Refusing to overwrite prior SDL3 CI directories" >&2
  exit 1
fi
git clone --depth 1 --branch "$SDL_TAG" "$SDL_REPO" "$SOURCE"
SDL_SHA="$(git -C "$SOURCE" rev-parse HEAD)"
echo "SDL3 upstream=$SDL_REPO tag=$SDL_TAG sha=$SDL_SHA"

cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
  -DSDL_SHARED=ON -DSDL_STATIC=OFF \
  -DSDL_X11=OFF -DSDL_WAYLAND=OFF \
  -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF \
  -DSDL_DISABLE_UNINSTALL=ON

cmake --build "$BUILD" --parallel 2
cmake --install "$BUILD"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
pkg-config --modversion sdl3
echo "SDL3 shared library:"
find "$PREFIX/lib" -maxdepth 2 -type f -name '*SDL3*' -print

cd "$ROOT"
bash scripts/build_taisei_sim.sh
