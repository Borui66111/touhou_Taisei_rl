#!/usr/bin/env bash
# Build libtaisei_sim from PUBLIC Taisei + the reconstruction patch (fail-closed).
#
# The original simulator (touhourl/taisei-sim@789200ff, my_test/UPSTREAM.lock) is no
# longer reachable. This builds a reconstruction whose behaviour was validated against
# the archived original evidence (research/rsi/validate_rebuilt_sim.py).
#
# Usage: build_rebuilt_sim.sh [CHECKOUT_DIR]
#   default CHECKOUT_DIR: my_test/upstream/taisei-sim-rebuild
# Requirements: git, meson >= 1.8, ninja, a C11 compiler, gettext (msgfmt), Python with
# backports.zstd (Python < 3.14), and the SDL3 build used by CI
# (research/rsi/build_native_with_shared_sdl3.sh; pass its prefix via PKG_CONFIG_PATH).
# Outputs (printed at the end, also written to <CHECKOUT_DIR>/rebuilt_sim.env):
#   TAISEI_SIM_LIBRARY=<.../libtaisei_sim.so>   TAISEI_SIM_RESOURCES=<.../share/taisei>
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKBENCH_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
LOCK_FILE="$SCRIPT_DIR/UPSTREAM_PUBLIC.lock"
ROOT="$(realpath -m -- "${1:-$WORKBENCH_ROOT/upstream/taisei-sim-rebuild}")"

die() { echo "ERROR: $*" >&2; exit 1; }
lock() { grep -E "^$1=" "$LOCK_FILE" | head -n1 | cut -d= -f2-; }

[[ -f "$LOCK_FILE" ]] || die "missing lock file $LOCK_FILE"
REPO_URL="$(lock REPO_URL)"; COMMIT="$(lock COMMIT)"
PATCH="$SCRIPT_DIR/$(lock PATCH)"; PATCH_SHA256="$(lock PATCH_SHA256)"
[[ "$REPO_URL" == "https://github.com/taisei-project/taisei.git" ]] || die "unexpected REPO_URL $REPO_URL"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "malformed COMMIT"
[[ -f "$PATCH" ]] || die "missing patch $PATCH"
[[ "$(sha256sum "$PATCH" | cut -d' ' -f1)" == "$PATCH_SHA256" ]] || die "patch sha256 mismatch"
command -v meson >/dev/null || die "meson >= 1.8 required"
command -v msgfmt >/dev/null || die "gettext (msgfmt) required"

if [[ -e "$ROOT" ]]; then
  git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$ROOT exists and is not a git checkout; refusing to overwrite"
  [[ "$(git -C "$ROOT" remote get-url origin)" == "$REPO_URL" ]] || die "origin mismatch in $ROOT"
  [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$COMMIT" ]] || die "HEAD mismatch in $ROOT (expected $COMMIT)"
else
  mkdir -p -- "$(dirname -- "$ROOT")"
  STAGE="$(mktemp -d "$(dirname -- "$ROOT")/.taisei-rebuild-stage.XXXXXX")"
  trap 'rm -rf -- "$STAGE"' EXIT
  git clone --no-checkout "$REPO_URL" "$STAGE/checkout"
  git -C "$STAGE/checkout" checkout --detach "$COMMIT"
  git -C "$STAGE/checkout" submodule update --init --recursive
  mv -T -n -- "$STAGE/checkout" "$ROOT"
fi

# Submodules must be exactly those of the archived original build (escrow native_build.log).
while IFS= read -r line; do
  path="${line%%=*}"; path="${path#SUBMODULE_}"; want="${line#*=}"
  have="$(git -C "$ROOT" ls-tree HEAD "$path" | awk '{print $3}')"
  [[ "$have" == "$want" ]] || die "submodule $path at $have, expected $want"
  status="$(git -C "$ROOT" submodule status -- "$path")"
  [[ "$status" == " $want "* ]] || die "submodule $path not initialized at $want: $status"
done < <(grep -E '^SUBMODULE_' "$LOCK_FILE")

# Apply the reconstruction patch exactly once; any other local change is refused.
if git -C "$ROOT" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "Reconstruction patch already applied."
else
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "checkout has unexpected local changes"
  git -C "$ROOT" apply --check "$PATCH"
  git -C "$ROOT" apply "$PATCH"
fi
printf 'Verified public Taisei %s @ %s + %s\n' "$REPO_URL" "$COMMIT" "$(basename "$PATCH")"

cd "$ROOT"
if [[ -d build/meson-info ]]; then
  meson setup build --buildtype=debugoptimized -Dr_null=enabled -Da_null=enabled -Dsim=true --reconfigure
else
  meson setup build --buildtype=debugoptimized -Dr_null=enabled -Da_null=enabled -Dsim=true
fi
meson compile -C build
rm -rf -- "$ROOT/install"
DESTDIR="$ROOT/install" meson install -C build --no-rebuild >/dev/null

LIB="$(find "$ROOT/install" -type f -name 'libtaisei_sim.so' | head -n1)"
RES="$(dirname "$(find "$ROOT/install" -type f -name '00-taisei.zip' | head -n1)")"
[[ -f "$LIB" && -d "$RES" ]] || die "build did not produce libtaisei_sim.so and resources"
printf 'TAISEI_SIM_LIBRARY=%s\nTAISEI_SIM_RESOURCES=%s\n' "$LIB" "$RES" | tee "$ROOT/rebuilt_sim.env"
echo "Validate with: python research/rsi/validate_rebuilt_sim.py --library \"$LIB\" --resource-path \"$RES\""
