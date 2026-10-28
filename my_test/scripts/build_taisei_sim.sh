#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKBENCH_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
LOCK_FILE="$WORKBENCH_ROOT/UPSTREAM.lock"
ROOT_INPUT="${1:-$WORKBENCH_ROOT/upstream/taisei-sim-pinned}"
if ! ROOT="$(realpath -m -- "$ROOT_INPUT")"; then
  echo "ERROR: could not canonicalize upstream checkout path: $ROOT_INPUT" >&2
  exit 1
fi
case "$ROOT" in
  "$WORKBENCH_ROOT"/*) ;;
  *)
    echo "ERROR: upstream checkout path must be inside the workbench ($WORKBENCH_ROOT): $ROOT" >&2
    exit 1
    ;;
esac
STAGE_DIR=''
cleanup_stage() {
  if [[ -n "$STAGE_DIR" && -d "$STAGE_DIR" ]]; then
    rm -rf -- "$STAGE_DIR"
  fi
}
trap cleanup_stage EXIT

if ! command -v git >/dev/null; then
  echo 'ERROR: git required' >&2
  exit 1
fi
if [[ ! -f "$LOCK_FILE" ]]; then
  echo "ERROR: missing upstream lock file: $LOCK_FILE" >&2
  exit 1
fi
mapfile -t LOCK_LINES < "$LOCK_FILE"
if [[ "${#LOCK_LINES[@]}" -ne 2 || "${LOCK_LINES[0]}" != 'REPO_URL=https://github.com/touhourl/taisei-sim.git' || "${LOCK_LINES[1]}" != 'COMMIT=789200ff255c681b3ed4230a3f26f0b8c12c78fe' ]]; then
  echo "ERROR: malformed or unexpected upstream lock file: $LOCK_FILE" >&2
  exit 1
fi
REPO_URL="${LOCK_LINES[0]#REPO_URL=}"
COMMIT="${LOCK_LINES[1]#COMMIT=}"

FRESH_CHECKOUT=0
if [[ -e "$ROOT" ]]; then
  if [[ ! -d "$ROOT" ]] || ! git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: existing path is not a valid Git checkout; refusing to overwrite: $ROOT" >&2
    exit 1
  fi
  if [[ "$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null)" != "$(cd -- "$ROOT" && pwd)" ]]; then
    echo "ERROR: existing path is not the root of its Git checkout: $ROOT" >&2
    exit 1
  fi
  REMOTE_URL="$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)"
  if [[ "$REMOTE_URL" != "$REPO_URL" ]]; then
    echo "ERROR: checkout origin mismatch (expected $REPO_URL, found ${REMOTE_URL:-<missing>}); refusing to proceed" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all --ignore-submodules=none)" ]]; then
    echo "ERROR: checkout has local changes; refusing to modify it: $ROOT" >&2
    exit 1
  fi
  HEAD_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$HEAD_COMMIT" != "$COMMIT" ]]; then
    echo "ERROR: checkout HEAD mismatch (expected $COMMIT, found ${HEAD_COMMIT:-<invalid>}); refusing to switch revisions" >&2
    exit 1
  fi
else
  FRESH_CHECKOUT=1
  TARGET_PARENT="$(dirname -- "$ROOT")"
  mkdir -p -- "$TARGET_PARENT"
  WORKBENCH_DEVICE="$(stat -c '%d' -- "$WORKBENCH_ROOT")"
  TARGET_PARENT_DEVICE="$(stat -c '%d' -- "$TARGET_PARENT")"
  if [[ "$WORKBENCH_DEVICE" != "$TARGET_PARENT_DEVICE" ]]; then
    echo "ERROR: upstream checkout target is on a different filesystem than the workbench; refusing non-atomic publish: $TARGET_PARENT" >&2
    exit 1
  fi
  STAGE_DIR="$(mktemp -d "$WORKBENCH_ROOT/.taisei-sim-stage.XXXXXX")"
  STAGING_CHECKOUT="$STAGE_DIR/checkout"
  git clone --no-checkout "$REPO_URL" "$STAGING_CHECKOUT"
  git -C "$STAGING_CHECKOUT" checkout --detach "$COMMIT"
  git -C "$STAGING_CHECKOUT" submodule update --init --recursive
  STAGING_COMMIT="$(git -C "$STAGING_CHECKOUT" rev-parse HEAD)"
  if [[ "$STAGING_COMMIT" != "$COMMIT" ]]; then
    echo "ERROR: checked-out commit mismatch (expected $COMMIT, found $STAGING_COMMIT)" >&2
    exit 1
  fi
  STAGING_REMOTE_URL="$(git -C "$STAGING_CHECKOUT" remote get-url origin)"
  if [[ "$STAGING_REMOTE_URL" != "$REPO_URL" ]]; then
    echo "ERROR: checkout origin mismatch (expected $REPO_URL, found ${STAGING_REMOTE_URL:-<missing>}); refusing to proceed" >&2
    exit 1
  fi
  STAGING_SUBMODULE_STATUS="$(git -C "$STAGING_CHECKOUT" submodule status --recursive)"
  while IFS= read -r SUBMODULE_LINE; do
    case "$SUBMODULE_LINE" in
      -*|+*|U*)
        echo "ERROR: submodule is not initialized at its recorded revision: $SUBMODULE_LINE" >&2
        exit 1
        ;;
    esac
  done <<< "$STAGING_SUBMODULE_STATUS"
  mkdir -p "$(dirname -- "$ROOT")"
  if [[ -e "$ROOT" ]]; then
    echo "ERROR: target appeared while preparing checkout; refusing to overwrite: $ROOT" >&2
    exit 1
  fi
  mv -T -n -- "$STAGING_CHECKOUT" "$ROOT"
  if [[ -e "$STAGING_CHECKOUT" ]]; then
    echo "ERROR: target appeared while publishing checkout; refusing to overwrite: $ROOT" >&2
    exit 1
  fi
fi
cd "$ROOT"
if [[ "$FRESH_CHECKOUT" -eq 0 ]]; then
  git submodule update --init --recursive
fi
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$COMMIT" ]]; then
  echo "ERROR: checked-out commit mismatch (expected $COMMIT, found $ACTUAL_COMMIT)" >&2
  exit 1
fi
if [[ "$(git remote get-url origin)" != "$REPO_URL" ]]; then
  echo "ERROR: checkout origin changed unexpectedly" >&2
  exit 1
fi
SUBMODULE_STATUS="$(git submodule status --recursive)"
while IFS= read -r SUBMODULE_LINE; do
  case "$SUBMODULE_LINE" in
    -*|+*|U*)
      echo "ERROR: submodule is not initialized at its recorded revision: $SUBMODULE_LINE" >&2
      exit 1
      ;;
  esac
done <<< "$SUBMODULE_STATUS"
printf 'Verified Taisei source: %s (%s)\n' "$REPO_URL" "$ACTUAL_COMMIT"
printf 'Verified all recursive submodules are initialized.\n'

if ! command -v meson >/dev/null; then
  echo 'ERROR: Python venv with Meson >= 1.8 is required. See README_WINDOWS.md.' >&2
  exit 1
fi
if [[ -d build/meson-info ]]; then
  meson setup build --buildtype=debugoptimized -Dr_null=enabled -Da_null=enabled --reconfigure
else
  meson setup build --buildtype=debugoptimized -Dr_null=enabled -Da_null=enabled
fi
meson compile -C build
printf '\nCompiled headless Taisei candidate libraries:\n'
find "$ROOT/build" -name 'libtaisei_sim.so*' | head -8
