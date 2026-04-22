#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DIR="$ROOT_DIR/third_party/specfem2d"
DEFAULT_SOURCE="$(cd "$ROOT_DIR/.." && pwd)/specfem2d"
REMOTE_URL="https://github.com/SPECFEM/specfem2d.git"
TARGET_DIR="$DEFAULT_DIR"
SOURCE_REPO="${SPECFEM2D_SOURCE:-$DEFAULT_SOURCE}"
REF=""
REBUILD="0"
JOBS="$(python3 - <<'PY'
import os
print(os.cpu_count() or 4)
PY
)"

usage() {
  cat <<'EOF'
Usage: install_specfem2d_local.sh [--dir DIR] [--source PATH_OR_URL] [--ref REF] [--jobs N] [--rebuild]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --source)
      SOURCE_REPO="$2"
      shift 2
      ;;
    --ref)
      REF="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --rebuild)
      REBUILD="1"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

TARGET_DIR="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$TARGET_DIR")"
mkdir -p "$(dirname "$TARGET_DIR")"

if [[ ! -e "$TARGET_DIR/configure" ]]; then
  if [[ -d "$SOURCE_REPO/.git" || -e "$SOURCE_REPO/configure" ]]; then
    echo "Cloning SPECFEM2D from local source $SOURCE_REPO"
    git clone "$SOURCE_REPO" "$TARGET_DIR"
  else
    echo "Cloning SPECFEM2D from $REMOTE_URL"
    git clone --depth 1 "$REMOTE_URL" "$TARGET_DIR"
  fi
fi

if [[ -n "$REF" && -d "$TARGET_DIR/.git" ]]; then
  git -C "$TARGET_DIR" fetch --tags --force
  git -C "$TARGET_DIR" checkout "$REF"
fi

pushd "$TARGET_DIR" >/dev/null
if [[ "$REBUILD" == "1" ]]; then
  make clean || true
fi
env FC="${FC:-gfortran}" ./configure
make -j "$JOBS" all
popd >/dev/null

if [[ ! -x "$TARGET_DIR/bin/xmeshfem2D" || ! -x "$TARGET_DIR/bin/xspecfem2D" ]]; then
  echo "SPECFEM2D build completed but expected executables were not found in $TARGET_DIR/bin" >&2
  exit 1
fi

echo "SPECFEM2D local installation ready at $TARGET_DIR"
echo "Executables:"
echo "  $TARGET_DIR/bin/xmeshfem2D"
echo "  $TARGET_DIR/bin/xspecfem2D"
