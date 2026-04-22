#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SPECFEM2D_DIR="$ROOT_DIR/third_party/specfem2d"
SPECFEM3D_DIR="$ROOT_DIR/third_party/specfem3d"
SPECFEM2D_SOURCE="${SPECFEM2D_SOURCE:-$(cd "$ROOT_DIR/.." && pwd)/specfem2d}"
SPECFEM3D_SOURCE="${SPECFEM3D_SOURCE:-$(cd "$ROOT_DIR/.." && pwd)/specfem3d}"
SPECFEM2D_REF=""
SPECFEM3D_REF=""
JOBS=""
REBUILD="0"

usage() {
  cat <<'EOF'
Usage: install_specfem_local.sh [--jobs N] [--rebuild] [--specfem2d-dir DIR] [--specfem3d-dir DIR] [--specfem2d-source PATH_OR_URL] [--specfem3d-source PATH_OR_URL] [--specfem2d-ref REF] [--specfem3d-ref REF]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --rebuild)
      REBUILD="1"
      shift 1
      ;;
    --specfem2d-dir)
      SPECFEM2D_DIR="$2"
      shift 2
      ;;
    --specfem3d-dir)
      SPECFEM3D_DIR="$2"
      shift 2
      ;;
    --specfem2d-source)
      SPECFEM2D_SOURCE="$2"
      shift 2
      ;;
    --specfem3d-source)
      SPECFEM3D_SOURCE="$2"
      shift 2
      ;;
    --specfem2d-ref)
      SPECFEM2D_REF="$2"
      shift 2
      ;;
    --specfem3d-ref)
      SPECFEM3D_REF="$2"
      shift 2
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

cmd_2d=("$ROOT_DIR/scripts/install_specfem2d_local.sh" --dir "$SPECFEM2D_DIR" --source "$SPECFEM2D_SOURCE")
cmd_3d=("$ROOT_DIR/scripts/install_specfem3d_local.sh" --dir "$SPECFEM3D_DIR" --source "$SPECFEM3D_SOURCE")

if [[ -n "$JOBS" ]]; then
  cmd_2d+=(--jobs "$JOBS")
  cmd_3d+=(--jobs "$JOBS")
fi
if [[ "$REBUILD" == "1" ]]; then
  cmd_2d+=(--rebuild)
  cmd_3d+=(--rebuild)
fi
if [[ -n "$SPECFEM2D_REF" ]]; then
  cmd_2d+=(--ref "$SPECFEM2D_REF")
fi
if [[ -n "$SPECFEM3D_REF" ]]; then
  cmd_3d+=(--ref "$SPECFEM3D_REF")
fi

"${cmd_2d[@]}"
"${cmd_3d[@]}"
