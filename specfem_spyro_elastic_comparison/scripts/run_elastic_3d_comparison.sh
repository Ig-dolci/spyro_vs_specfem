#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIREDRAKE_PYTHON="${FIREDRAKE_PYTHON:-/Users/ddolci/dev_code/venv-firedrake/bin/python}"
EDGE_LENGTH="0.2"
NPROC="1"
SOURCE_Z="-1.1"
SOURCE_Y="1.5"
RECEIVER_Z="-1.9"
RECEIVER_Y="1.5"
SOURCE_DIR_X="0.7071067811865476"
SOURCE_DIR_Y="0.7071067811865476"
SOURCE_DIR_Z="0.0"
SOURCE_DELAY="0.2"
SPYRO_DELAY_MODE="specfem-single-force-ricker"

usage() {
  cat <<'EOF'
Usage: run_elastic_3d_comparison.sh [--edge-length H] [--nproc N] [--source-z Z] [--source-y Y] [--receiver-z Z] [--receiver-y Y] [--source-dir-x X] [--source-dir-y Y] [--source-dir-z Z] [--source-delay T] [--spyro-delay-mode MODE]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --edge-length)
      EDGE_LENGTH="$2"
      shift 2
      ;;
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --source-z)
      SOURCE_Z="$2"
      shift 2
      ;;
    --source-y)
      SOURCE_Y="$2"
      shift 2
      ;;
    --receiver-z)
      RECEIVER_Z="$2"
      shift 2
      ;;
    --receiver-y)
      RECEIVER_Y="$2"
      shift 2
      ;;
    --source-dir-x)
      SOURCE_DIR_X="$2"
      shift 2
      ;;
    --source-dir-y)
      SOURCE_DIR_Y="$2"
      shift 2
      ;;
    --source-dir-z)
      SOURCE_DIR_Z="$2"
      shift 2
      ;;
    --source-delay)
      SOURCE_DELAY="$2"
      shift 2
      ;;
    --spyro-delay-mode)
      SPYRO_DELAY_MODE="$2"
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

"$FIREDRAKE_PYTHON" "$ROOT_DIR/scripts/run_spyro_elastic_forward_3d.py" \
  --edge-length "$EDGE_LENGTH" \
  --source-z "$SOURCE_Z" \
  --source-y "$SOURCE_Y" \
  --receiver-z "$RECEIVER_Z" \
  --receiver-y "$RECEIVER_Y" \
  --source-dir-x "$SOURCE_DIR_X" \
  --source-dir-y "$SOURCE_DIR_Y" \
  --source-dir-z "$SOURCE_DIR_Z" \
  --source-delay "$SOURCE_DELAY" \
  --source-delay-mode "$SPYRO_DELAY_MODE"

"$ROOT_DIR/scripts/run_specfem3d_case.sh" \
  --edge-length "$EDGE_LENGTH" \
  --nproc "$NPROC" \
  --source-z "$SOURCE_Z" \
  --source-y "$SOURCE_Y" \
  --receiver-z "$RECEIVER_Z" \
  --receiver-y "$RECEIVER_Y" \
  --source-dir-x "$SOURCE_DIR_X" \
  --source-dir-y "$SOURCE_DIR_Y" \
  --source-dir-z "$SOURCE_DIR_Z" \
  --source-delay "$SOURCE_DELAY"

"$FIREDRAKE_PYTHON" "$ROOT_DIR/scripts/compare_spyro_specfem3d.py"

echo "3D comparison results saved to $ROOT_DIR/results/comparison_3d"
