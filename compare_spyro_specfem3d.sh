#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$NOTEBOOK_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" "$ROOT_DIR/scripts/compare_spyro_specfem3d.py" "$@"
