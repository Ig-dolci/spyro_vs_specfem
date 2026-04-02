#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$NOTEBOOK_DIR/compare_spyro_specfem3d.sh" "$@"
