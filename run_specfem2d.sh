#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$NOTEBOOK_DIR/.." && pwd)"

args=("$@")
has_results_dir=0
has_log_dir=0

for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --results-dir)
      has_results_dir=1
      ((i++))
      ;;
    --log-dir)
      has_log_dir=1
      ((i++))
      ;;
  esac
done

if [[ $has_results_dir -eq 0 ]]; then
  args+=(--results-dir "$ROOT_DIR/results/specfem")
fi
if [[ $has_log_dir -eq 0 ]]; then
  args+=(--log-dir "$ROOT_DIR/logs")
fi

exec bash "$ROOT_DIR/scripts/run_specfem_case.sh" "${args[@]}"
