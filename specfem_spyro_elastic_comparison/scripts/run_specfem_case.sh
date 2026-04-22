#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_DIR="$ROOT_DIR/specfem_case"
DATA_DIR="$CASE_DIR/DATA"
OUTPUT_DIR="$CASE_DIR/OUTPUT_FILES"
LOG_DIR="$ROOT_DIR/logs"
RESULTS_DIR="$ROOT_DIR/results/specfem"
EDGE_LENGTH="0.02"
SOURCE_Z="-1.1"
RECEIVER_Z="-1.9"
NUM_RECEIVERS="100"
FINAL_TIME="1.0"
DT="0.0005"
SPECFEM_ROOT="${SPECFEM2D_ROOT:-$ROOT_DIR/third_party/specfem2d}"
INSTALL_SCRIPT="$ROOT_DIR/scripts/install_specfem2d_local.sh"

now_seconds() {
  python3 - <<'PY'
import time
print(f"{time.perf_counter():.9f}")
PY
}

elapsed_seconds() {
  python3 - "$1" "$2" <<'PY'
import sys
start = float(sys.argv[1])
end = float(sys.argv[2])
print(f"{end - start:.6f}")
PY
}

usage() {
  cat <<'EOF'
Usage: run_specfem_case.sh [--edge-length H] [--source-z Z] [--receiver-z Z] [--results-dir DIR] [--log-dir DIR] [--specfem-root DIR]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --edge-length)
      EDGE_LENGTH="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --source-z)
      SOURCE_Z="$2"
      shift 2
      ;;
    --receiver-z)
      RECEIVER_Z="$2"
      shift 2
      ;;
    --num-receivers)
      NUM_RECEIVERS="$2"
      shift 2
      ;;
    --final-time)
      FINAL_TIME="$2"
      shift 2
      ;;
    --dt)
      DT="$2"
      shift 2
      ;;
    --specfem-root)
      SPECFEM_ROOT="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
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

RESULTS_DIR="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$RESULTS_DIR")"
LOG_DIR="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$LOG_DIR")"
SPECFEM_ROOT="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$SPECFEM_ROOT")"

if [[ ! -x "$SPECFEM_ROOT/bin/xmeshfem2D" || ! -x "$SPECFEM_ROOT/bin/xspecfem2D" ]]; then
  echo "SPECFEM2D executables were not found at $SPECFEM_ROOT; bootstrapping a local build."
  "$INSTALL_SCRIPT" --dir "$SPECFEM_ROOT"
fi

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
total_start="$(now_seconds)"

setup_start="$(now_seconds)"
python3 "$ROOT_DIR/scripts/setup_specfem_case.py" \
  --edge-length "$EDGE_LENGTH" \
  --source-z "$SOURCE_Z" \
  --receiver-z "$RECEIVER_Z" \
  --num-receivers "$NUM_RECEIVERS" \
  --final-time "$FINAL_TIME" \
  --dt "$DT" \
  --specfem-root "$SPECFEM_ROOT" \
  --data-dir "$DATA_DIR" \
  --results-dir "$RESULTS_DIR"
setup_end="$(now_seconds)"
setup_seconds="$(elapsed_seconds "$setup_start" "$setup_end")"

mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"/*

ln -sfn "$SPECFEM_ROOT/bin/xmeshfem2D" "$CASE_DIR/xmeshfem2D"
ln -sfn "$SPECFEM_ROOT/bin/xspecfem2D" "$CASE_DIR/xspecfem2D"

cp "$DATA_DIR/Par_file" "$OUTPUT_DIR/"
cp "$DATA_DIR/SOURCE" "$OUTPUT_DIR/"
cp "$DATA_DIR/STATIONS" "$OUTPUT_DIR/"
cp "$DATA_DIR/source_time_function.dat" "$OUTPUT_DIR/"

cd "$CASE_DIR"

mesher_start="$(now_seconds)"
./xmeshfem2D | tee "$LOG_DIR/specfem_mesher.log"
mesh_status=${PIPESTATUS[0]}
if [[ $mesh_status -ne 0 ]]; then
  exit "$mesh_status"
fi
mesher_end="$(now_seconds)"
mesher_seconds="$(elapsed_seconds "$mesher_start" "$mesher_end")"

solver_start="$(now_seconds)"
./xspecfem2D | tee "$LOG_DIR/specfem_solver.log"
solver_status=${PIPESTATUS[0]}
if [[ $solver_status -ne 0 ]]; then
  exit "$solver_status"
fi
solver_end="$(now_seconds)"
solver_seconds="$(elapsed_seconds "$solver_start" "$solver_end")"

mkdir -p "$RESULTS_DIR/raw_seismograms"
rm -f "$RESULTS_DIR/raw_seismograms"/*.semd
cp -f "$OUTPUT_DIR"/*.semd "$RESULTS_DIR/raw_seismograms/"
cp -f "$LOG_DIR/specfem_mesher.log" "$RESULTS_DIR/"
cp -f "$LOG_DIR/specfem_solver.log" "$RESULTS_DIR/"
cp -f "$DATA_DIR/Par_file" "$RESULTS_DIR/Par_file.used"
cp -f "$DATA_DIR/SOURCE" "$RESULTS_DIR/SOURCE.used"
cp -f "$DATA_DIR/STATIONS" "$RESULTS_DIR/STATIONS.used"

total_end="$(now_seconds)"
total_seconds="$(elapsed_seconds "$total_start" "$total_end")"
postprocess_seconds="$(python3 - "$setup_seconds" "$mesher_seconds" "$solver_seconds" "$total_seconds" <<'PY'
import sys
setup = float(sys.argv[1])
mesher = float(sys.argv[2])
solver = float(sys.argv[3])
total = float(sys.argv[4])
print(f"{total - setup - mesher - solver:.6f}")
PY
)"

python3 - "$RESULTS_DIR/timing.json" "$setup_seconds" "$mesher_seconds" "$solver_seconds" "$postprocess_seconds" "$total_seconds" <<'PY'
import json
import sys

output_path = sys.argv[1]
payload = {
    "setup_seconds": float(sys.argv[2]),
    "mesher_seconds": float(sys.argv[3]),
    "solver_seconds": float(sys.argv[4]),
    "postprocess_seconds": float(sys.argv[5]),
    "total_script_seconds": float(sys.argv[6]),
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY

echo "SPECFEM2D results saved to $RESULTS_DIR"
