#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_DIR="$ROOT_DIR/specfem3d_case"
DATA_DIR="$CASE_DIR/DATA"
MESH_DIR="$CASE_DIR/MESH-default"
OUTPUT_DIR="$CASE_DIR/OUTPUT_FILES"
LOG_DIR="$ROOT_DIR/logs"
RESULTS_DIR="$ROOT_DIR/results/specfem3d"
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
SPECFEM_ROOT="${SPECFEM3D_ROOT:-$ROOT_DIR/third_party/specfem3d}"
INSTALL_SCRIPT="$ROOT_DIR/scripts/install_specfem3d_local.sh"

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
Usage: run_specfem3d_case.sh [--edge-length H] [--nproc N] [--source-z Z] [--source-y Y] [--receiver-z Z] [--receiver-y Y] [--source-dir-x X] [--source-dir-y Y] [--source-dir-z Z] [--source-delay T] [--results-dir DIR] [--log-dir DIR] [--specfem-root DIR]
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
    --results-dir)
      RESULTS_DIR="$2"
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

if [[ ! -x "$SPECFEM_ROOT/bin/xdecompose_mesh" || ! -x "$SPECFEM_ROOT/bin/xgenerate_databases" || ! -x "$SPECFEM_ROOT/bin/xspecfem3D" ]]; then
  echo "SPECFEM3D executables were not found at $SPECFEM_ROOT; bootstrapping a local build."
  "$INSTALL_SCRIPT" --dir "$SPECFEM_ROOT"
fi

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
total_start="$(now_seconds)"

setup_start="$(now_seconds)"
python3 "$ROOT_DIR/scripts/setup_specfem3d_case.py" \
  --edge-length "$EDGE_LENGTH" \
  --nproc "$NPROC" \
  --source-z "$SOURCE_Z" \
  --source-y "$SOURCE_Y" \
  --receiver-z "$RECEIVER_Z" \
  --receiver-y "$RECEIVER_Y" \
  --source-dir-x "$SOURCE_DIR_X" \
  --source-dir-y "$SOURCE_DIR_Y" \
  --source-dir-z "$SOURCE_DIR_Z" \
  --source-delay "$SOURCE_DELAY" \
  --specfem-root "$SPECFEM_ROOT" \
  --data-dir "$DATA_DIR" \
  --mesh-dir "$MESH_DIR" \
  --results-dir "$RESULTS_DIR"
setup_end="$(now_seconds)"
setup_seconds="$(elapsed_seconds "$setup_start" "$setup_end")"

mkdir -p "$OUTPUT_DIR" "$CASE_DIR/bin" "$OUTPUT_DIR/DATABASES_MPI"
rm -rf "$OUTPUT_DIR"/*
mkdir -p "$OUTPUT_DIR/DATABASES_MPI"

ln -sfn "$SPECFEM_ROOT/bin/xdecompose_mesh" "$CASE_DIR/bin/xdecompose_mesh"
ln -sfn "$SPECFEM_ROOT/bin/xgenerate_databases" "$CASE_DIR/bin/xgenerate_databases"
ln -sfn "$SPECFEM_ROOT/bin/xspecfem3D" "$CASE_DIR/bin/xspecfem3D"

cp "$DATA_DIR/Par_file" "$OUTPUT_DIR/"
cp "$DATA_DIR/FORCESOLUTION" "$OUTPUT_DIR/"
cp "$DATA_DIR/STATIONS" "$OUTPUT_DIR/"

cd "$CASE_DIR"

decompose_start="$(now_seconds)"
./bin/xdecompose_mesh "$NPROC" ./MESH-default ./OUTPUT_FILES/DATABASES_MPI | tee "$LOG_DIR/specfem3d_decompose.log"
decompose_status=${PIPESTATUS[0]}
if [[ $decompose_status -ne 0 ]]; then
  exit "$decompose_status"
fi
decompose_end="$(now_seconds)"
decompose_seconds="$(elapsed_seconds "$decompose_start" "$decompose_end")"

database_start="$(now_seconds)"
if [[ "$NPROC" -eq 1 ]]; then
  ./bin/xgenerate_databases | tee "$LOG_DIR/specfem3d_databases.log"
  database_status=${PIPESTATUS[0]}
else
  mpirun -np "$NPROC" ./bin/xgenerate_databases | tee "$LOG_DIR/specfem3d_databases.log"
  database_status=${PIPESTATUS[0]}
fi
if [[ $database_status -ne 0 ]]; then
  exit "$database_status"
fi
database_end="$(now_seconds)"
database_seconds="$(elapsed_seconds "$database_start" "$database_end")"

solver_start="$(now_seconds)"
if [[ "$NPROC" -eq 1 ]]; then
  ./bin/xspecfem3D | tee "$LOG_DIR/specfem3d_solver.log"
  solver_status=${PIPESTATUS[0]}
else
  mpirun -np "$NPROC" ./bin/xspecfem3D | tee "$LOG_DIR/specfem3d_solver.log"
  solver_status=${PIPESTATUS[0]}
fi
if [[ $solver_status -ne 0 ]]; then
  exit "$solver_status"
fi
solver_end="$(now_seconds)"
solver_seconds="$(elapsed_seconds "$solver_start" "$solver_end")"

mkdir -p "$RESULTS_DIR/raw_seismograms"
rm -f "$RESULTS_DIR/raw_seismograms"/*.semd
shopt -s nullglob
seismograms=("$OUTPUT_DIR"/*.semd)
if [[ ${#seismograms[@]} -eq 0 ]]; then
  echo "No SPECFEM3D displacement seismograms were produced" >&2
  exit 1
fi
cp -f "${seismograms[@]}" "$RESULTS_DIR/raw_seismograms/"
shopt -u nullglob

cp -f "$LOG_DIR/specfem3d_decompose.log" "$RESULTS_DIR/"
cp -f "$LOG_DIR/specfem3d_databases.log" "$RESULTS_DIR/"
cp -f "$LOG_DIR/specfem3d_solver.log" "$RESULTS_DIR/"
cp -f "$DATA_DIR/Par_file" "$RESULTS_DIR/Par_file.used"
cp -f "$DATA_DIR/FORCESOLUTION" "$RESULTS_DIR/FORCESOLUTION.used"
cp -f "$DATA_DIR/STATIONS" "$RESULTS_DIR/STATIONS.used"

total_end="$(now_seconds)"
total_seconds="$(elapsed_seconds "$total_start" "$total_end")"
postprocess_seconds="$(python3 - "$setup_seconds" "$decompose_seconds" "$database_seconds" "$solver_seconds" "$total_seconds" <<'PY'
import sys
setup = float(sys.argv[1])
decompose = float(sys.argv[2])
database = float(sys.argv[3])
solver = float(sys.argv[4])
total = float(sys.argv[5])
print(f"{total - setup - decompose - database - solver:.6f}")
PY
)"

python3 - "$RESULTS_DIR/timing.json" "$setup_seconds" "$decompose_seconds" "$database_seconds" "$solver_seconds" "$postprocess_seconds" "$total_seconds" <<'PY'
import json
import sys

output_path = sys.argv[1]
payload = {
    "setup_seconds": float(sys.argv[2]),
    "decompose_seconds": float(sys.argv[3]),
    "database_seconds": float(sys.argv[4]),
    "solver_seconds": float(sys.argv[5]),
    "postprocess_seconds": float(sys.argv[6]),
    "total_script_seconds": float(sys.argv[7]),
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY

echo "SPECFEM3D results saved to $RESULTS_DIR"
