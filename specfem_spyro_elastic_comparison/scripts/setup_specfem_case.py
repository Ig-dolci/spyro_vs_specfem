#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SPECFEM2D_ROOT = Path(
    os.environ.get("SPECFEM2D_ROOT", ROOT_DIR / "third_party" / "specfem2d")
).expanduser()
DEFAULT_CASE_DIR = ROOT_DIR / "specfem_case"
DEFAULT_DATA_DIR = DEFAULT_CASE_DIR / "DATA"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "specfem"

DOMAIN_X = 3.0
DOMAIN_Z = 3.0
DEFAULT_EDGE_LENGTH = 0.02
DEFAULT_FINAL_TIME = 1.0
DEFAULT_DT = 0.0005
F0 = 5.0
DEFAULT_SOURCE_Z = -1.1
SOURCE_X = 1.5
DEFAULT_RECEIVER_Z = -1.9
RECEIVER_X_START = 1.
RECEIVER_X_END = 2.0
DEFAULT_NUM_RECEIVERS = 100
PML_THICKNESS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the reflective SPECFEM2D elastic comparison case.",
    )
    parser.add_argument(
        "--edge-length",
        type=float,
        default=DEFAULT_EDGE_LENGTH,
        help="Structured element size used along x and z.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where the SPECFEM2D DATA files are written.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where case metadata is written.",
    )
    parser.add_argument(
        "--specfem-root",
        type=Path,
        default=DEFAULT_SPECFEM2D_ROOT,
        help="Root directory of the local SPECFEM2D checkout/build.",
    )
    parser.add_argument(
        "--source-z",
        type=float,
        default=DEFAULT_SOURCE_Z,
        help="Source z-coordinate in SPECFEM2D coordinates.",
    )
    parser.add_argument(
        "--receiver-z",
        type=float,
        default=DEFAULT_RECEIVER_Z,
        help="Receiver z-coordinate in SPECFEM2D coordinates.",
    )
    parser.add_argument(
        "--num-receivers",
        type=int,
        default=DEFAULT_NUM_RECEIVERS,
        help="Number of uniformly spaced receivers between x=1.0 and x=2.0.",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=DEFAULT_FINAL_TIME,
        help="Physical simulation time in seconds.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help="Time step in seconds.",
    )
    return parser.parse_args()


def compute_mesh_shape(edge_length: float) -> tuple[int, int]:
    nx = int(round(DOMAIN_X / edge_length))
    nz = int(round(DOMAIN_Z / edge_length))
    if not math.isclose(nx * edge_length, DOMAIN_X, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"edge_length={edge_length} does not divide DOMAIN_X={DOMAIN_X}")
    if not math.isclose(nz * edge_length, DOMAIN_Z, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"edge_length={edge_length} does not divide DOMAIN_Z={DOMAIN_Z}")
    return nx, nz


def ricker_wavelet(
    time_value: float,
    frequency: float,
    delay: float,
) -> float:
    shifted_time = time_value - delay
    scaled = (math.pi * frequency * shifted_time) ** 2
    return (1.0 - 2.0 * scaled) * math.exp(-scaled)


def compute_nstep(final_time: float, dt: float) -> int:
    if final_time <= 0.0:
        raise ValueError(f"final_time must be positive, got {final_time}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    return int(round(final_time / dt)) + 1


def full_ricker_wavelet(final_time: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    time_axis = np.arange(compute_nstep(final_time, dt), dtype=float) * dt
    wavelet = np.array(
        [ricker_wavelet(time_value, F0, delay=0.2) for time_value in time_axis],
        dtype=float,
    )
    return time_axis, wavelet


def format_fortran_real(value: float) -> str:
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return f"{text}d0"


def replace_parameter(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(?P<prefix>\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<prefix>{value}", text, count=1)
    if count != 1:
        raise ValueError(f"Could not update parameter {key}")
    return updated


def resolve_base_case_dir(specfem_root: Path) -> Path:
    base_case_dir = (
        specfem_root / "EXAMPLES" / "benchmarks" / "semi_infinite_homogeneous" / "DATA"
    )
    if not base_case_dir.exists():
        raise FileNotFoundError(
            f"Could not find SPECFEM2D example DATA directory at {base_case_dir}. "
            f"Install SPECFEM2D locally under {ROOT_DIR / 'third_party' / 'specfem2d'} "
            "or pass --specfem-root / set SPECFEM2D_ROOT."
        )
    return base_case_dir


def build_par_file(
    nx: int,
    nz: int,
    edge_length: float,
    dt: float,
    nstep: int,
    base_case_dir: Path,
) -> str:
    text = (base_case_dir / "Par_file").read_text(encoding="utf-8")
    replacements = {
        "title": f"Spyro elastic forward comparison with fully reflective boundaries (h={edge_length:.5f})",
        "NSTEP": str(nstep),
        "DT": format_fortran_real(dt),
        "time_stepping_scheme": "1",
        "NTSTEP_BETWEEN_OUTPUT_SEISMOS": str(nstep),
        "NTSTEP_BETWEEN_OUTPUT_SAMPLE": "1",
        "USER_T0": "0.0d0",
        "save_ASCII_seismograms": ".true.",
        "save_binary_seismograms_single": ".false.",
        "save_binary_seismograms_double": ".false.",
        "SU_FORMAT": ".false.",
        "use_existing_STATIONS": ".true.",
        "nreceiversets": "0",
        "PML_BOUNDARY_CONDITIONS": ".false.",
        "NELEM_PML_THICKNESS": str(PML_THICKNESS),
        "STACEY_ABSORBING_CONDITIONS": ".false.",
        "ADD_PERIODIC_CONDITIONS": ".false.",
        "PERIODIC_HORIZ_DIST": "3.0d0",
        "interfacesfile": "interfaces_elastic_spyro.dat",
        "xmin": "0.d0",
        "xmax": "3.d0",
        "nx": str(nx),
        "absorbbottom": ".false.",
        "absorbright": ".false.",
        "absorbtop": ".false.",
        "absorbleft": ".false.",
        "NTSTEP_BETWEEN_OUTPUT_INFO": "250",
        "SAVE_MESH_FILES": ".false.",
        "OUTPUT_ENERGY": ".false.",
        "NTSTEP_BETWEEN_OUTPUT_ENERGY": "250",
        "NTSTEP_BETWEEN_OUTPUT_IMAGES": str(nstep),
    }
    for key, value in replacements.items():
        text = replace_parameter(text, key, value)

    text = re.sub(
        r"^1 1 2700\.d0 3000\.d0 1732\.05d0 0 0 20\. 10\. 0 0 0 0 0 0$",
        "1 1 0.1d0 1.5d0 1.0d0 0 0 9999.d0 9999.d0 0 0 0 0 0 0",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^1 50 1\s+50 1$",
        f"1 {nx} 1 {nz} 1",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return text


def build_source_file(source_z: float, base_case_dir: Path) -> str:
    text = (base_case_dir / "SOURCE").read_text(encoding="utf-8")
    replacements = {
        "xs": "1.5",
        "zs": f"{source_z:.6f}",
        "source_type": "1",
        "time_function_type": "8",
        "name_of_source_file": "DATA/source_time_function.dat",
        "f0": "5.0",
        "tshift": "0.0",
        "anglesource": "90.0",
        "factor": "1.0d0",
    }
    for key, value in replacements.items():
        text = replace_parameter(text, key, value)
    return text


def build_interfaces_file(nz: int) -> str:
    return "\n".join(
        [
            "#",
            "# number of interfaces",
            "#",
            " 2",
            "#",
            "# interface number 1 (bottom of the mesh)",
            "#",
            " 2",
            " 0 -3",
            " 3 -3",
            "#",
            "# interface number 2",
            "#",
            " 2",
            " 0 0",
            " 3 0",
            "#",
            "# number of spectral elements in the vertical direction",
            "#",
            f" {nz}",
            "",
        ]
    )


def build_stations_file(receiver_z: float, num_receivers: int) -> str:
    x_values = np.linspace(RECEIVER_X_START, RECEIVER_X_END, num_receivers)
    lines = []
    for index, x_value in enumerate(x_values, start=1):
        lines.append(f"S{index:04d} AA {x_value:.6f} {receiver_z:.6f} 0.0 0.0")
    return "\n".join(lines) + "\n"


def build_note(source_z: float, receiver_z: float) -> str:
    baseline_note = (
        "This case removes both PML and horizontal periodicity so all four boundaries are reflective / non-absorbing."
    )
    if source_z == DEFAULT_SOURCE_Z and receiver_z == DEFAULT_RECEIVER_Z:
        return baseline_note
    return (
        f"{baseline_note} The source and receiver depths were overridden to "
        f"{abs(source_z):.2f} and {abs(receiver_z):.2f} in magnitude for a convergence-sensitivity study."
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    specfem_root = args.specfem_root.resolve()
    base_case_dir = resolve_base_case_dir(specfem_root)
    nx, nz = compute_mesh_shape(args.edge_length)
    nstep = compute_nstep(args.final_time, args.dt)

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "Par_file").write_text(
        build_par_file(
            nx=nx,
            nz=nz,
            edge_length=args.edge_length,
            dt=args.dt,
            nstep=nstep,
            base_case_dir=base_case_dir,
        ),
        encoding="utf-8",
    )
    (data_dir / "SOURCE").write_text(
        build_source_file(args.source_z, base_case_dir=base_case_dir),
        encoding="utf-8",
    )
    (data_dir / "interfaces_elastic_spyro.dat").write_text(
        build_interfaces_file(nz=nz),
        encoding="utf-8",
    )
    (data_dir / "STATIONS").write_text(
        build_stations_file(args.receiver_z, args.num_receivers),
        encoding="utf-8",
    )

    time_axis, wavelet = full_ricker_wavelet(args.final_time, args.dt)
    np.savetxt(
        data_dir / "source_time_function.dat",
        np.column_stack([time_axis, wavelet]),
        fmt="%.9f %.16e",
    )

    metadata = {
        "description": "SPECFEM2D case assembled to approximate a fully reflective variant of the Spyro elastic_forward notebook",
        "spectral_degree": 4,
        "ngllx": 5,
        "coordinate_order": ["x", "z"],
        "component_order": ["ux", "uz"],
        "domain": {"x": [0.0, DOMAIN_X], "z": [-DOMAIN_Z, 0.0]},
        "mesh": {
            "nx": nx,
            "nz": nz,
            "edge_length": args.edge_length,
            "expected_total_elements": nx * nz,
        },
        "source": {
            "x": SOURCE_X,
            "z": args.source_z,
            "force_angle_degrees_from_vertical": 90.0,
            "time_function_type": 8,
            "wavelet_delay": 0.2,
        },
        "time_axis": {
            "dt": args.dt,
            "final_time": args.final_time,
            "nstep": nstep,
        },
        "receivers": {
            "count": args.num_receivers,
            "x_start": RECEIVER_X_START,
            "x_end": RECEIVER_X_END,
            "z": args.receiver_z,
        },
        "boundary_conditions": {
            "periodic_horizontal": False,
            "periodic_vertical": False,
            "absorbing_boundaries": False,
            "top_bottom": "non-absorbing / free",
            "left_right": "non-absorbing / free",
        },
        "note": build_note(args.source_z, args.receiver_z),
    }
    (results_dir / "specfem_case_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"SPECFEM2D case files written to {data_dir}")


if __name__ == "__main__":
    main()
