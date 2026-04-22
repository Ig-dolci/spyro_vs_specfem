#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SPECFEM3D_ROOT = Path(
    os.environ.get("SPECFEM3D_ROOT", ROOT_DIR / "third_party" / "specfem3d")
).expanduser()
DEFAULT_PAR_TEMPLATE_RELATIVE_PATH = (
    Path("EXAMPLES")
    / "applications"
    / "homogeneous_halfspace_HEX8_elastic_no_absorbing"
    / "DATA"
    / "Par_file"
)
DEFAULT_CASE_DIR = ROOT_DIR / "specfem3d_case"
DEFAULT_DATA_DIR = DEFAULT_CASE_DIR / "DATA"
DEFAULT_MESH_DIR = DEFAULT_CASE_DIR / "MESH-default"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "specfem3d"

DOMAIN_X = 3.0
DOMAIN_Y = 3.0
DOMAIN_Z = 3.0
DEFAULT_EDGE_LENGTH = 0.2
DEFAULT_NPROC = 1

FINAL_TIME = 1.5
DT = 0.0005
NSTEP = int(round(FINAL_TIME / DT)) + 1
F0 = 5.0
DELAY = 0.2

DENSITY = 0.1
VP = 1.5
VS = 1.0

SOURCE_X = 1.5
DEFAULT_SOURCE_Y = 1.5
DEFAULT_SOURCE_Z = -1.1
DEFAULT_SOURCE_DIRECTION_X = 0.7071067811865476
DEFAULT_SOURCE_DIRECTION_Y = 0.7071067811865476
DEFAULT_SOURCE_DIRECTION_Z = 0.0

RECEIVER_X_START = 1.2
RECEIVER_X_END = 1.8
DEFAULT_RECEIVER_Y = 1.5
DEFAULT_RECEIVER_Z = -1.9
NUM_RECEIVERS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the reflective SPECFEM3D elastic comparison case.",
    )
    parser.add_argument(
        "--edge-length",
        type=float,
        default=DEFAULT_EDGE_LENGTH,
        help="Structured element size used along x, y, and z.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=DEFAULT_NPROC,
        help="Number of SPECFEM3D MPI processes to configure.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where the SPECFEM3D DATA files are written.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_MESH_DIR,
        help="Directory where the SPECFEM3D MESH-default files are written.",
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
        default=DEFAULT_SPECFEM3D_ROOT,
        help="Root directory of the local SPECFEM3D checkout/build.",
    )
    parser.add_argument(
        "--source-z",
        type=float,
        default=DEFAULT_SOURCE_Z,
        help="Source z-coordinate in SPECFEM3D Cartesian coordinates.",
    )
    parser.add_argument(
        "--source-y",
        type=float,
        default=DEFAULT_SOURCE_Y,
        help="Source y-coordinate in SPECFEM3D Cartesian coordinates.",
    )
    parser.add_argument(
        "--receiver-z",
        type=float,
        default=DEFAULT_RECEIVER_Z,
        help="Receiver z-coordinate in SPECFEM3D Cartesian coordinates.",
    )
    parser.add_argument(
        "--receiver-y",
        type=float,
        default=DEFAULT_RECEIVER_Y,
        help="Receiver y-coordinate in SPECFEM3D Cartesian coordinates.",
    )
    parser.add_argument(
        "--source-dir-x",
        type=float,
        default=DEFAULT_SOURCE_DIRECTION_X,
        help="Physical x component of the force direction.",
    )
    parser.add_argument(
        "--source-dir-y",
        type=float,
        default=DEFAULT_SOURCE_DIRECTION_Y,
        help="Physical y component of the force direction.",
    )
    parser.add_argument(
        "--source-dir-z",
        type=float,
        default=DEFAULT_SOURCE_DIRECTION_Z,
        help="Physical z component of the force direction.",
    )
    parser.add_argument(
        "--source-delay",
        type=float,
        default=DELAY,
        help="Ricker delay in seconds.",
    )
    return parser.parse_args()


def replace_parameter(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(?P<prefix>\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<prefix>{value}", text, count=1)
    if count != 1:
        raise ValueError(f"Could not update parameter {key}")
    return updated


def compute_axis_count(length: float, edge_length: float, axis_name: str) -> int:
    count = int(round(length / edge_length))
    if count <= 0:
        raise ValueError(f"Invalid mesh count for {axis_name}: {count}")
    if not math.isclose(count * edge_length, length, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"edge_length={edge_length} does not divide {axis_name} length {length}")
    return count


def compute_mesh_shape(edge_length: float) -> tuple[int, int, int]:
    nx = compute_axis_count(DOMAIN_X, edge_length, "x")
    ny = compute_axis_count(DOMAIN_Y, edge_length, "y")
    nz = compute_axis_count(DOMAIN_Z, edge_length, "z")
    return nx, ny, nz


def resolve_par_template(specfem_root: Path) -> Path:
    par_template = specfem_root / DEFAULT_PAR_TEMPLATE_RELATIVE_PATH
    if not par_template.exists():
        raise FileNotFoundError(
            f"Could not find SPECFEM3D Par_file template at {par_template}. "
            f"Install SPECFEM3D locally under {ROOT_DIR / 'third_party' / 'specfem3d'} "
            "or pass --specfem-root / set SPECFEM3D_ROOT."
        )
    return par_template


def build_par_file(par_template: Path, nproc: int) -> str:
    text = par_template.read_text(encoding="utf-8")
    replacements = {
        "SIMULATION_TYPE": "1",
        "NOISE_TOMOGRAPHY": "0",
        "SAVE_FORWARD": ".false.",
        "SUPPRESS_UTM_PROJECTION": ".true.",
        "UTM_PROJECTION_ZONE": "0",
        "NPROC": str(nproc),
        "NSTEP": str(NSTEP),
        "DT": f"{DT:.4f}d0",
        "LTS_MODE": ".false.",
        "MODEL": "default",
        "ATTENUATION": ".false.",
        "ANISOTROPY": ".false.",
        "GRAVITY": ".false.",
        "PML_CONDITIONS": ".false.",
        "PML_INSTEAD_OF_FREE_SURFACE": ".false.",
        "STACEY_ABSORBING_CONDITIONS": ".false.",
        "STACEY_INSTEAD_OF_FREE_SURFACE": ".false.",
        "BOTTOM_FREE_SURFACE": ".false.",
        "SAVE_MESH_FILES": ".false.",
        "LOCAL_PATH": "./OUTPUT_FILES/DATABASES_MPI",
        "NTSTEP_BETWEEN_OUTPUT_INFO": "250",
        "USE_SOURCES_RECEIVERS_Z": ".true.",
        "USE_FORCE_POINT_SOURCE": ".true.",
        "USE_RICKER_TIME_FUNCTION": ".true.",
        "USE_EXTERNAL_SOURCE_FILE": ".false.",
        "PRINT_SOURCE_TIME_FUNCTION": ".false.",
        "NTSTEP_BETWEEN_OUTPUT_SEISMOS": str(NSTEP),
        "NTSTEP_BETWEEN_OUTPUT_SAMPLE": "1",
        "SAVE_SEISMOGRAMS_DISPLACEMENT": ".true.",
        "SAVE_SEISMOGRAMS_VELOCITY": ".false.",
        "SAVE_SEISMOGRAMS_ACCELERATION": ".false.",
        "SAVE_SEISMOGRAMS_PRESSURE": ".false.",
        "SAVE_SEISMOGRAMS_STRAIN": ".false.",
        "USE_BINARY_FOR_SEISMOGRAMS": ".false.",
        "SU_FORMAT": ".false.",
        "ASDF_FORMAT": ".false.",
        "HDF5_FORMAT": ".false.",
        "WRITE_SEISMOGRAMS_BY_MAIN": ".true.",
        "SAVE_ALL_SEISMOS_IN_ONE_FILE": ".false.",
        "OUTPUT_ENERGY": ".false.",
        "NUMBER_OF_SIMULTANEOUS_RUNS": "1",
        "BROADCAST_SAME_MESH_AND_MODEL": ".false.",
    }
    for key, value in replacements.items():
        text = replace_parameter(text, key, value)
    return text


def build_force_solution(
    source_z: float,
    source_y: float,
    source_direction_xyz: tuple[float, float, float],
    source_delay: float,
) -> str:
    source_dir_x, source_dir_y, source_dir_z = source_direction_xyz
    return "\n".join(
        [
            "FORCE  001",
            f"time shift:     {source_delay:.4f}",
            f"hdurorf0:       {F0:.1f}",
            f"latorUTM:       {source_y:.6f}",
            f"longorUTM:      {SOURCE_X:.6f}",
            f"depth:          {source_z:.6f}",
            "source time function:            1",
            "factor force source:             1.d0",
            f"component dir vect source E:     {source_dir_x:.8f}d0",
            f"component dir vect source N:     {source_dir_y:.8f}d0",
            f"component dir vect source Z_UP:  {source_dir_z:.8f}d0",
            "",
        ]
    )


def build_stations_file(receiver_z: float, receiver_y: float) -> str:
    lines = []
    dx = (RECEIVER_X_END - RECEIVER_X_START) / (NUM_RECEIVERS - 1)
    for index in range(NUM_RECEIVERS):
        x_value = RECEIVER_X_START + index * dx
        lines.append(f"S{index + 1:04d} AA {receiver_y:.6f} {x_value:.6f} 0.0 {receiver_z:.6f}")
    return "\n".join(lines) + "\n"


def node_id(nx: int, ny: int, i: int, j: int, k: int) -> int:
    return k * (nx + 1) * (ny + 1) + i * (ny + 1) + j + 1


def element_id(nx: int, ny: int, i: int, j: int, k: int) -> int:
    return k * nx * ny + i * ny + j + 1


def generate_mesh_default(mesh_dir: Path, edge_length: float) -> dict[str, int | float]:
    nx, ny, nz = compute_mesh_shape(edge_length)
    dx = DOMAIN_X / nx
    dy = DOMAIN_Y / ny
    dz = DOMAIN_Z / nz

    mesh_dir.mkdir(parents=True, exist_ok=True)

    num_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    nodes_lines = [str(num_nodes)]
    for k in range(nz + 1):
        z_value = -k * dz
        for i in range(nx + 1):
            x_value = i * dx
            for j in range(ny + 1):
                y_value = j * dy
                nid = node_id(nx, ny, i, j, k)
                nodes_lines.append(f"{nid} {x_value:.9f} {y_value:.9f} {z_value:.9f}")
    (mesh_dir / "nodes_coords_file").write_text("\n".join(nodes_lines) + "\n", encoding="utf-8")

    num_elements = nx * ny * nz
    mesh_lines = [str(num_elements)]
    materials_lines = []
    top_faces: list[str] = []
    bottom_faces: list[str] = []
    xmin_faces: list[str] = []
    xmax_faces: list[str] = []
    ymin_faces: list[str] = []
    ymax_faces: list[str] = []

    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                eid = element_id(nx, ny, i, j, k)
                # SPECFEM3D HEX8 ordering follows the same convention used by the
                # official MESH-default examples: one x-face first, then the
                # opposite x-face.
                n1 = node_id(nx, ny, i, j, k)
                n2 = node_id(nx, ny, i, j, k + 1)
                n3 = node_id(nx, ny, i, j + 1, k + 1)
                n4 = node_id(nx, ny, i, j + 1, k)
                n5 = node_id(nx, ny, i + 1, j, k)
                n6 = node_id(nx, ny, i + 1, j, k + 1)
                n7 = node_id(nx, ny, i + 1, j + 1, k + 1)
                n8 = node_id(nx, ny, i + 1, j + 1, k)
                mesh_lines.append(f"{eid} {n1} {n2} {n3} {n4} {n5} {n6} {n7} {n8}")
                materials_lines.append(f"{eid} 1")

                if k == 0:
                    top_faces.append(f"{eid} {n5} {n8} {n4} {n1}")
                if k == nz - 1:
                    bottom_faces.append(f"{eid} {n7} {n6} {n2} {n3}")
                if i == 0:
                    xmin_faces.append(f"{eid} {n4} {n3} {n2} {n1}")
                if i == nx - 1:
                    xmax_faces.append(f"{eid} {n5} {n6} {n7} {n8}")
                if j == 0:
                    ymin_faces.append(f"{eid} {n1} {n2} {n6} {n5}")
                if j == ny - 1:
                    ymax_faces.append(f"{eid} {n8} {n7} {n3} {n4}")

    (mesh_dir / "mesh_file").write_text("\n".join(mesh_lines) + "\n", encoding="utf-8")
    (mesh_dir / "materials_file").write_text("\n".join(materials_lines) + "\n", encoding="utf-8")
    (mesh_dir / "nummaterial_velocity_file").write_text(
        f"2 1 {DENSITY:.6f} {VP:.6f} {VS:.6f} 9999.0 9999.0 0\n",
        encoding="utf-8",
    )

    surface_map = {
        "free_or_absorbing_surface_file_zmax": top_faces,
        "absorbing_surface_file_bottom": bottom_faces,
        "absorbing_surface_file_xmin": xmin_faces,
        "absorbing_surface_file_xmax": xmax_faces,
        "absorbing_surface_file_ymin": ymin_faces,
        "absorbing_surface_file_ymax": ymax_faces,
    }
    for file_name, entries in surface_map.items():
        content = [str(len(entries)), *entries]
        (mesh_dir / file_name).write_text("\n".join(content) + "\n", encoding="utf-8")

    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "num_nodes": num_nodes,
        "num_elements": num_elements,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }


def build_note(
    source_z: float,
    source_y: float,
    receiver_z: float,
    receiver_y: float,
    source_direction_xyz: tuple[float, float, float],
    source_delay: float,
) -> str:
    baseline_note = (
        "This case is a 3D extrusion of the Spyro elastic notebook geometry with a constant-y "
        "receiver line, reflective/free boundaries on all six faces, and a default mixed x-y point-force direction."
    )
    if (
        source_z == DEFAULT_SOURCE_Z
        and source_y == DEFAULT_SOURCE_Y
        and receiver_z == DEFAULT_RECEIVER_Z
        and receiver_y == DEFAULT_RECEIVER_Y
        and all(
            math.isclose(component, default_component, rel_tol=0.0, abs_tol=1.0e-12)
            for component, default_component in zip(
                source_direction_xyz,
                (
                    DEFAULT_SOURCE_DIRECTION_X,
                    DEFAULT_SOURCE_DIRECTION_Y,
                    DEFAULT_SOURCE_DIRECTION_Z,
                ),
                strict=True,
            )
        )
        and math.isclose(source_delay, DELAY, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        return baseline_note
    source_dir_x, source_dir_y, source_dir_z = source_direction_xyz
    return (
        f"{baseline_note} The source and receiver coordinates were overridden to "
        f"(z, y)=({source_z:.2f}, {source_y:.2f}) and ({receiver_z:.2f}, {receiver_y:.2f}), "
        f"the source direction xyz was set to ({source_dir_x:.2f}, {source_dir_y:.2f}, {source_dir_z:.2f}), "
        f"and the Ricker delay was set to {source_delay:.3f} s."
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    mesh_dir = args.mesh_dir.resolve()
    results_dir = args.results_dir.resolve()
    specfem_root = args.specfem_root.resolve()
    par_template = resolve_par_template(specfem_root)
    source_direction_xyz = (args.source_dir_x, args.source_dir_y, args.source_dir_z)
    if math.isclose(sum(component * component for component in source_direction_xyz), 0.0, abs_tol=1.0e-20):
        raise ValueError("Source direction cannot be the zero vector")

    data_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    mesh_info = generate_mesh_default(mesh_dir, args.edge_length)

    (data_dir / "Par_file").write_text(build_par_file(par_template, args.nproc), encoding="utf-8")
    (data_dir / "FORCESOLUTION").write_text(
        build_force_solution(
            args.source_z,
            args.source_y,
            source_direction_xyz,
            args.source_delay,
        ),
        encoding="utf-8",
    )
    (data_dir / "STATIONS").write_text(
        build_stations_file(args.receiver_z, args.receiver_y),
        encoding="utf-8",
    )

    metadata = {
        "description": "SPECFEM3D case assembled to approximate a reflective 3D extrusion of Spyro's elastic_forward notebook",
        "spectral_degree": 4,
        "ngllx": 5,
        "coordinate_order": ["x", "y", "z"],
        "source_file_coordinate_order": ["y", "x", "z"],
        "station_file_coordinate_order": ["y", "x", "elevation", "z"],
        "domain": {
            "x": [0.0, DOMAIN_X],
            "y": [0.0, DOMAIN_Y],
            "z": [-DOMAIN_Z, 0.0],
        },
        "mesh": {
            "nx": int(mesh_info["nx"]),
            "ny": int(mesh_info["ny"]),
            "nz": int(mesh_info["nz"]),
            "edge_length": args.edge_length,
            "effective_spacing": {
                "dx": float(mesh_info["dx"]),
                "dy": float(mesh_info["dy"]),
                "dz": float(mesh_info["dz"]),
            },
            "num_nodes": int(mesh_info["num_nodes"]),
            "expected_total_elements": int(mesh_info["num_elements"]),
            "path": str(mesh_dir),
        },
        "time_axis": {
            "dt": DT,
            "nstep": NSTEP,
            "final_time": (NSTEP - 1) * DT,
        },
        "material": {
            "density": DENSITY,
            "vp": VP,
            "vs": VS,
            "attenuation": False,
        },
        "source": {
            "x": SOURCE_X,
            "y": args.source_y,
            "z": args.source_z,
            "time_shift": args.source_delay,
            "frequency": F0,
            "force_direction_xyz": [
                float(source_direction_xyz[0]),
                float(source_direction_xyz[1]),
                float(source_direction_xyz[2]),
            ],
            "force_direction_specfem_enz_up": [
                float(source_direction_xyz[0]),
                float(source_direction_xyz[1]),
                float(source_direction_xyz[2]),
            ],
        },
        "receivers": {
            "count": NUM_RECEIVERS,
            "x_start": RECEIVER_X_START,
            "x_end": RECEIVER_X_END,
            "y": args.receiver_y,
            "z": args.receiver_z,
        },
        "boundary_conditions": {
            "periodic": False,
            "absorbing_boundaries": False,
            "top": "free",
            "bottom": "free",
            "xmin": "free",
            "xmax": "free",
            "ymin": "free",
            "ymax": "free",
        },
        "note": build_note(
            args.source_z,
            args.source_y,
            args.receiver_z,
            args.receiver_y,
            source_direction_xyz,
            args.source_delay,
        ),
    }
    (results_dir / "specfem3d_case_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"SPECFEM3D case files written to {data_dir}")
    print(f"SPECFEM3D mesh files written to {mesh_dir}")


if __name__ == "__main__":
    main()
