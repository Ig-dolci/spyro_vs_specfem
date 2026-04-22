#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPYRO_REPO = REPO_ROOT / "spyro"
if str(SPYRO_REPO) not in sys.path:
    sys.path.insert(0, str(SPYRO_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spyro


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_SPYRO_RESULTS_DIR = RESULTS_DIR / "spyro"
DEFAULT_SOURCE_Z = -1.1
DEFAULT_SOURCE_X = 1.5
DEFAULT_RECEIVER_Z = -1.9
DEFAULT_RECEIVER_X_START = 1.0
DEFAULT_RECEIVER_X_END = 2.0
DEFAULT_NUM_RECEIVERS = 100
DEFAULT_FINAL_TIME = 1.5
DEFAULT_DT = 0.001
DEFAULT_USE_VERTEX_ONLY_MESH = True
DEFAULT_CELL_TYPE = "Q"
DEFAULT_VARIANT = "lumped"


def canonical_cell_type_label(cell_type: str) -> str:
    normalized = cell_type.strip().lower()
    if normalized in {"q", "quadrilateral", "quadrilaterals"}:
        return "quadrilateral"
    if normalized in {"t", "triangle", "triangles"}:
        return "triangle"
    return normalized


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value for use_vertex_only_mesh, got {value!r}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reflective Spyro elastic forward comparison case.",
    )
    parser.add_argument(
        "--edge-length",
        type=float,
        default=0.02,
        help="Mesh edge length used to build the mesh.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SPYRO_RESULTS_DIR,
        help="Directory where Spyro outputs are written.",
    )
    parser.add_argument(
        "--source-z",
        type=float,
        default=DEFAULT_SOURCE_Z,
        help="Source z-coordinate in Spyro's (z, x) convention.",
    )
    parser.add_argument(
        "--source-x",
        type=float,
        default=DEFAULT_SOURCE_X,
        help="Source x-coordinate in Spyro's (z, x) convention.",
    )
    parser.add_argument(
        "--receiver-z",
        type=float,
        default=DEFAULT_RECEIVER_Z,
        help="Receiver z-coordinate in Spyro's (z, x) convention.",
    )
    parser.add_argument(
        "--receiver-x-start",
        type=float,
        default=DEFAULT_RECEIVER_X_START,
        help="Starting x-coordinate of the receiver transect.",
    )
    parser.add_argument(
        "--receiver-x-end",
        type=float,
        default=DEFAULT_RECEIVER_X_END,
        help="Ending x-coordinate of the receiver transect.",
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
    parser.add_argument(
        "--use-vertex-only-mesh",
        type=parse_bool,
        default=DEFAULT_USE_VERTEX_ONLY_MESH,
        help=(
            "Whether Spyro should use VertexOnlyMesh for source/receiver "
            "projection. Accepts true/false."
        ),
    )
    parser.add_argument(
        "--cell-type",
        type=str,
        default=DEFAULT_CELL_TYPE,
        help="Spyro cell type. Use Q for quadrilateral SEM or T for triangular KMV.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=DEFAULT_VARIANT,
        help="Spyro element variant, e.g. lumped or equispaced.",
    )
    return parser.parse_args()


def build_case(
    source_z: float,
    source_x: float,
    receiver_z: float,
    receiver_x_start: float,
    receiver_x_end: float,
    num_receivers: int,
    final_time: float,
    dt: float,
    use_vertex_only_mesh: bool,
    cell_type: str = DEFAULT_CELL_TYPE,
    variant: str = DEFAULT_VARIANT,
) -> dict:
    receiver_locations = spyro.create_transect(
        (receiver_z, receiver_x_start),
        (receiver_z, receiver_x_end),
        num_receivers,
    )
    return {
        "options": {
            "cell_type": cell_type,
            "variant": variant,
            "degree": 4,
            "dimension": 2,
        },
        "parallelism": {
            "type": "automatic",
        },
        "mesh": {
            "length_z": 3.0,
            "length_x": 3.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(source_z, source_x)],
            "frequency": 5.0,
            "delay": 0.2,
            "delay_type": "time",
            "receiver_locations": receiver_locations,
            "amplitude": np.array([0.0, 1.0]),
            "use_vertex_only_mesh": use_vertex_only_mesh,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": final_time,
            "dt": dt,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "fwi_velocity_model_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
        "synthetic_data": {
            "type": "object",
            "density": 0.1,
            "p_wave_velocity": 1.5,
            "s_wave_velocity": 1.0,
            "real_velocity_file": None,
        },
    }


def build_note(
    source_z: float,
    source_x: float,
    receiver_z: float,
    receiver_x_start: float,
    receiver_x_end: float,
    use_vertex_only_mesh: bool,
) -> str:
    projection_note = (
        "VertexOnlyMesh-based source/receiver projection is enabled in Spyro"
        if use_vertex_only_mesh
        else "VertexOnlyMesh-based source/receiver projection is disabled in Spyro"
    )
    baseline_note = (
        "This run removes the notebook's periodic mesh so the case becomes fully reflective. "
        f"{projection_note}, and PVD writing stays disabled to keep the comparison run lighter."
    )
    if (
        source_z == DEFAULT_SOURCE_Z
        and source_x == DEFAULT_SOURCE_X
        and receiver_z == DEFAULT_RECEIVER_Z
        and receiver_x_start == DEFAULT_RECEIVER_X_START
        and receiver_x_end == DEFAULT_RECEIVER_X_END
    ):
        return baseline_note
    return (
        f"{baseline_note[:-1]} The source/receiver geometry was overridden to "
        f"source=({source_z:.3f}, {source_x:.3f}) and receiver transect "
        f"z={receiver_z:.3f}, x in [{receiver_x_start:.3f}, {receiver_x_end:.3f}] "
        "for a convergence-sensitivity study."
    )


def save_shot_image(
    data: np.ndarray,
    time_axis: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    vlim = float(np.max(np.abs(data)))
    if vlim == 0.0:
        vlim = 1.0

    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(
        data,
        aspect="auto",
        cmap="gray",
        vmin=-vlim,
        vmax=vlim,
        extent=(0, data.shape[1] - 1, time_axis[-1], time_axis[0]),
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Receiver index")
    ax.set_ylabel("Time [s]")
    fig.colorbar(image, ax=ax, pad=0.01, label="Displacement")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_trace_image(
    time_axis: np.ndarray,
    uz_trace: np.ndarray,
    ux_trace: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time_axis, uz_trace, color="black", linewidth=1.0)
    axes[0].set_ylabel("uz")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Spyro central receiver trace")

    axes[1].plot(time_axis, ux_trace, color="black", linewidth=1.0)
    axes[1].set_ylabel("ux")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    case = build_case(
        source_z=args.source_z,
        source_x=args.source_x,
        receiver_z=args.receiver_z,
        receiver_x_start=args.receiver_x_start,
        receiver_x_end=args.receiver_x_end,
        num_receivers=args.num_receivers,
        final_time=args.final_time,
        dt=args.dt,
        use_vertex_only_mesh=args.use_vertex_only_mesh,
        cell_type=args.cell_type,
        variant=args.variant,
    )
    wave = spyro.IsotropicWave(case)
    mesh_start = time.perf_counter()
    wave.set_mesh(input_mesh_parameters={"edge_length": args.edge_length, "periodic": False})
    mesh_end = time.perf_counter()
    solve_start = time.perf_counter()
    wave.forward_solve()
    solve_end = time.perf_counter()
    postprocess_start = time.perf_counter()

    receivers_output = np.asarray(wave.receivers_output, dtype=float)
    time_axis = np.arange(receivers_output.shape[0], dtype=float) * case["time_axis"]["dt"]
    receiver_locations = np.asarray(case["acquisition"]["receiver_locations"], dtype=float)
    source_locations = np.asarray(case["acquisition"]["source_locations"], dtype=float)
    source_wavelet = spyro.full_ricker_wavelet(
        dt=case["time_axis"]["dt"],
        final_time=case["time_axis"]["final_time"],
        frequency=case["acquisition"]["frequency"],
        delay=case["acquisition"]["delay"],
        delay_type=case["acquisition"]["delay_type"],
    )

    mesh_coordinates = wave.mesh.coordinates.dat.data_ro.copy()
    mesh_num_cells = int(wave.mesh.num_cells())
    function_space_dim = int(wave.function_space.dim())
    mesh_nx = int(round(case["mesh"]["length_x"] / args.edge_length))
    mesh_nz = int(round(case["mesh"]["length_z"] / args.edge_length))

    np.savez(
        output_dir / "spyro_receivers.npz",
        time=time_axis,
        receivers_output=receivers_output,
        receiver_locations=receiver_locations,
        source_locations=source_locations,
        source_wavelet=source_wavelet,
        mesh_num_cells=mesh_num_cells,
        function_space_dim=function_space_dim,
        mesh_coordinates=mesh_coordinates,
    )

    metadata = {
        "description": "Reflective-boundary variant of spyro/notebook_tutorials/elastic_forward.ipynb",
        "notebook_path": str(SPYRO_REPO / "notebook_tutorials" / "elastic_forward.ipynb"),
        "coordinate_order": ["z", "x"],
        "component_order": ["uz", "ux"],
        "spectral_degree": 4,
        "cell_type": canonical_cell_type_label(args.cell_type),
        "variant": args.variant,
        "periodicity": "none",
        "source_receiver_projection": (
            "vertex_only_mesh"
            if args.use_vertex_only_mesh
            else "standard_mesh_projection"
        ),
        "use_vertex_only_mesh": args.use_vertex_only_mesh,
        "edge_length": args.edge_length,
        "mesh_num_cells": mesh_num_cells,
        "function_space_dim": function_space_dim,
        "mesh_shape": {"nx": mesh_nx, "nz": mesh_nz},
        "mesh_bounds": {
            "z": [float(mesh_coordinates[:, 0].min()), float(mesh_coordinates[:, 0].max())],
            "x": [float(mesh_coordinates[:, 1].min()), float(mesh_coordinates[:, 1].max())],
        },
        "source_location": source_locations[0].tolist(),
        "receiver_start": receiver_locations[0].tolist(),
        "receiver_end": receiver_locations[-1].tolist(),
        "num_receivers": int(receiver_locations.shape[0]),
        "dt": case["time_axis"]["dt"],
        "final_time": case["time_axis"]["final_time"],
        "frequency": case["acquisition"]["frequency"],
        "amplitude": case["acquisition"]["amplitude"].tolist(),
        "density": case["synthetic_data"]["density"],
        "vp": case["synthetic_data"]["p_wave_velocity"],
        "vs": case["synthetic_data"]["s_wave_velocity"],
        "note": build_note(
            args.source_z,
            args.source_x,
            args.receiver_z,
            args.receiver_x_start,
            args.receiver_x_end,
            args.use_vertex_only_mesh,
        ),
        "timing_seconds": {
            "mesh_setup_seconds": mesh_end - mesh_start,
            "forward_solve_seconds": solve_end - solve_start,
        },
    }
    save_shot_image(
        receivers_output[:, :, 0],
        time_axis,
        "Spyro shot record - uz",
        output_dir / "spyro_shot_uz.png",
    )
    save_shot_image(
        receivers_output[:, :, 1],
        time_axis,
        "Spyro shot record - ux",
        output_dir / "spyro_shot_ux.png",
    )

    central_receiver = receivers_output.shape[1] // 2
    save_trace_image(
        time_axis,
        receivers_output[:, central_receiver, 0],
        receivers_output[:, central_receiver, 1],
        output_dir / "spyro_central_trace.png",
    )
    total_end = time.perf_counter()
    metadata["timing_seconds"]["postprocessing_seconds"] = total_end - postprocess_start
    metadata["timing_seconds"]["total_script_seconds"] = total_end - total_start
    (output_dir / "spyro_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Spyro results saved to {output_dir}")
    print(f"Spyro mesh elements: {mesh_num_cells}")
    print(f"Spyro function-space DOFs: {function_space_dim}")
    print(f"Receiver data shape: {receivers_output.shape}")


if __name__ == "__main__":
    main()
