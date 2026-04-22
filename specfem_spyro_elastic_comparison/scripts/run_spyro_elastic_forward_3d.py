#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPYRO_REPO = REPO_ROOT / "spyro"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import spyro
except ModuleNotFoundError as exc:
    if exc.name != "spyro":
        raise
    if not SPYRO_REPO.exists():
        raise ModuleNotFoundError(
            f"Could not import spyro from the current environment and no sibling checkout was found at {SPYRO_REPO}"
        ) from exc
    if str(SPYRO_REPO) not in sys.path:
        sys.path.insert(0, str(SPYRO_REPO))
    import spyro


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_SPYRO_RESULTS_DIR = RESULTS_DIR / "spyro_3d"

DOMAIN_LENGTH_Z = 3.0
DOMAIN_LENGTH_X = 3.0
DOMAIN_LENGTH_Y = 3.0

DEFAULT_EDGE_LENGTH = 0.2
DEFAULT_SOURCE_Z = -1.1
DEFAULT_SOURCE_Y = 1.5
DEFAULT_RECEIVER_Z = -1.9
DEFAULT_RECEIVER_Y = 1.5
DEFAULT_SOURCE_DIRECTION_X = 0.7071067811865476
DEFAULT_SOURCE_DIRECTION_Y = 0.7071067811865476
DEFAULT_SOURCE_DIRECTION_Z = 0.0
DEFAULT_SOURCE_DELAY = 0.2
DEFAULT_SOURCE_DELAY_MODE = "explicit"
SOURCE_FREQUENCY = 5.0
DEFAULT_FINAL_TIME = 1.5
DEFAULT_DT = 0.001

SOURCE_X = 1.5
RECEIVER_X_START = 1.2
RECEIVER_X_END = 1.8
DEFAULT_NUM_RECEIVERS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reflective 3D Spyro elastic forward comparison case.",
    )
    parser.add_argument(
        "--edge-length",
        type=float,
        default=DEFAULT_EDGE_LENGTH,
        help="Hexahedral element size used to build the mesh.",
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
        help="Source z-coordinate in Spyro's (z, x, y) convention.",
    )
    parser.add_argument(
        "--source-y",
        type=float,
        default=DEFAULT_SOURCE_Y,
        help="Source y-coordinate in Spyro's (z, x, y) convention.",
    )
    parser.add_argument(
        "--receiver-z",
        type=float,
        default=DEFAULT_RECEIVER_Z,
        help="Receiver z-coordinate in Spyro's (z, x, y) convention.",
    )
    parser.add_argument(
        "--receiver-y",
        type=float,
        default=DEFAULT_RECEIVER_Y,
        help="Receiver y-coordinate in Spyro's (z, x, y) convention.",
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
        default=DEFAULT_SOURCE_DELAY,
        help="Ricker delay in seconds.",
    )
    parser.add_argument(
        "--source-delay-mode",
        choices=("explicit", "specfem-single-force-ricker"),
        default=DEFAULT_SOURCE_DELAY_MODE,
        help=(
            "How to interpret the Spyro delay. 'explicit' uses --source-delay directly; "
            "'specfem-single-force-ricker' uses the effective SPECFEM3D timing for a single point-force Ricker source."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help="Time step in seconds.",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=DEFAULT_FINAL_TIME,
        help="Simulation final time in seconds.",
    )
    parser.add_argument(
        "--num-receivers",
        type=int,
        default=DEFAULT_NUM_RECEIVERS,
        help="Number of receivers along the transect.",
    )
    return parser.parse_args()


def resolve_source_delay(source_delay: float, frequency: float, source_delay_mode: str) -> float:
    if source_delay_mode == "explicit":
        return source_delay
    if source_delay_mode == "specfem-single-force-ricker":
        return 1.2 / frequency
    raise ValueError(f"Unsupported source delay mode: {source_delay_mode}")


def build_case(
    source_z: float,
    source_y: float,
    receiver_z: float,
    receiver_y: float,
    source_direction_xyz: tuple[float, float, float],
    source_delay: float,
    num_receivers: int = DEFAULT_NUM_RECEIVERS,
    dt: float = DEFAULT_DT,
    final_time: float = DEFAULT_FINAL_TIME,
) -> dict:
    source_dir_x, source_dir_y, source_dir_z = source_direction_xyz
    receiver_locations = spyro.create_transect(
        (receiver_z, RECEIVER_X_START, receiver_y),
        (receiver_z, RECEIVER_X_END, receiver_y),
        num_receivers,
    )
    return {
        "options": {
            "cell_type": "Q",
            "variant": "lumped",
            "degree": 4,
            "dimension": 3,
        },
        "parallelism": {
            "type": "automatic",
        },
        "mesh": {
            "length_z": DOMAIN_LENGTH_Z,
            "length_x": DOMAIN_LENGTH_X,
            "length_y": DOMAIN_LENGTH_Y,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(source_z, SOURCE_X, source_y)],
            "frequency": SOURCE_FREQUENCY,
            "delay": source_delay,
            "delay_type": "time",
            "receiver_locations": receiver_locations,
            "amplitude": np.array([source_dir_z, source_dir_x, source_dir_y], dtype=float),
            "use_vertex_only_mesh": True,
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
    source_y: float,
    receiver_z: float,
    receiver_y: float,
    source_direction_xyz: tuple[float, float, float],
    source_delay_requested: float,
    source_delay_effective: float,
    source_delay_mode: str,
) -> str:
    baseline_note = (
        "This run is a 3D extrusion of spyro/notebook_tutorials/elastic_forward.ipynb "
        "with a constant-y receiver line, non-periodic reflective boundaries, "
        "VertexOnlyMesh-based source/receiver projection handled natively by Spyro, and a default mixed x-y point-force direction."
    )
    if (
        source_z == DEFAULT_SOURCE_Z
        and source_y == DEFAULT_SOURCE_Y
        and receiver_z == DEFAULT_RECEIVER_Z
        and receiver_y == DEFAULT_RECEIVER_Y
        and np.allclose(
            np.asarray(source_direction_xyz, dtype=float),
            np.array(
                [
                    DEFAULT_SOURCE_DIRECTION_X,
                    DEFAULT_SOURCE_DIRECTION_Y,
                    DEFAULT_SOURCE_DIRECTION_Z,
                ],
                dtype=float,
            ),
        )
        and np.isclose(source_delay_requested, DEFAULT_SOURCE_DELAY)
        and np.isclose(source_delay_effective, DEFAULT_SOURCE_DELAY)
        and source_delay_mode == DEFAULT_SOURCE_DELAY_MODE
    ):
        return baseline_note
    source_dir_x, source_dir_y, source_dir_z = source_direction_xyz
    return (
        f"{baseline_note[:-1]} The source and receiver coordinates were overridden to "
        f"(z, y)=({source_z:.2f}, {source_y:.2f}) and ({receiver_z:.2f}, {receiver_y:.2f}), "
        f"the source direction xyz was set to ({source_dir_x:.2f}, {source_dir_y:.2f}, {source_dir_z:.2f}), "
        f"and the Spyro source-delay mode '{source_delay_mode}' mapped the requested delay "
        f"{source_delay_requested:.3f} s to the effective delay {source_delay_effective:.3f} s."
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
    uy_trace: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_axis, uz_trace, color="black", linewidth=1.0)
    axes[0].set_ylabel("uz")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Spyro central receiver trace")

    axes[1].plot(time_axis, ux_trace, color="black", linewidth=1.0)
    axes[1].set_ylabel("ux")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, uy_trace, color="black", linewidth=1.0)
    axes[2].set_ylabel("uy")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    source_direction_xyz = (args.source_dir_x, args.source_dir_y, args.source_dir_z)
    if np.linalg.norm(np.asarray(source_direction_xyz, dtype=float)) == 0.0:
        raise ValueError("Source direction cannot be the zero vector")
    effective_source_delay = resolve_source_delay(
        source_delay=args.source_delay,
        frequency=SOURCE_FREQUENCY,
        source_delay_mode=args.source_delay_mode,
    )

    case = build_case(
        source_z=args.source_z,
        source_y=args.source_y,
        receiver_z=args.receiver_z,
        receiver_y=args.receiver_y,
        source_direction_xyz=source_direction_xyz,
        source_delay=effective_source_delay,
        num_receivers=args.num_receivers,
        dt=args.dt,
        final_time=args.final_time,
    )
    wave = spyro.IsotropicWave(case)
    mesh_start = time.perf_counter()
    wave.set_mesh(input_mesh_parameters={"edge_length": args.edge_length, "periodic": False})
    mesh_end = time.perf_counter()
    solve_start = time.perf_counter()
    wave.forward_solve()
    solve_end = time.perf_counter()
    postprocess_start = time.perf_counter()

    from firedrake import COMM_WORLD
    is_rank_zero = COMM_WORLD.rank == 0

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
    mesh_nx = int(round(case["mesh"]["length_x"] / args.edge_length))
    mesh_ny = int(round(case["mesh"]["length_y"] / args.edge_length))
    mesh_nz = int(round(case["mesh"]["length_z"] / args.edge_length))

    if is_rank_zero:
        np.savez(
            output_dir / "spyro_receivers.npz",
            time=time_axis,
            receivers_output=receivers_output,
            receiver_locations=receiver_locations,
            source_locations=source_locations,
            source_wavelet=source_wavelet,
            mesh_num_cells=mesh_num_cells,
            mesh_coordinates=mesh_coordinates,
        )

        metadata = {
            "description": "3D extrusion of spyro/notebook_tutorials/elastic_forward.ipynb with reflective boundaries",
            "notebook_path": str(SPYRO_REPO / "notebook_tutorials" / "elastic_forward.ipynb"),
            "coordinate_order": ["z", "x", "y"],
            "component_order": ["uz", "ux", "uy"],
            "spectral_degree": 4,
            "cell_type": "hexahedral",
            "variant": "lumped",
            "periodicity": "none",
            "source_receiver_projection": "vertex_only_mesh",
            "edge_length": args.edge_length,
            "mesh_num_cells": mesh_num_cells,
            "mesh_shape": {"nx": mesh_nx, "ny": mesh_ny, "nz": mesh_nz},
            "mesh_bounds": {
                "z": [float(mesh_coordinates[:, 0].min()), float(mesh_coordinates[:, 0].max())],
                "x": [float(mesh_coordinates[:, 1].min()), float(mesh_coordinates[:, 1].max())],
                "y": [float(mesh_coordinates[:, 2].min()), float(mesh_coordinates[:, 2].max())],
            },
            "source_location": source_locations[0].tolist(),
            "receiver_start": receiver_locations[0].tolist(),
            "receiver_end": receiver_locations[-1].tolist(),
            "num_receivers": int(receiver_locations.shape[0]),
            "dt": case["time_axis"]["dt"],
            "final_time": case["time_axis"]["final_time"],
            "frequency": case["acquisition"]["frequency"],
            "source_delay": case["acquisition"]["delay"],
            "source_delay_requested": args.source_delay,
            "source_delay_mode": args.source_delay_mode,
            "source_direction_xyz": [
                float(source_direction_xyz[0]),
                float(source_direction_xyz[1]),
                float(source_direction_xyz[2]),
            ],
            "amplitude": case["acquisition"]["amplitude"].tolist(),
            "density": case["synthetic_data"]["density"],
            "vp": case["synthetic_data"]["p_wave_velocity"],
            "vs": case["synthetic_data"]["s_wave_velocity"],
            "note": build_note(
                args.source_z,
                args.source_y,
                args.receiver_z,
                args.receiver_y,
                source_direction_xyz,
                args.source_delay,
                effective_source_delay,
                args.source_delay_mode,
            ),
            "timing_seconds": {
                "mesh_setup_seconds": mesh_end - mesh_start,
                "forward_solve_seconds": solve_end - solve_start,
            },
        }

        save_shot_image(
            receivers_output[:, :, 0],
            time_axis,
            "Spyro 3D shot record - uz",
            output_dir / "spyro_shot_uz.png",
        )
        save_shot_image(
            receivers_output[:, :, 1],
            time_axis,
            "Spyro 3D shot record - ux",
            output_dir / "spyro_shot_ux.png",
        )
        save_shot_image(
            receivers_output[:, :, 2],
            time_axis,
            "Spyro 3D shot record - uy",
            output_dir / "spyro_shot_uy.png",
        )

        central_receiver = receivers_output.shape[1] // 2
        save_trace_image(
            time_axis,
            receivers_output[:, central_receiver, 0],
            receivers_output[:, central_receiver, 1],
            receivers_output[:, central_receiver, 2],
            output_dir / "spyro_central_trace.png",
        )

        total_end = time.perf_counter()
        metadata["timing_seconds"]["postprocessing_seconds"] = total_end - postprocess_start
        metadata["timing_seconds"]["total_script_seconds"] = total_end - total_start
        (output_dir / "spyro_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"Spyro 3D results saved to {output_dir}")
        print(f"Spyro 3D mesh elements: {mesh_num_cells}")
        print(f"Receiver data shape: {receivers_output.shape}")


if __name__ == "__main__":
    main()
