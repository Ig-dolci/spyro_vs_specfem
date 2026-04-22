#!/usr/bin/env python3
"""3D global self-convergence study with top free surface + Dirichlet BCs.

Replaces the MMS-based 3D global convergence study with a self-convergence
approach using a Ricker wavelet source and a fine reference mesh.

The error is the relative L2 norm of the displacement field at final time,
evaluated on each coarse mesh's GLL quadrature grid with the reference
solution interpolated via tensor-product Lagrange basis functions.

Usage (serial):
    python run_spyro_global_selfconv_top_free_3d.py

Usage (MPI, 4 cores):
    mpiexec -np 4 python run_spyro_global_selfconv_top_free_3d.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPYRO_REPO = REPO_ROOT / "spyro"
if str(SPYRO_REPO) not in sys.path:
    sys.path.insert(0, str(SPYRO_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import spyro
from compare_spyro_specfem import relative_l2, resolve_early_window_end
import firedrake
from firedrake import Constant

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "global_selfconv_top_free_3d"
DEFAULT_SPACE_HS = [0.125, 0.1, 0.0625, 0.05]
DEFAULT_REFERENCE_H = 0.04
# KMV p=3 vs spectral p=4: similar DOF density, h's nearly equal
DEFAULT_SPACE_HS_KMV = [0.125, 0.1, 1.0/15, 1.0/19]
DEFAULT_REFERENCE_H_KMV = 1.0/24
DEFAULT_FINAL_TIME = 0.5
DEFAULT_DT = 0.0005
DEFAULT_DT_KMV = 0.0002


def gll_nodes_and_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return *n* Gauss-Lobatto-Legendre nodes and weights on [-1, 1]."""
    if n < 2:
        raise ValueError("Need at least 2 GLL nodes")
    if n == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    from numpy.polynomial import legendre as L
    c = np.zeros(n)
    c[n - 1] = 1.0
    dc = L.legder(c)
    interior = np.sort(L.legroots(dc).real)
    nodes = np.empty(n)
    nodes[0] = -1.0
    nodes[1:-1] = interior
    nodes[-1] = 1.0
    pn = L.legval(nodes, c)
    weights = 2.0 / (n * (n - 1) * pn ** 2)
    return nodes, weights


@dataclass(frozen=True)
class GlobalSelfconvConfig3D:
    length_z: float = 1.0
    length_x: float = 1.0
    length_y: float = 1.0
    z_min: float = -1.0
    x_min: float = 0.0
    y_min: float = 0.0
    source_z: float = -0.35
    source_x: float = 0.51
    source_y: float = 0.47
    source_dir_z: float = 0.0
    source_dir_x: float = 0.7071067811865476
    source_dir_y: float = 0.7071067811865476
    receiver_z: float = -0.30
    receiver_x_start: float = 0.20
    receiver_x_end: float = 0.80
    receiver_y: float = 0.47
    num_receivers: int = 21
    density: float = 0.1
    p_wave_velocity: float = 1.5
    s_wave_velocity: float = 1.0
    degree: int = 4
    cell_type: str = "Q"
    variant: str = "lumped"
    dimension: int = 3
    final_time: float = DEFAULT_FINAL_TIME
    dt: float = DEFAULT_DT
    frequency: float = 5.0
    source_delay: float = 0.2
    ngllx: int = 5
    use_vertex_only_mesh: bool = True


# ------------------------------------------------------------------
# MPI helpers
# ------------------------------------------------------------------
def mpi_world():
    if MPI is None:
        return None
    return MPI.COMM_WORLD


def mpi_is_root() -> bool:
    comm = mpi_world()
    return comm is None or comm.rank == 0


def mpi_barrier() -> None:
    comm = mpi_world()
    if comm is not None:
        comm.barrier()


def mpi_broadcast(value):
    comm = mpi_world()
    if comm is None or comm.size == 1:
        return value
    return comm.bcast(value if comm.rank == 0 else None, root=0)


def mpi_sum_int(value: int) -> int:
    comm = mpi_world()
    if comm is None or comm.size == 1:
        return int(value)
    return int(comm.allreduce(int(value), op=MPI.SUM))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 3D global self-convergence study with top free surface "
            "and Dirichlet BCs on all other faces."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--space-hs", nargs="+", type=float, default=DEFAULT_SPACE_HS)
    parser.add_argument("--reference-h", type=float, default=DEFAULT_REFERENCE_H)
    parser.add_argument("--final-time", type=float, default=DEFAULT_FINAL_TIME)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument(
        "--method", choices=["spectral", "kmv"], default="spectral",
        help="Element method: spectral (Q+lumped) or kmv (T+KMV).",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


# ------------------------------------------------------------------
# Mesh and quadrature helpers
# ------------------------------------------------------------------
def compute_mesh_shape(config: GlobalSelfconvConfig3D, h: float) -> tuple[int, int, int]:
    nx = int(round(config.length_x / h))
    nz = int(round(config.length_z / h))
    ny = int(round(config.length_y / h))
    if not math.isclose(nx * h, config.length_x, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"h={h} does not divide length_x={config.length_x}")
    if not math.isclose(nz * h, config.length_z, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"h={h} does not divide length_z={config.length_z}")
    if not math.isclose(ny * h, config.length_y, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"h={h} does not divide length_y={config.length_y}")
    return nx, nz, ny


def edge_length_tag(h: float) -> str:
    return f"{h:.8f}".rstrip("0").rstrip(".").replace(".", "p")


def coord_key_3d(z: float, x: float, y: float) -> tuple[float, float, float]:
    return (round(float(z), 10), round(float(x), 10), round(float(y), 10))


def canonical_h_values(space_hs: list[float], reference_h: float) -> list[float]:
    unique = sorted({float(v) for v in space_hs}, reverse=True)
    if len(unique) < 3:
        raise ValueError("Provide at least three distinct h values.")
    if any(v <= 0.0 for v in unique):
        raise ValueError("All h values must be strictly positive.")
    if reference_h <= 0.0:
        raise ValueError("reference_h must be strictly positive.")
    if any(v <= reference_h for v in unique):
        raise ValueError("All sweep h values must be coarser than reference_h.")
    return unique


def build_global_quadrature_grid_3d(
    config: GlobalSelfconvConfig3D,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    nx, nz, ny = compute_mesh_shape(config, h)
    nodes, weights_1d = gll_nodes_and_weights(config.ngllx)
    jacobian = 0.125 * h * h * h  # 3D
    point_map: dict[tuple[float, float, float], tuple[float, float, float]] = {}
    weight_map: dict[tuple[float, float, float], float] = {}

    for ex in range(nx):
        xl = config.x_min + ex * h
        for ez in range(nz):
            zb = config.z_min + ez * h
            for ey in range(ny):
                yl = config.y_min + ey * h
                for i, nx_node in enumerate(nodes):
                    xc = xl + 0.5 * (nx_node + 1.0) * h
                    for j, nz_node in enumerate(nodes):
                        zc = zb + 0.5 * (nz_node + 1.0) * h
                        for k, ny_node in enumerate(nodes):
                            yc = yl + 0.5 * (ny_node + 1.0) * h
                            key = coord_key_3d(zc, xc, yc)
                            point_map[key] = (zc, xc, yc)
                            w = jacobian * weights_1d[i] * weights_1d[j] * weights_1d[k]
                            weight_map[key] = weight_map.get(key, 0.0) + w

    ordered_keys = sorted(point_map.keys())
    points = np.asarray([point_map[k] for k in ordered_keys], dtype=float)
    weights = np.asarray([weight_map[k] for k in ordered_keys], dtype=float)
    return points, weights


def lagrange_basis(xi: float, nodes: np.ndarray) -> np.ndarray:
    match = np.where(np.isclose(xi, nodes, atol=1.0e-12, rtol=0.0))[0]
    if match.size:
        basis = np.zeros_like(nodes)
        basis[match[0]] = 1.0
        return basis
    basis = np.empty_like(nodes)
    for idx, node in enumerate(nodes):
        val = 1.0
        for jdx, other in enumerate(nodes):
            if jdx == idx:
                continue
            val *= (xi - other) / (node - other)
        basis[idx] = val
    return basis


class StructuredReferenceField3D:
    """Reference solution stored on 3D GLL grid with tensor-product interpolation."""

    def __init__(
        self,
        config: GlobalSelfconvConfig3D,
        h: float,
        points: np.ndarray,
        field_values: np.ndarray,
    ) -> None:
        self.config = config
        self.h = h
        self.nx, self.nz, self.ny = compute_mesh_shape(config, h)
        self.nodes, _ = gll_nodes_and_weights(config.ngllx)
        ng = config.ngllx
        self.element_values = np.empty(
            (self.nx, self.nz, self.ny, ng, ng, ng, 3), dtype=float,
        )

        field_map = {
            coord_key_3d(z, x, y): (uz, ux, uy)
            for (z, x, y), (uz, ux, uy) in zip(points, field_values)
        }
        for ex in range(self.nx):
            xl = config.x_min + ex * h
            xc = xl + 0.5 * (self.nodes + 1.0) * h
            for ez in range(self.nz):
                zb = config.z_min + ez * h
                zc = zb + 0.5 * (self.nodes + 1.0) * h
                for ey in range(self.ny):
                    yl = config.y_min + ey * h
                    yc = yl + 0.5 * (self.nodes + 1.0) * h
                    for i, xv in enumerate(xc):
                        for j, zv in enumerate(zc):
                            for k, yv in enumerate(yc):
                                key = coord_key_3d(zv, xv, yv)
                                if key not in field_map:
                                    raise KeyError(f"Missing reference value at {key}")
                                self.element_values[ex, ez, ey, i, j, k, :] = field_map[key]

    def evaluate(self, z: float, x: float, y: float) -> tuple[float, float, float]:
        cfg = self.config
        x_max = cfg.x_min + cfg.length_x
        z_max = cfg.z_min + cfg.length_z
        y_max = cfg.y_min + cfg.length_y

        def find_element(coord, cmin, cmax, n):
            if math.isclose(coord, cmax, abs_tol=1.0e-9):
                return n - 1
            sc = (coord - cmin) / self.h
            return max(0, min(n - 1, int(math.floor(sc))))

        ex = find_element(x, cfg.x_min, x_max, self.nx)
        ez = find_element(z, cfg.z_min, z_max, self.nz)
        ey = find_element(y, cfg.y_min, y_max, self.ny)

        xi_x = max(-1.0, min(1.0, 2.0 * (x - (cfg.x_min + ex * self.h)) / self.h - 1.0))
        xi_z = max(-1.0, min(1.0, 2.0 * (z - (cfg.z_min + ez * self.h)) / self.h - 1.0))
        xi_y = max(-1.0, min(1.0, 2.0 * (y - (cfg.y_min + ey * self.h)) / self.h - 1.0))

        bx = lagrange_basis(xi_x, self.nodes)
        bz = lagrange_basis(xi_z, self.nodes)
        by = lagrange_basis(xi_y, self.nodes)
        vals = np.einsum("i,j,k,ijkc->c", bx, bz, by, self.element_values[ex, ez, ey])
        return float(vals[0]), float(vals[1]), float(vals[2])


def _build_boundary_conditions(config: GlobalSelfconvConfig3D) -> list:
    """Build boundary conditions for the 3D case.

    For extruded meshes (spectral/quad), the y-direction faces use
    "bottom"/"top" labels.  For non-extruded meshes (KMV/tet via
    fire.BoxMesh), all faces have numeric IDs:
        1: z=0 (free), 2: z=-Lz, 3: x=0, 4: x=Lx, 5: y=0, 6: y=Ly.
    """
    zero = Constant((0.0, 0.0, 0.0))
    is_tet = config.cell_type in ("T", "triangle", "tetrahedra")
    bcs = [
        ("u", 2, zero),  # z = -Lz
        ("u", 3, zero),  # x = 0
        ("u", 4, zero),  # x = Lx
    ]
    if is_tet:
        bcs.append(("u", 5, zero))  # y = 0
        bcs.append(("u", 6, zero))  # y = Ly
    else:
        bcs.append(("u", "bottom", zero))  # y = 0
        bcs.append(("u", "top", zero))     # y = Ly
    # z = 0 (face 1) is left free.
    return bcs


# ------------------------------------------------------------------
# Build Spyro case dictionary
# ------------------------------------------------------------------
def build_case_dict(config: GlobalSelfconvConfig3D) -> dict:
    receiver_locations = spyro.create_transect(
        (config.receiver_z, config.receiver_x_start, config.receiver_y),
        (config.receiver_z, config.receiver_x_end, config.receiver_y),
        config.num_receivers,
    )
    amplitude = np.array(
        [config.source_dir_z, config.source_dir_x, config.source_dir_y], dtype=float,
    )
    step_count = int(round(config.final_time / config.dt))
    return {
        "options": {
            "cell_type": config.cell_type,
            "variant": config.variant,
            "degree": config.degree,
            "dimension": config.dimension,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": config.length_z,
            "length_x": config.length_x,
            "length_y": config.length_y,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(config.source_z, config.source_x, config.source_y)],
            "frequency": config.frequency,
            "delay": config.source_delay,
            "delay_type": "time",
            "receiver_locations": receiver_locations,
            "amplitude": amplitude,
            "use_vertex_only_mesh": config.use_vertex_only_mesh,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": config.final_time,
            "dt": config.dt,
            "output_frequency": step_count + 1,
            "gradient_sampling_frequency": max(1, step_count - 1),
        },
        "synthetic_data": {
            "type": "object",
            "density": config.density,
            "p_wave_velocity": config.p_wave_velocity,
            "s_wave_velocity": config.s_wave_velocity,
            "real_velocity_file": None,
        },
        "boundary_conditions": _build_boundary_conditions(config),
        "visualization": {
            "forward_output": False,
            "fwi_velocity_model_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
            "time": False,
            "mechanical_energy": False,
        },
    }


# ------------------------------------------------------------------
# Run a single case
# ------------------------------------------------------------------
def sample_final_field(wave: spyro.IsotropicWave, points: np.ndarray) -> np.ndarray:
    return np.asarray(wave.u_n.at(points.tolist()), dtype=float)


def run_case(
    config: GlobalSelfconvConfig3D,
    h: float,
    run_dir: Path,
    force: bool,
    sample_field: bool = True,
) -> tuple:
    """Run a single case. Returns (summary_dict, wave_or_None).

    When loaded from cache wave is None.  Set *sample_field=False* to skip
    GLL field sampling (used for KMV where Firedrake handles the error).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.json"
    samples_path = run_dir / "final_field_samples.npz"
    receivers_path = run_dir / "receiver_traces.npz"

    cache_valid = summary_path.exists() and receivers_path.exists() and not force
    if sample_field:
        cache_valid = cache_valid and samples_path.exists()
    if cache_valid:
        return json.loads(summary_path.read_text(encoding="utf-8")), None

    case = build_case_dict(config)
    wave = spyro.IsotropicWave(case)

    total_start = time.perf_counter()
    mesh_start = time.perf_counter()
    wave.set_mesh(input_mesh_parameters={"edge_length": h, "periodic": False})
    mesh_end = time.perf_counter()
    solve_start = time.perf_counter()
    wave.forward_solve()
    solve_end = time.perf_counter()

    # Field sampling — spectral only (collective in MPI)
    sample_start = time.perf_counter()
    if sample_field:
        points, quadrature_weights = build_global_quadrature_grid_3d(config, h)
        field_values = sample_final_field(wave, points)
    sample_end = time.perf_counter()

    # Save receiver traces
    receivers_output = np.asarray(wave.receivers_output, dtype=float)
    time_axis = np.arange(receivers_output.shape[0], dtype=float) * config.dt

    mesh_num_cells = mpi_sum_int(int(wave.mesh.cell_set.size))
    total_end = time.perf_counter()

    if mpi_is_root():
        receiver_locations = np.asarray(case["acquisition"]["receiver_locations"], dtype=float)
        np.savez(
            receivers_path,
            time=time_axis,
            receivers_output=receivers_output,
            receiver_locations=receiver_locations,
        )
        if sample_field:
            np.savez(
                samples_path,
                points=points,
                quadrature_weights=quadrature_weights,
                field_values=field_values,
            )

        nx, nz, ny = compute_mesh_shape(config, h)
        summary = {
            "description": "3D global self-convergence with top free surface and Dirichlet on other faces.",
            "edge_length": float(h),
            "mesh_num_cells": mesh_num_cells,
            "mesh_shape": {"nx": nx, "nz": nz, "ny": ny},
            "function_space_dim": int(wave.function_space.dim()),
            "sample_grid_points": int(points.shape[0]) if sample_field else 0,
            "t_final": float(wave.current_time),
            "coordinate_order": ["z", "x", "y"],
            "component_order": ["uz", "ux", "uy"],
            "boundary_mode": "top-free-dirichlet",
            "source_location": [config.source_z, config.source_x, config.source_y],
            "num_receivers": config.num_receivers,
            "timing_seconds": {
                "mesh_setup_seconds": mesh_end - mesh_start,
                "forward_solve_seconds": solve_end - solve_start,
                "field_sampling_seconds": (sample_end - sample_start) if sample_field else 0.0,
                "total_script_seconds": total_end - total_start,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        summary = None

    mpi_barrier()
    summary = mpi_broadcast(summary)
    return summary, wave


# ------------------------------------------------------------------
# Error metrics
# ------------------------------------------------------------------
def load_field_samples(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = np.load(run_dir / "final_field_samples.npz")
    return (
        np.asarray(bundle["points"], dtype=float),
        np.asarray(bundle["quadrature_weights"], dtype=float),
        np.asarray(bundle["field_values"], dtype=float),
    )


def load_receiver_traces(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    bundle = np.load(run_dir / "receiver_traces.npz")
    return (
        np.asarray(bundle["time"], dtype=float),
        np.asarray(bundle["receivers_output"], dtype=float),
    )


def compute_global_metrics(
    config: GlobalSelfconvConfig3D,
    h: float,
    run_dir: Path,
    reference_h: float,
    reference_field: StructuredReferenceField3D,
) -> dict:
    points, quadrature_weights, field_values = load_field_samples(run_dir)
    reference_values = np.asarray(
        [reference_field.evaluate(z, x, y) for z, x, y in points],
        dtype=float,
    )
    error = field_values - reference_values

    sq_err = np.sum(quadrature_weights * np.sum(error * error, axis=1))
    sq_ref = np.sum(quadrature_weights * np.sum(reference_values * reference_values, axis=1))
    component_metrics = {}
    for idx, name in enumerate(["uz", "ux", "uy"]):
        sq_e = np.sum(quadrature_weights * error[:, idx] ** 2)
        sq_r = np.sum(quadrature_weights * reference_values[:, idx] ** 2)
        component_metrics[f"abs_error_{name}"] = math.sqrt(float(sq_e))
        component_metrics[f"rel_error_{name}"] = math.sqrt(float(sq_e)) / math.sqrt(float(sq_r)) if sq_r > 0 else 0.0

    abs_error = math.sqrt(float(sq_err))
    ref_norm = math.sqrt(float(sq_ref))
    nx, nz, ny = compute_mesh_shape(config, h)
    return {
        "h": float(h),
        "dt": float(config.dt),
        "reference_h": float(reference_h),
        "expected_order": float(config.degree + 1),
        "t_final": float(config.final_time),
        "mesh_num_cells": int(nx * nz * ny),
        "sample_grid_points": int(points.shape[0]),
        "abs_error": abs_error,
        "rel_error": abs_error / ref_norm if ref_norm > 0 else abs_error,
        **component_metrics,
        "reference_norm": ref_norm,
    }


def compute_receiver_metrics(
    reference_dir: Path,
    candidate_dir: Path,
) -> dict:
    ref_time, ref_recv = load_receiver_traces(reference_dir)
    cand_time, cand_recv = load_receiver_traces(candidate_dir)
    common_nt = min(len(ref_time), len(cand_time))
    ref_time = ref_time[:common_nt]
    cand_time = cand_time[:common_nt]
    if not np.allclose(ref_time, cand_time, atol=1.0e-9):
        raise ValueError("Time axes do not match between reference and candidate.")

    ref_recv = ref_recv[:common_nt]
    cand_recv = cand_recv[:common_nt]
    early_window_end = resolve_early_window_end(ref_time)
    early_mask = ref_time <= early_window_end

    metrics: dict = {"early_window_end": float(early_window_end)}
    for idx, name in enumerate(["uz", "ux", "uy"]):
        ref_c = ref_recv[:, :, idx]
        cand_c = cand_recv[:, :, idx]
        metrics[name] = {
            "relative_l2_full": relative_l2(ref_c, cand_c),
            "relative_l2_early": relative_l2(ref_c[early_mask], cand_c[early_mask]),
        }
    ref_all = ref_recv[:, :, :3]
    cand_all = cand_recv[:, :, :3]
    metrics["u"] = {
        "relative_l2_full": relative_l2(ref_all, cand_all),
        "relative_l2_early": relative_l2(ref_all[early_mask], cand_all[early_mask]),
    }
    return metrics


def compute_global_metrics_firedrake(
    wave_ref,
    wave_coarse,
    config: GlobalSelfconvConfig3D,
    h: float,
    reference_h: float,
) -> dict:
    """Compute global L2 error via Firedrake cross-mesh interpolation (KMV).

    Both solutions are projected into a DG space on the coarse mesh so that
    cross-mesh interpolation is always supported (target must be CG/DG/Q/DQ).
    Firedrake assemble is globally collective, so the result is correct in MPI.
    """
    V_dg = firedrake.VectorFunctionSpace(wave_coarse.mesh, "DG", config.degree)
    u_ref_dg = firedrake.Function(V_dg).interpolate(wave_ref.u_n)
    u_h_dg = firedrake.Function(V_dg).interpolate(wave_coarse.u_n)
    diff = firedrake.Function(V_dg).assign(u_ref_dg - u_h_dg)

    sq_err = float(firedrake.assemble(firedrake.inner(diff, diff) * firedrake.dx))
    sq_ref = float(firedrake.assemble(firedrake.inner(u_ref_dg, u_ref_dg) * firedrake.dx))
    abs_error = math.sqrt(sq_err)
    ref_norm = math.sqrt(sq_ref)

    component_metrics: dict = {}
    for idx, name in enumerate(["uz", "ux", "uy"]):
        sq_e = float(firedrake.assemble(diff[idx] ** 2 * firedrake.dx))
        sq_r = float(firedrake.assemble(u_ref_dg[idx] ** 2 * firedrake.dx))
        component_metrics[f"abs_error_{name}"] = math.sqrt(sq_e)
        component_metrics[f"rel_error_{name}"] = (
            math.sqrt(sq_e) / math.sqrt(sq_r) if sq_r > 0 else 0.0
        )

    mesh_num_cells = mpi_sum_int(int(wave_coarse.mesh.cell_set.size))
    return {
        "h": float(h),
        "dt": float(config.dt),
        "reference_h": float(reference_h),
        "expected_order": float(config.degree + 1),
        "t_final": float(config.final_time),
        "mesh_num_cells": mesh_num_cells,
        "sample_grid_points": 0,
        "abs_error": abs_error,
        "rel_error": abs_error / ref_norm if ref_norm > 0 else abs_error,
        **component_metrics,
        "reference_norm": ref_norm,
    }


# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------
def compute_observed_rates(rows: list[dict]) -> list[dict]:
    enriched = []
    for i, row in enumerate(rows):
        r = dict(row)
        if i == 0:
            r["rate"] = None
        else:
            prev = rows[i - 1]
            if row["rel_error"] > 0 and prev["rel_error"] > 0:
                r["rate"] = float(
                    math.log(prev["rel_error"] / row["rel_error"])
                    / math.log(prev["h"] / row["h"])
                )
            else:
                r["rate"] = None
        enriched.append(r)
    return enriched


def fit_order(rows: list[dict]) -> float:
    hs = np.asarray([r["h"] for r in rows], dtype=float)
    es = np.asarray([r["rel_error"] for r in rows], dtype=float)
    return float(np.polyfit(np.log(hs), np.log(es), 1)[0])


def reference_order_curve(rows: list[dict], order: float) -> list[float]:
    h0, e0 = rows[0]["h"], rows[0]["rel_error"]
    return [e0 * (r["h"] / h0) ** order for r in rows]


def save_rows_as_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_convergence_figure(
    output_path: Path,
    rows: list[dict],
    expected_order: float,
    receiver_metrics: list[dict],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax = axes[0]
    hs = [r["h"] for r in rows]
    ax.loglog(hs, [r["rel_error"] for r in rows], marker="o", linewidth=1.8, label=f"global (fit={fit_order(rows):.2f})")
    for name, marker in [("rel_error_uz", "s"), ("rel_error_ux", "D"), ("rel_error_uy", "^")]:
        ax.loglog(hs, [r[name] for r in rows], marker=marker, linewidth=1.4, label=name.replace("rel_error_", ""))
    ax.loglog(hs, reference_order_curve(rows, expected_order), linestyle="--", linewidth=1.2, color="dimgray", label=f"$O(h^{{{int(expected_order)}}})$")

    for i in range(1, len(rows)):
        if rows[i]["rate"] is not None:
            hm = math.sqrt(rows[i - 1]["h"] * rows[i]["h"])
            em = math.sqrt(rows[i - 1]["rel_error"] * rows[i]["rel_error"])
            ax.annotate(f"{rows[i]['rate']:.2f}", xy=(hm, em), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)

    ax.set_title("3D global final-time displacement error")
    ax.set_xlabel("h")
    ax.set_ylabel("Relative L2 error")
    ax.xaxis.set_major_locator(mticker.FixedLocator(hs))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3g}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, which="both", linestyle=":")
    ax.legend(fontsize="small")

    ax = axes[1]
    if receiver_metrics:
        recv_hs = [m["h"] for m in receiver_metrics]
        for key, label, marker in [
            ("u_rel_l2_full", "u (full)", "o"),
            ("uz_rel_l2_full", "uz (full)", "s"),
            ("ux_rel_l2_full", "ux (full)", "D"),
            ("uy_rel_l2_full", "uy (full)", "^"),
            ("u_rel_l2_early", "u (early)", "v"),
        ]:
            vals = [m[key] for m in receiver_metrics]
            if all(v > 0 for v in vals):
                fit_val = np.polyfit(np.log(np.asarray(recv_hs)), np.log(np.asarray(vals)), 1)
                ax.loglog(recv_hs, vals, marker=marker, linewidth=1.4, label=f"{label} (fit={fit_val[0]:.2f})")
            else:
                ax.loglog(recv_hs, vals, marker=marker, linewidth=1.4, label=label)
        h0 = recv_hs[0]
        e0 = receiver_metrics[0]["u_rel_l2_full"]
        guide = [e0 * (h / h0) ** expected_order for h in recv_hs]
        ax.loglog(recv_hs, guide, linestyle="--", linewidth=1.2, color="dimgray", label=f"$O(h^{{{int(expected_order)}}})$")
        ax.set_title("Receiver-trace error vs reference")
        ax.set_xlabel("h")
        ax.set_ylabel("Relative L2 error")
        ax.xaxis.set_major_locator(mticker.FixedLocator(recv_hs))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3g}"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(True, which="both", linestyle=":")
        ax.legend(fontsize="small")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(
    path: Path,
    rows: list[dict],
    config: GlobalSelfconvConfig3D,
    expected_order: float,
    observed_order: float,
    reference_h: float,
    receiver_metrics: list[dict],
) -> None:
    lines = [
        "Spyro 3D global self-convergence study (top free surface + Dirichlet)",
        "",
        f"Reference mesh size h_ref = {reference_h:.5f}",
        f"Expected spatial order for degree {config.degree} = O(h^{int(expected_order)})",
        f"Observed least-squares fit (global) = {observed_order:.4f}",
        f"dt = {config.dt:.6f}",
        f"final_time = {config.final_time:.3f}",
        f"domain = {config.length_z} x {config.length_x} x {config.length_y}",
        f"boundary_mode = top-free-dirichlet",
        "",
        "Per-run global results:",
    ]
    for row in rows:
        rate_text = "---" if row["rate"] is None else f"{row['rate']:.4f}"
        lines.append(
            f"  h={row['h']:.5f}  cells={row['mesh_num_cells']}  "
            f"rel_err={row['rel_error']:.6e}  rate={rate_text}"
        )
    if receiver_metrics:
        lines.extend(["", "Per-run receiver metrics:"])
        for m in receiver_metrics:
            lines.append(
                f"  h={m['h']:.5f}  u_full={m['u_rel_l2_full']:.6e}  "
                f"u_early={m['u_rel_l2_early']:.6e}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    method = args.method
    is_kmv = method == "kmv"

    if is_kmv and args.output_dir == DEFAULT_OUTPUT_DIR:
        output_dir = (ROOT_DIR / "results" / "global_selfconv_top_free_3d_kmv").resolve()
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_kmv:
        dt_val = args.dt if args.dt != DEFAULT_DT else DEFAULT_DT_KMV
        config = GlobalSelfconvConfig3D(
            final_time=args.final_time,
            dt=dt_val,
            degree=args.degree,
            cell_type="T",
            variant="lumped",
        )
    else:
        config = GlobalSelfconvConfig3D(
            final_time=args.final_time,
            dt=args.dt,
            degree=args.degree,
        )

    # Use KMV-specific defaults if user didn't override h values
    space_hs = list(args.space_hs)
    ref_h = args.reference_h
    if is_kmv:
        if space_hs == DEFAULT_SPACE_HS:
            space_hs = list(DEFAULT_SPACE_HS_KMV)
        if ref_h == DEFAULT_REFERENCE_H:
            ref_h = DEFAULT_REFERENCE_H_KMV

    h_values = canonical_h_values(space_hs, ref_h)
    reference_h = float(ref_h)
    expected_order = float(config.degree + 1)

    if mpi_is_root():
        method_label = "KMV" if is_kmv else "Spectral"
        print(f"Running 3D global self-convergence study ({method_label})", flush=True)
        print(f"  h_sweep = {h_values}", flush=True)
        print(f"  h_ref   = {reference_h}", flush=True)
        print(f"  BCs     = top-free-dirichlet", flush=True)
        print(f"  dt={config.dt}, T={config.final_time}, p={config.degree}", flush=True)
        comm = mpi_world()
        if comm is not None:
            print(f"  MPI ranks = {comm.size}", flush=True)
        print()

    # 1. Reference case
    ref_run_dir = output_dir / "runs" / f"h_{edge_length_tag(reference_h)}"
    if mpi_is_root():
        print(f"[ref] Running reference h={reference_h:.5f} ...", flush=True)
    ref_summary, wave_ref = run_case(
        config, reference_h, ref_run_dir, args.force, sample_field=not is_kmv,
    )
    if mpi_is_root():
        print(
            f"[ref] Done: cells={ref_summary['mesh_num_cells']}, "
            f"time={ref_summary['timing_seconds']['total_script_seconds']:.1f}s",
            flush=True,
        )

    # Build reference field (root only, for spectral metrics)
    reference_field = None
    if not is_kmv and mpi_is_root():
        ref_points, _, ref_values = load_field_samples(ref_run_dir)
        reference_field = StructuredReferenceField3D(config, reference_h, ref_points, ref_values)

    # 2. Sweep cases
    rows = []
    receiver_metrics_list = []
    for idx, h in enumerate(h_values, start=1):
        run_dir = output_dir / "runs" / f"h_{edge_length_tag(h)}"
        if mpi_is_root():
            print(f"[{idx}/{len(h_values)}] Running h={h:.5f} ...", end=" ", flush=True)
        case_summary, wave_coarse = run_case(
            config, h, run_dir, args.force, sample_field=not is_kmv,
        )

        if is_kmv:
            # Firedrake error computation (all ranks participate)
            if wave_ref is None or wave_coarse is None:
                if wave_ref is None:
                    _, wave_ref = run_case(
                        config, reference_h, ref_run_dir,
                        force=True, sample_field=False,
                    )
                if wave_coarse is None:
                    case_summary, wave_coarse = run_case(
                        config, h, run_dir, force=True, sample_field=False,
                    )
            row = compute_global_metrics_firedrake(
                wave_ref, wave_coarse, config, h, reference_h,
            )
            del wave_coarse

            if mpi_is_root():
                (run_dir / "global_metrics.json").write_text(
                    json.dumps(row, indent=2), encoding="utf-8",
                )
                rows.append(row)
                print(
                    f"cells={case_summary['mesh_num_cells']}  rel_err={row['rel_error']:.4e}  "
                    f"time={case_summary['timing_seconds']['total_script_seconds']:.1f}s",
                    flush=True,
                )
        else:
            if mpi_is_root():
                row = compute_global_metrics(config, h, run_dir, reference_h, reference_field)
                (run_dir / "global_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                rows.append(row)
                print(
                    f"cells={case_summary['mesh_num_cells']}  rel_err={row['rel_error']:.4e}  "
                    f"time={case_summary['timing_seconds']['total_script_seconds']:.1f}s",
                    flush=True,
                )

        if mpi_is_root():
            try:
                recv_metrics = compute_receiver_metrics(ref_run_dir, run_dir)
                receiver_metrics_list.append({
                    "h": float(h),
                    "u_rel_l2_full": recv_metrics["u"]["relative_l2_full"],
                    "u_rel_l2_early": recv_metrics["u"]["relative_l2_early"],
                    "uz_rel_l2_full": recv_metrics["uz"]["relative_l2_full"],
                    "uz_rel_l2_early": recv_metrics["uz"]["relative_l2_early"],
                    "ux_rel_l2_full": recv_metrics["ux"]["relative_l2_full"],
                    "ux_rel_l2_early": recv_metrics["ux"]["relative_l2_early"],
                    "uy_rel_l2_full": recv_metrics["uy"]["relative_l2_full"],
                    "uy_rel_l2_early": recv_metrics["uy"]["relative_l2_early"],
                })
            except Exception as exc:
                print(f"  Warning: receiver metrics: {exc}", flush=True)

    if is_kmv and wave_ref is not None:
        del wave_ref

    if not mpi_is_root():
        return

    rows = compute_observed_rates(rows)
    observed_order = fit_order(rows)

    # 3. Save outputs
    tag = f"global_selfconv_top_free_3d{'_kmv' if is_kmv else ''}"
    figure_path = output_dir / f"{tag}.png"
    csv_path = output_dir / f"{tag}.csv"
    json_path = output_dir / f"{tag}_summary.json"
    txt_path = output_dir / f"{tag}_summary.txt"

    save_convergence_figure(figure_path, rows, expected_order, receiver_metrics_list)
    save_rows_as_csv(csv_path, rows)

    summary = {
        "config": asdict(config),
        "method": method,
        "reference_h": reference_h,
        "reference_run": ref_summary,
        "expected_spatial_order": expected_order,
        "observed_spatial_order_fit": observed_order,
        "boundary_mode": "top-free-dirichlet",
        "space_convergence": rows,
        "receiver_metrics": receiver_metrics_list,
        "files": {
            "convergence_figure": str(figure_path),
            "space_csv": str(csv_path),
            "summary_json": str(json_path),
            "summary_txt": str(txt_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_text(txt_path, rows, config, expected_order, observed_order, reference_h, receiver_metrics_list)

    print(f"\nObserved global order = {observed_order:.4f} (expected {expected_order:.1f})")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
