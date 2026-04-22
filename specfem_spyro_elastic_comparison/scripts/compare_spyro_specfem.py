#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SPYRO_RESULTS_DIR = ROOT_DIR / "results" / "spyro"
DEFAULT_SPECFEM_RESULTS_DIR = ROOT_DIR / "results" / "specfem"
DEFAULT_COMPARISON_DIR = ROOT_DIR / "results" / "comparison"
EARLY_WINDOW_END_CAP = 1.1
EARLY_WINDOW_FRACTION = 0.75


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.ravel()))
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm((candidate - reference).ravel()) / denominator)


def equation1_relative_l2(spyro: np.ndarray, specfem: np.ndarray) -> float:
    """Relative L2 error as defined in Equation 1 of the report."""
    return relative_l2(specfem, spyro)


def best_fit_scale(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(np.dot(candidate.ravel(), candidate.ravel()))
    if denominator == 0.0:
        return 1.0
    return float(np.dot(reference.ravel(), candidate.ravel()) / denominator)


def mean_trace_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    correlations: list[float] = []
    for trace_index in range(reference.shape[1]):
        ref_trace = reference[:, trace_index]
        cand_trace = candidate[:, trace_index]
        ref_std = float(np.std(ref_trace))
        cand_std = float(np.std(cand_trace))
        if ref_std == 0.0 or cand_std == 0.0:
            continue
        correlations.append(float(np.corrcoef(ref_trace, cand_trace)[0, 1]))
    if not correlations:
        return 0.0
    return float(np.mean(correlations))


def load_specfem_component(
    results_dir: Path,
    components: tuple[str, ...],
) -> tuple[str, np.ndarray, np.ndarray]:
    files = []
    selected_component = None
    for component in components:
        files = sorted((results_dir / "raw_seismograms").glob(f"AA.S*.{component}.semd"))
        if files:
            selected_component = component
            break
    if not files or selected_component is None:
        tried = ", ".join(components)
        raise FileNotFoundError(f"No SPECFEM seismograms found for components: {tried}")

    traces = []
    time_axis = None
    for file_path in files:
        data = np.loadtxt(file_path)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"Unexpected seismogram format in {file_path}")
        if time_axis is None:
            time_axis = data[:, 0]
        traces.append(data[:, 1])
    return selected_component, np.asarray(time_axis, dtype=float), np.column_stack(traces)


def parse_specfem_mesh_counts(log_text: str) -> dict[str, int | None]:
    total_match = re.search(
        r"Total number of spectral elements in the mesh =\s*(\d+)",
        log_text,
    )
    regular_match = re.search(r"of which\s*(\d+)\s*are regular elements", log_text)
    pml_match = re.search(r"and\s*(\d+)\s*are PML elements", log_text)
    return {
        "total": int(total_match.group(1)) if total_match else None,
        "regular": int(regular_match.group(1)) if regular_match else None,
        "pml": int(pml_match.group(1)) if pml_match else None,
    }


def normalize_mesh_counts(mesh_counts: dict[str, int | None]) -> dict[str, int]:
    total = int(mesh_counts["total"] or 0)
    pml = int(mesh_counts["pml"] or 0)
    regular = mesh_counts["regular"]
    if regular is None:
        regular = total - pml
    return {
        "total": total,
        "regular": int(regular),
        "pml": pml,
    }


def resolve_early_window_end(time_axis: np.ndarray) -> float:
    if len(time_axis) == 0:
        return EARLY_WINDOW_END_CAP
    return min(EARLY_WINDOW_END_CAP, EARLY_WINDOW_FRACTION * float(time_axis[-1]))


def save_component_panel(
    component_name: str,
    time_axis: np.ndarray,
    spyro_data: np.ndarray,
    specfem_data: np.ndarray,
    output_path: Path,
) -> None:
    difference = specfem_data - spyro_data
    value_limit = np.max(np.abs(spyro_data))
    if value_limit == 0.0:
        value_limit = 1.0
    normalized_difference = difference / value_limit
    difference_limit = float(np.max(np.abs(normalized_difference)))

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    panels = [
        (r"$\mathrm{Spyro}$", spyro_data, value_limit, "gray", None, "Displacement"),
        (r"$\mathrm{SPECFEM2D}$", specfem_data, value_limit, "gray", None, "Displacement"),
        (
            r"$\varepsilon_{L2}$",
            normalized_difference,
            difference_limit,
            "RdBu_r",
            ticker.PercentFormatter(xmax=1.0, decimals=0),
            r"Relative difference [$\%$]",
        ),
    ]

    for axis, (title, panel_data, panel_limit, cmap, colorbar_format, colorbar_label) in zip(axes, panels):
        image = axis.imshow(
            panel_data,
            aspect="auto",
            cmap=cmap,
            vmin=-panel_limit,
            vmax=panel_limit,
            extent=(0, panel_data.shape[1] - 1, time_axis[-1], time_axis[0]),
            interpolation="nearest",
        )
        axis.set_title(f"{component_name} - {title}")
        axis.set_ylabel(r"$t$ [s]")
        colorbar = fig.colorbar(image, ax=axis, pad=0.01, format=colorbar_format)
        colorbar.set_label(colorbar_label)

    axes[-1].set_xlabel(r"Receiver index $i_r$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_trace_panel(
    time_axis: np.ndarray,
    spyro_uz: np.ndarray,
    spyro_ux: np.ndarray,
    specfem_uz: np.ndarray,
    specfem_ux: np.ndarray,
    output_path: Path,
) -> None:
    receiver_indices = [0, spyro_uz.shape[1] // 2, spyro_uz.shape[1] - 1]
    fig, axes = plt.subplots(len(receiver_indices), 2, figsize=(12, 8), sharex=True)

    for row_index, receiver_index in enumerate(receiver_indices):
        axes[row_index, 0].plot(
            time_axis,
            spyro_uz[:, receiver_index],
            label=r"$\mathrm{Spyro}$",
            linewidth=1.0,
        )
        axes[row_index, 0].plot(
            time_axis,
            specfem_uz[:, receiver_index],
            label=r"$\mathrm{SPECFEM2D}$",
            linewidth=1.0,
            linestyle="--",
        )
        axes[row_index, 0].set_title(rf"Receiver $i_r={receiver_index}$ - $u_z$")
        axes[row_index, 0].grid(True, alpha=0.3)

        axes[row_index, 1].plot(
            time_axis,
            spyro_ux[:, receiver_index],
            label=r"$\mathrm{Spyro}$",
            linewidth=1.0,
        )
        axes[row_index, 1].plot(
            time_axis,
            specfem_ux[:, receiver_index],
            label=r"$\mathrm{SPECFEM2D}$",
            linewidth=1.0,
            linestyle="--",
        )
        axes[row_index, 1].set_title(rf"Receiver $i_r={receiver_index}$ - $u_x$")
        axes[row_index, 1].grid(True, alpha=0.3)

    axes[0, 0].legend(loc="upper right")
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$t$ [s]")
    for axis in axes[:, 0]:
        axis.set_ylabel("Displacement")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Spyro and SPECFEM2D elastic forward results.",
    )
    parser.add_argument(
        "--spyro-dir",
        type=Path,
        default=DEFAULT_SPYRO_RESULTS_DIR,
        help="Directory containing Spyro 2D outputs.",
    )
    parser.add_argument(
        "--specfem-dir",
        type=Path,
        default=DEFAULT_SPECFEM_RESULTS_DIR,
        help="Directory containing SPECFEM2D outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help="Directory where comparison outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SPYRO_RESULTS_DIR = args.spyro_dir.resolve()
    SPECFEM_RESULTS_DIR = args.specfem_dir.resolve()
    COMPARISON_DIR = args.output_dir.resolve()
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    spyro_bundle = np.load(SPYRO_RESULTS_DIR / "spyro_receivers.npz")
    spyro_time = np.asarray(spyro_bundle["time"], dtype=float)
    spyro_data = np.asarray(spyro_bundle["receivers_output"], dtype=float)
    spyro_uz = spyro_data[:, :, 0]
    spyro_ux = spyro_data[:, :, 1]
    spyro_mesh_elements = int(np.asarray(spyro_bundle["mesh_num_cells"]).item())

    ux_channel, specfem_time_x, specfem_ux = load_specfem_component(
        SPECFEM_RESULTS_DIR,
        ("BXX", "FXX"),
    )
    uz_channel, specfem_time_z, specfem_uz = load_specfem_component(
        SPECFEM_RESULTS_DIR,
        ("BXZ", "FXZ"),
    )
    specfem_time_shift = float(-specfem_time_x[0])
    specfem_time_x = specfem_time_x + specfem_time_shift
    specfem_time_z = specfem_time_z + specfem_time_shift

    common_nt = min(len(spyro_time), len(specfem_time_x), len(specfem_time_z))
    spyro_time = spyro_time[:common_nt]
    specfem_time = specfem_time_x[:common_nt]
    spyro_uz = spyro_uz[:common_nt, :]
    spyro_ux = spyro_ux[:common_nt, :]
    specfem_uz = specfem_uz[:common_nt, :]
    specfem_ux = specfem_ux[:common_nt, :]

    if not np.allclose(spyro_time, specfem_time, atol=1.0e-9):
        raise ValueError("Spyro and SPECFEM time axes do not match")

    early_window_end = resolve_early_window_end(spyro_time)
    early_mask = spyro_time <= early_window_end

    ux_scale = best_fit_scale(spyro_ux, specfem_ux)
    uz_scale = best_fit_scale(spyro_uz, specfem_uz)

    metrics = {
        "ux": {
            "relative_l2_full": equation1_relative_l2(spyro_ux, specfem_ux),
            "relative_l2_early": equation1_relative_l2(spyro_ux[early_mask], specfem_ux[early_mask]),
            "scaled_relative_l2_full": equation1_relative_l2(spyro_ux, ux_scale * specfem_ux),
            "scaled_relative_l2_early": equation1_relative_l2(spyro_ux[early_mask], ux_scale * specfem_ux[early_mask]),
            "mean_trace_correlation_full": mean_trace_correlation(spyro_ux, specfem_ux),
            "mean_trace_correlation_early": mean_trace_correlation(spyro_ux[early_mask], specfem_ux[early_mask]),
            "best_fit_scale": ux_scale,
        },
        "uz": {
            "relative_l2_full": equation1_relative_l2(spyro_uz, specfem_uz),
            "relative_l2_early": equation1_relative_l2(spyro_uz[early_mask], specfem_uz[early_mask]),
            "scaled_relative_l2_full": equation1_relative_l2(spyro_uz, uz_scale * specfem_uz),
            "scaled_relative_l2_early": equation1_relative_l2(spyro_uz[early_mask], uz_scale * specfem_uz[early_mask]),
            "mean_trace_correlation_full": mean_trace_correlation(spyro_uz, specfem_uz),
            "mean_trace_correlation_early": mean_trace_correlation(spyro_uz[early_mask], specfem_uz[early_mask]),
            "best_fit_scale": uz_scale,
        },
    }

    specfem_log_text = (SPECFEM_RESULTS_DIR / "specfem_solver.log").read_text(encoding="utf-8")
    specfem_mesh_counts = normalize_mesh_counts(parse_specfem_mesh_counts(specfem_log_text))

    save_component_panel(
        rf"$u_x$ / $\mathrm{{{ux_channel}}}$",
        spyro_time,
        spyro_ux,
        specfem_ux,
        COMPARISON_DIR / "ux_bxx_comparison.png",
    )
    save_component_panel(
        rf"$u_z$ / $\mathrm{{{uz_channel}}}$",
        spyro_time,
        spyro_uz,
        specfem_uz,
        COMPARISON_DIR / "uz_bxz_comparison.png",
    )
    save_trace_panel(
        spyro_time,
        spyro_uz,
        spyro_ux,
        specfem_uz,
        specfem_ux,
        COMPARISON_DIR / "trace_comparison.png",
    )

    summary = {
        "description": "Comparison between fully reflective Spyro and SPECFEM2D elastic forward cases",
        "time_axis": {
            "dt": float(spyro_time[1] - spyro_time[0]),
            "num_samples": int(common_nt),
            "final_time": float(spyro_time[-1]),
            "early_window_end": early_window_end,
            "specfem_time_shift_applied": specfem_time_shift,
        },
        "spyro": {
            "spectral_degree": 4,
            "component_order": ["uz", "ux"],
            "coordinate_order": ["z", "x"],
            "periodicity": "none",
            "mesh_elements": spyro_mesh_elements,
        },
        "specfem2d": {
            "ngllx": 5,
            "spectral_degree": 4,
            "component_order": [f"{ux_channel} -> ux", f"{uz_channel} -> uz"],
            "periodicity": "none",
            "vertical_boundaries": "non-absorbing top and bottom",
            "absorbing_boundaries": False,
            "mesh_counts": specfem_mesh_counts,
        },
        "metrics": metrics,
        "limitations": [
            "Both cases are run without absorbing boundary conditions and without periodic boundaries, so all boundaries are reflective.",
            "This is no longer the original notebook boundary setup; it is a boundary-modified comparison case built from it.",
            "Residual differences now mostly reflect solver and implementation differences rather than intentionally mismatched boundary conditions.",
        ],
    }

    (COMPARISON_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    text_lines = [
        "Spyro vs SPECFEM2D elastic comparison",
        "",
        f"Spyro elements: {spyro_mesh_elements}",
        f"SPECFEM2D elements: {specfem_mesh_counts['total']}",
        f"SPECFEM2D regular elements: {specfem_mesh_counts['regular']}",
        f"SPECFEM2D PML elements: {specfem_mesh_counts['pml']}",
        "",
        "Fourth-order SEM correspondence:",
        "- Spyro uses degree = 4 on quadrilateral spectral elements.",
        "- SPECFEM2D uses NGLLX = 5, i.e. polynomial degree 4.",
        "",
        "Boundary-condition caveat:",
        "- Both runs were done without absorbing boundaries.",
        "- Both runs were also done without periodic boundaries, so all boundaries are reflective.",
        "- This is a modified comparison case, not the notebook's original doubly-periodic setup.",
        "",
        f"ux / {ux_channel} metrics:",
        f"- Relative L2 (full): {metrics['ux']['relative_l2_full']:.6f}",
        f"- Relative L2 (early): {metrics['ux']['relative_l2_early']:.6f}",
        f"- Scaled relative L2 (full): {metrics['ux']['scaled_relative_l2_full']:.6f}",
        f"- Mean trace correlation (full): {metrics['ux']['mean_trace_correlation_full']:.6f}",
        "",
        f"uz / {uz_channel} metrics:",
        f"- Relative L2 (full): {metrics['uz']['relative_l2_full']:.6f}",
        f"- Relative L2 (early): {metrics['uz']['relative_l2_early']:.6f}",
        f"- Scaled relative L2 (full): {metrics['uz']['scaled_relative_l2_full']:.6f}",
        f"- Mean trace correlation (full): {metrics['uz']['mean_trace_correlation_full']:.6f}",
        "",
        "Generated files:",
        f"- {COMPARISON_DIR / 'ux_bxx_comparison.png'}",
        f"- {COMPARISON_DIR / 'uz_bxz_comparison.png'}",
        f"- {COMPARISON_DIR / 'trace_comparison.png'}",
        f"- {COMPARISON_DIR / 'comparison_summary.json'}",
    ]
    (COMPARISON_DIR / "comparison_summary.txt").write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Comparison results saved to {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
