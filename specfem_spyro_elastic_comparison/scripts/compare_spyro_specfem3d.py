#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from compare_spyro_specfem import (
    best_fit_scale,
    equation1_relative_l2,
    load_specfem_component,
    mean_trace_correlation,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SPYRO_RESULTS_DIR = ROOT_DIR / "results" / "spyro_3d"
DEFAULT_SPECFEM_RESULTS_DIR = ROOT_DIR / "results" / "specfem3d"
DEFAULT_COMPARISON_DIR = ROOT_DIR / "results" / "comparison_3d"
EARLY_WINDOW_END_CAP = 1.1
EARLY_WINDOW_FRACTION = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Spyro and SPECFEM3D elastic forward results.",
    )
    parser.add_argument(
        "--spyro-dir",
        type=Path,
        default=DEFAULT_SPYRO_RESULTS_DIR,
        help="Directory containing Spyro 3D outputs.",
    )
    parser.add_argument(
        "--specfem-dir",
        type=Path,
        default=DEFAULT_SPECFEM_RESULTS_DIR,
        help="Directory containing SPECFEM3D outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help="Directory where comparison outputs are written.",
    )
    return parser.parse_args()


def save_component_panel(
    component_name: str,
    time_axis: np.ndarray,
    spyro_data: np.ndarray,
    specfem_data: np.ndarray,
    output_path: Path,
) -> None:
    difference = specfem_data - spyro_data
    value_limit = max(float(np.max(np.abs(spyro_data))), float(np.max(np.abs(specfem_data))))
    if value_limit == 0.0:
        value_limit = 1.0
    normalized_difference = difference / value_limit
    difference_limit = float(np.max(np.abs(normalized_difference)))
    if difference_limit == 0.0:
        difference_limit = 1.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    panels = [
        ("Spyro", spyro_data, value_limit, "gray", None, "Displacement"),
        ("SPECFEM3D", specfem_data, value_limit, "gray", None, "Displacement"),
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
        axis.set_ylabel("t [s]")
        colorbar = fig.colorbar(image, ax=axis, pad=0.01, format=colorbar_format)
        colorbar.set_label(colorbar_label)

    axes[-1].set_xlabel("Receiver index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_trace_panel(
    time_axis: np.ndarray,
    spyro_components: dict[str, np.ndarray],
    specfem_components: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    component_order = ["uz", "ux", "uy"]
    receiver_indices = [0, spyro_components["uz"].shape[1] // 2, spyro_components["uz"].shape[1] - 1]
    fig, axes = plt.subplots(len(receiver_indices), len(component_order), figsize=(15, 8.5), sharex=True)

    for row_index, receiver_index in enumerate(receiver_indices):
        for column_index, component in enumerate(component_order):
            axis = axes[row_index, column_index]
            axis.plot(
                time_axis,
                spyro_components[component][:, receiver_index],
                label="Spyro",
                linewidth=1.0,
            )
            axis.plot(
                time_axis,
                specfem_components[component][:, receiver_index],
                label="SPECFEM3D",
                linewidth=1.0,
                linestyle="--",
            )
            axis.set_title(f"Receiver {receiver_index} - {component}")
            axis.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="upper right")
    for axis in axes[-1, :]:
        axis.set_xlabel("t [s]")
    for axis in axes[:, 0]:
        axis.set_ylabel("Displacement")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def resolve_early_window_end(time_axis: np.ndarray) -> float:
    if len(time_axis) == 0:
        return EARLY_WINDOW_END_CAP
    return min(EARLY_WINDOW_END_CAP, EARLY_WINDOW_FRACTION * float(time_axis[-1]))


def main() -> None:
    args = parse_args()
    spyro_dir = args.spyro_dir.resolve()
    specfem_dir = args.specfem_dir.resolve()
    comparison_dir = args.output_dir.resolve()
    comparison_dir.mkdir(parents=True, exist_ok=True)

    spyro_bundle = np.load(spyro_dir / "spyro_receivers.npz")
    spyro_metadata = json.loads((spyro_dir / "spyro_metadata.json").read_text(encoding="utf-8"))
    specfem_metadata = json.loads((specfem_dir / "specfem3d_case_metadata.json").read_text(encoding="utf-8"))

    spyro_time = np.asarray(spyro_bundle["time"], dtype=float)
    spyro_data = np.asarray(spyro_bundle["receivers_output"], dtype=float)
    spyro_uz = spyro_data[:, :, 0]
    spyro_ux = spyro_data[:, :, 1]
    spyro_uy = spyro_data[:, :, 2]

    ux_channel, specfem_time_x, specfem_ux = load_specfem_component(specfem_dir, ("BXX", "FXX"))
    uy_channel, specfem_time_y, specfem_uy = load_specfem_component(specfem_dir, ("BXY", "FXY"))
    uz_channel, specfem_time_z, specfem_uz = load_specfem_component(specfem_dir, ("BXZ", "FXZ"))

    # SPECFEM3D internal Ricker peaks at native t≈0 regardless of time_shift.
    # Align by shifting so that SPECFEM native t=0 maps to Spyro t=source_delay.
    spyro_source_delay = float(spyro_metadata.get("source_delay") or 0.0)
    specfem_time_shift = spyro_source_delay
    specfem_time_x = specfem_time_x + specfem_time_shift
    specfem_time_y = specfem_time_y + specfem_time_shift
    specfem_time_z = specfem_time_z + specfem_time_shift

    # Interpolate SPECFEM onto Spyro time axis if dt differs
    if not np.allclose(spyro_time, specfem_time_x[:len(spyro_time)], atol=1.0e-6):
        nrecv = specfem_ux.shape[1]
        t_end = min(spyro_time[-1], specfem_time_x[-1])
        mask = spyro_time <= t_end + 1e-12
        common_time = spyro_time[mask]
        new_ux = np.column_stack([np.interp(common_time, specfem_time_x, specfem_ux[:, j]) for j in range(nrecv)])
        new_uy = np.column_stack([np.interp(common_time, specfem_time_y, specfem_uy[:, j]) for j in range(nrecv)])
        new_uz = np.column_stack([np.interp(common_time, specfem_time_z, specfem_uz[:, j]) for j in range(nrecv)])
        specfem_ux, specfem_uy, specfem_uz = new_ux, new_uy, new_uz
        spyro_ux = spyro_ux[mask, :]
        spyro_uy = spyro_uy[mask, :]
        spyro_uz = spyro_uz[mask, :]
        spyro_time = common_time
        specfem_time = common_time
    else:
        common_nt = min(len(spyro_time), len(specfem_time_x), len(specfem_time_y), len(specfem_time_z))
        spyro_time = spyro_time[:common_nt]
        specfem_time = specfem_time_x[:common_nt]
        spyro_uz = spyro_uz[:common_nt, :]
        spyro_ux = spyro_ux[:common_nt, :]
        spyro_uy = spyro_uy[:common_nt, :]
        specfem_uz = specfem_uz[:common_nt, :]
        specfem_ux = specfem_ux[:common_nt, :]
        specfem_uy = specfem_uy[:common_nt, :]

    early_window_end = resolve_early_window_end(spyro_time)
    early_mask = spyro_time <= early_window_end

    ux_scale = best_fit_scale(spyro_ux, specfem_ux)
    uy_scale = best_fit_scale(spyro_uy, specfem_uy)
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
        "uy": {
            "relative_l2_full": equation1_relative_l2(spyro_uy, specfem_uy),
            "relative_l2_early": equation1_relative_l2(spyro_uy[early_mask], specfem_uy[early_mask]),
            "scaled_relative_l2_full": equation1_relative_l2(spyro_uy, uy_scale * specfem_uy),
            "scaled_relative_l2_early": equation1_relative_l2(spyro_uy[early_mask], uy_scale * specfem_uy[early_mask]),
            "mean_trace_correlation_full": mean_trace_correlation(spyro_uy, specfem_uy),
            "mean_trace_correlation_early": mean_trace_correlation(spyro_uy[early_mask], specfem_uy[early_mask]),
            "best_fit_scale": uy_scale,
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

    save_component_panel(
        f"ux / {ux_channel}",
        spyro_time,
        spyro_ux,
        specfem_ux,
        comparison_dir / "ux_bxx_comparison.png",
    )
    save_component_panel(
        f"uy / {uy_channel}",
        spyro_time,
        spyro_uy,
        specfem_uy,
        comparison_dir / "uy_bxy_comparison.png",
    )
    save_component_panel(
        f"uz / {uz_channel}",
        spyro_time,
        spyro_uz,
        specfem_uz,
        comparison_dir / "uz_bxz_comparison.png",
    )
    save_trace_panel(
        spyro_time,
        {"uz": spyro_uz, "ux": spyro_ux, "uy": spyro_uy},
        {"uz": specfem_uz, "ux": specfem_ux, "uy": specfem_uy},
        comparison_dir / "trace_comparison.png",
    )

    summary = {
        "description": "Comparison between 3D Spyro and SPECFEM3D elastic forward cases",
        "time_axis": {
            "dt": float(spyro_time[1] - spyro_time[0]),
            "num_samples": int(len(spyro_time)),
            "final_time": float(spyro_time[-1]),
            "early_window_end": early_window_end,
            "specfem_time_shift_applied": specfem_time_shift,
        },
        "spyro": {
            "spectral_degree": int(spyro_metadata["spectral_degree"]),
            "component_order": spyro_metadata["component_order"],
            "coordinate_order": spyro_metadata["coordinate_order"],
            "periodicity": spyro_metadata["periodicity"],
            "mesh_elements": int(spyro_metadata["mesh_num_cells"]),
            "edge_length": float(spyro_metadata["edge_length"]),
        },
        "specfem3d": {
            "ngllx": int(specfem_metadata["ngllx"]),
            "spectral_degree": int(specfem_metadata["spectral_degree"]),
            "component_order": [
                f"{ux_channel} -> ux",
                f"{uy_channel} -> uy",
                f"{uz_channel} -> uz",
            ],
            "mesh": specfem_metadata["mesh"],
            "boundary_conditions": specfem_metadata["boundary_conditions"],
        },
        "metrics": metrics,
        "limitations": [
            "The 3D setup is an extrusion of the 2D notebook geometry using a single receiver line at constant y.",
            "Both runs are configured without absorbing boundaries, so all six boundaries are reflective/free.",
            "Best-fit scaling is reported because source-amplitude normalization is not guaranteed to be identical between Spyro and SPECFEM3D.",
        ],
    }

    (comparison_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    text_lines = [
        "Spyro vs SPECFEM3D elastic comparison",
        "",
        f"Spyro elements: {spyro_metadata['mesh_num_cells']}",
        f"SPECFEM3D expected elements: {specfem_metadata['mesh']['expected_total_elements']}",
        f"SPECFEM3D mesh shape: {specfem_metadata['mesh']['nx']} x {specfem_metadata['mesh']['ny']} x {specfem_metadata['mesh']['nz']}",
        "",
        "Fourth-order SEM correspondence:",
        "- Spyro uses degree = 4 on hexahedral spectral elements.",
        "- SPECFEM3D uses NGLLX = 5, i.e. polynomial degree 4 on HEX8 elements.",
        "",
        "Boundary-condition caveat:",
        "- Both runs were done without absorbing boundaries.",
        "- This is a 3D extrusion of the 2D elastic notebook geometry, not the notebook itself.",
        "",
        f"ux / {ux_channel} metrics:",
        f"- Relative L2 (full): {metrics['ux']['relative_l2_full']:.6f}",
        f"- Relative L2 (early): {metrics['ux']['relative_l2_early']:.6f}",
        f"- Scaled relative L2 (full): {metrics['ux']['scaled_relative_l2_full']:.6f}",
        f"- Mean trace correlation (full): {metrics['ux']['mean_trace_correlation_full']:.6f}",
        "",
        f"uy / {uy_channel} metrics:",
        f"- Relative L2 (full): {metrics['uy']['relative_l2_full']:.6f}",
        f"- Relative L2 (early): {metrics['uy']['relative_l2_early']:.6f}",
        f"- Scaled relative L2 (full): {metrics['uy']['scaled_relative_l2_full']:.6f}",
        f"- Mean trace correlation (full): {metrics['uy']['mean_trace_correlation_full']:.6f}",
        "",
        f"uz / {uz_channel} metrics:",
        f"- Relative L2 (full): {metrics['uz']['relative_l2_full']:.6f}",
        f"- Relative L2 (early): {metrics['uz']['relative_l2_early']:.6f}",
        f"- Scaled relative L2 (full): {metrics['uz']['scaled_relative_l2_full']:.6f}",
        f"- Mean trace correlation (full): {metrics['uz']['mean_trace_correlation_full']:.6f}",
        "",
        "Generated files:",
        f"- {comparison_dir / 'ux_bxx_comparison.png'}",
        f"- {comparison_dir / 'uy_bxy_comparison.png'}",
        f"- {comparison_dir / 'uz_bxz_comparison.png'}",
        f"- {comparison_dir / 'trace_comparison.png'}",
        f"- {comparison_dir / 'comparison_summary.json'}",
    ]
    (comparison_dir / "comparison_summary.txt").write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Comparison results saved to {comparison_dir}")


if __name__ == "__main__":
    main()
