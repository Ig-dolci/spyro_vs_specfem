#!/usr/bin/env python3
"""Generate detailed trace comparison plots (Spyro vs SPECFEM) with error panels
and zoom insets highlighting the differences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT_DIR = Path(__file__).resolve().parents[1]

SPECFEM2D_RICKER_SHIFT = 0.24  # SPECFEM2D external Ricker delay used in this case


def load_spyro_2d(spyro_dir: Path):
    data = np.load(spyro_dir / "spyro_receivers.npz")
    time = data["time"]
    recv = data["receivers_output"]  # (ntime, nrecv, 2) — [iz, ix]
    return time, recv[:, :, 1], recv[:, :, 0]  # ux, uz


def load_specfem_2d(specfem_dir: Path, nrecv: int, component: str):
    traces = []
    seis_dir = specfem_dir / "raw_seismograms"
    time_axis = None
    for i in range(1, nrecv + 1):
        fname = seis_dir / f"AA.S{i:04d}.{component}.semd"
        d = np.loadtxt(fname)
        if time_axis is None:
            time_axis = d[:, 0] + SPECFEM2D_RICKER_SHIFT
        traces.append(d[:, 1])
    return time_axis, np.column_stack(traces)


def load_spyro_3d(spyro_dir: Path):
    data = np.load(spyro_dir / "spyro_receivers.npz")
    time = data["time"]
    recv = data["receivers_output"]  # (ntime, nrecv, 3) — [iz, ix, iy]
    return time, recv[:, :, 1], recv[:, :, 2], recv[:, :, 0]  # ux, uy, uz


def resolve_spyro_3d_source_delay(spyro_dir: Path, default: float = 0.2) -> float:
    metadata_path = spyro_dir / "spyro_metadata.json"
    if not metadata_path.exists():
        return default
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return float(metadata.get("source_delay") or default)


def load_specfem_3d(specfem_dir: Path, nrecv: int, component: str, time_shift: float):
    traces = []
    seis_dir = specfem_dir / "raw_seismograms"
    time_axis = None
    for i in range(1, nrecv + 1):
        fname = seis_dir / f"AA.S{i:04d}.{component}.semd"
        d = np.loadtxt(fname)
        if time_axis is None:
            time_axis = d[:, 0] + time_shift
        traces.append(d[:, 1])
    return time_axis, np.column_stack(traces)


def interp_to_common_time(time_ref, time_other, traces_other):
    """Interpolate traces_other onto time_ref."""
    out = np.empty((len(time_ref), traces_other.shape[1]))
    for j in range(traces_other.shape[1]):
        out[:, j] = np.interp(
            time_ref,
            time_other,
            traces_other[:, j],
            left=np.nan,
            right=np.nan,
        )
    return out


def trim_to_shared_time_window(time_spy, *spy_components, time_spec):
    """Restrict Spyro and SPECFEM plots to the time range available in both."""
    t_end = min(time_spy[-1], time_spec[-1])
    mask = time_spy <= t_end + 1.0e-12
    return time_spy[mask], [component[mask, :] for component in spy_components]


def plot_trace_with_error(ax_trace, ax_err, ax_zoom, time, spy, spec,
                          comp_label, zoom_center=None, zoom_half=0.08):
    """Plot overlaid traces, error panel, and zoomed inset for the central receiver."""
    nrecv = spy.shape[1]
    mid = nrecv // 2
    s = spy[:, mid]
    sp = spec[:, mid]
    diff = s - sp

    # Trace panel: single receiver, Spyro vs SPECFEM
    ax_trace.plot(time, s, color="#1f77b4", linewidth=1.0, label="Spyro")
    ax_trace.plot(time, sp, color="#ff7f0e", linewidth=1.0, linestyle="--",
                  label="SPECFEM")
    ax_trace.set_ylabel(f"Amplitude {comp_label}")
    ax_trace.legend(loc="upper right", fontsize=8, ncol=2)
    ax_trace.grid(True, alpha=0.3)
    ax_trace.set_title(
        f"Traços sobrepostos — {comp_label} (receptor {mid+1}/{nrecv})",
        fontsize=10)

    # Error panel
    ax_err.plot(time, diff, color="#d62728", linewidth=0.8)
    ax_err.set_ylabel(f"Erro (Spyro−SPECFEM)")
    ax_err.set_xlabel("Tempo (s)")
    ax_err.grid(True, alpha=0.3)
    ax_err.axhline(0, color="k", linewidth=0.5)
    ax_err.set_title(f"Diferença pontual — {comp_label}", fontsize=10)

    # Zoom inset — region around max error
    diff_abs = np.abs(diff)
    if zoom_center is None:
        i_max = np.argmax(diff_abs)
        zoom_center = time[i_max]

    t0 = max(time[0], zoom_center - zoom_half)
    t1 = min(time[-1], zoom_center + zoom_half)
    mask = (time >= t0) & (time <= t1)

    ax_zoom.plot(time[mask], s[mask], color="#1f77b4", linewidth=1.5,
                 label="Spyro")
    ax_zoom.plot(time[mask], sp[mask], color="#ff7f0e", linewidth=1.5,
                 linestyle="--", label="SPECFEM")
    ax_zoom.legend(fontsize=8, loc="best")
    ax_zoom.set_xlabel("Tempo (s)")
    ax_zoom.set_ylabel("Amplitude")
    ax_zoom.set_title(
        f"Zoom — {comp_label} (t ∈ [{t0:.2f}, {t1:.2f}] s)",
        fontsize=10)
    ax_zoom.grid(True, alpha=0.3)

    # Mark zoom region on trace panel
    ax_trace.axvspan(t0, t1, alpha=0.10, color="red")


def make_2d_figure(spyro_dir, specfem_dir, output_path):
    time_spy, ux_spy, uz_spy = load_spyro_2d(spyro_dir)
    nrecv = ux_spy.shape[1]

    time_spec, ux_spec = load_specfem_2d(specfem_dir, nrecv, "FXX")
    _, uz_spec = load_specfem_2d(specfem_dir, nrecv, "FXZ")

    time_common, trimmed_spy = trim_to_shared_time_window(
        time_spy, ux_spy, uz_spy, time_spec=time_spec
    )
    ux_spy, uz_spy = trimmed_spy
    ux_spec = interp_to_common_time(time_common, time_spec, ux_spec)
    uz_spec = interp_to_common_time(time_common, time_spec, uz_spec)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    plot_trace_with_error(axes[0, 0], axes[1, 0], axes[2, 0],
                          time_common, ux_spy, ux_spec, "$u_x$")
    plot_trace_with_error(axes[0, 1], axes[1, 1], axes[2, 1],
                          time_common, uz_spy, uz_spec, "$u_z$")

    fig.suptitle("Comparação 2D: Spyro vs SPECFEM2D — Traços e Erros",
                 fontsize=13, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def make_3d_figure(spyro_dir, specfem_dir, output_path):
    time_spy, ux_spy, uy_spy, uz_spy = load_spyro_3d(spyro_dir)
    nrecv = ux_spy.shape[1]
    specfem_time_shift = resolve_spyro_3d_source_delay(spyro_dir)

    time_spec, ux_spec = load_specfem_3d(specfem_dir, nrecv, "FXX", specfem_time_shift)
    _, uy_spec = load_specfem_3d(specfem_dir, nrecv, "FXY", specfem_time_shift)
    _, uz_spec = load_specfem_3d(specfem_dir, nrecv, "FXZ", specfem_time_shift)

    time_common, trimmed_spy = trim_to_shared_time_window(
        time_spy, ux_spy, uy_spy, uz_spy, time_spec=time_spec
    )
    ux_spy, uy_spy, uz_spy = trimmed_spy
    ux_spec = interp_to_common_time(time_common, time_spec, ux_spec)
    uy_spec = interp_to_common_time(time_common, time_spec, uy_spec)
    uz_spec = interp_to_common_time(time_common, time_spec, uz_spec)

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)

    plot_trace_with_error(axes[0, 0], axes[1, 0], axes[2, 0],
                          time_common, ux_spy, ux_spec, "$u_x$")
    plot_trace_with_error(axes[0, 1], axes[1, 1], axes[2, 1],
                          time_common, uy_spy, uy_spec, "$u_y$")
    plot_trace_with_error(axes[0, 2], axes[1, 2], axes[2, 2],
                          time_common, uz_spy, uz_spec, "$u_z$")

    fig.suptitle("Comparação 3D: Spyro vs SPECFEM3D — Traços e Erros",
                 fontsize=13, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed trace comparison plots with error panels.")
    parser.add_argument("--spyro-2d", type=Path,
                        default=ROOT_DIR / "results/elastic_evaluation/spyro_2d")
    parser.add_argument("--specfem-2d", type=Path,
                        default=ROOT_DIR / "results/elastic_evaluation/specfem_2d")
    parser.add_argument("--spyro-3d", type=Path,
                        default=ROOT_DIR / "results/elastic_evaluation/spyro_3d")
    parser.add_argument("--specfem-3d", type=Path,
                        default=ROOT_DIR / "results/elastic_evaluation/specfem_3d")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT_DIR / "results/elastic_evaluation")
    args = parser.parse_args()

    make_2d_figure(args.spyro_2d, args.specfem_2d,
                   args.output_dir / "trace_detail_2d.png")
    make_3d_figure(args.spyro_3d, args.specfem_3d,
                   args.output_dir / "trace_detail_3d.png")


if __name__ == "__main__":
    main()
