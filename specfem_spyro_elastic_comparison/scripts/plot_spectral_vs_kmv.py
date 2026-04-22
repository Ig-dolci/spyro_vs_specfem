#!/usr/bin/env python3
"""Generate comparison plots: Spectral vs KMV convergence with theoretical rate.

Produces 4 figures:
  - 2D global field convergence (spectral vs KMV vs theoretical)
  - 2D receiver convergence
  - 3D global field convergence
  - 3D receiver convergence
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

SPEC_COLOR = "C0"
KMV_COLOR = "C1"
THEORY_COLOR = "gray"


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def theoretical_line(h_arr, order, ref_h, ref_err):
    """Scale reference error by (h/ref_h)^order."""
    return ref_err * (np.array(h_arr) / ref_h) ** order


def plot_global_comparison(ax, spec_rows, kmv_rows, spec_order, kmv_order, spec_label, kmv_label):
    """Plot global L2 error vs h for spectral and KMV with theoretical lines."""
    h_s = [r["h"] for r in spec_rows]
    err_s = [r["rel_error"] for r in spec_rows]
    h_k = [r["h"] for r in kmv_rows]
    err_k = [r["rel_error"] for r in kmv_rows]

    ax.loglog(h_s, err_s, "o-", color=SPEC_COLOR, label=spec_label, markersize=6, linewidth=1.5)
    ax.loglog(h_k, err_k, "s-", color=KMV_COLOR, label=kmv_label, markersize=6, linewidth=1.5)

    # Theoretical lines anchored at coarsest point of each
    h_range_s = np.linspace(min(h_s) * 0.8, max(h_s) * 1.2, 50)
    th_s = theoretical_line(h_range_s, spec_order, h_s[0], err_s[0])
    ax.loglog(h_range_s, th_s, "--", color=SPEC_COLOR, alpha=0.4,
              label=f"$O(h^{{{spec_order:.0f}}})$ (spectral)")

    h_range_k = np.linspace(min(h_k) * 0.8, max(h_k) * 1.2, 50)
    th_k = theoretical_line(h_range_k, kmv_order, h_k[0], err_k[0])
    ax.loglog(h_range_k, th_k, "--", color=KMV_COLOR, alpha=0.4,
              label=f"$O(h^{{{kmv_order:.0f}}})$ (KMV)")

    ax.set_xlabel("$h$ (edge length)")
    ax.set_ylabel("Relative $L^2$ error")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)


def plot_receiver_comparison(ax, spec_summary, kmv_summary, spec_order, kmv_order, spec_label, kmv_label):
    """Plot receiver L2 error vs h for spectral and KMV."""
    spec_recv = spec_summary.get("receiver_metrics", [])
    kmv_recv = kmv_summary.get("receiver_metrics", [])

    if not spec_recv or not kmv_recv:
        ax.text(0.5, 0.5, "No receiver data", transform=ax.transAxes, ha="center")
        return

    h_s = [r["h"] for r in spec_recv]
    err_s = [r["u_rel_l2_full"] for r in spec_recv]
    h_k = [r["h"] for r in kmv_recv]
    err_k = [r["u_rel_l2_full"] for r in kmv_recv]

    # Filter out zero/nan
    valid_s = [(h, e) for h, e in zip(h_s, err_s) if e > 0 and np.isfinite(e)]
    valid_k = [(h, e) for h, e in zip(h_k, err_k) if e > 0 and np.isfinite(e)]

    if valid_s:
        hs, es = zip(*valid_s)
        ax.loglog(hs, es, "o-", color=SPEC_COLOR, label=spec_label, markersize=6, linewidth=1.5)
        h_range = np.linspace(min(hs) * 0.8, max(hs) * 1.2, 50)
        th = theoretical_line(h_range, spec_order, hs[0], es[0])
        ax.loglog(h_range, th, "--", color=SPEC_COLOR, alpha=0.4,
                  label=f"$O(h^{{{spec_order:.0f}}})$")

    if valid_k:
        hk, ek = zip(*valid_k)
        ax.loglog(hk, ek, "s-", color=KMV_COLOR, label=kmv_label, markersize=6, linewidth=1.5)
        h_range = np.linspace(min(hk) * 0.8, max(hk) * 1.2, 50)
        th = theoretical_line(h_range, kmv_order, hk[0], ek[0])
        ax.loglog(h_range, th, "--", color=KMV_COLOR, alpha=0.4,
                  label=f"$O(h^{{{kmv_order:.0f}}})$")

    ax.set_xlabel("$h$ (edge length)")
    ax.set_ylabel("Relative $L^2$ error (receivers)")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)


def make_comparison_figure(dim_label, spec_dir, kmv_dir, output_prefix,
                           spec_degree, kmv_degree):
    spec_summary = load_summary(spec_dir / f"{spec_dir.name}_summary.json")
    kmv_summary = load_summary(kmv_dir / f"{kmv_dir.name}_summary.json")

    spec_rows = spec_summary["space_convergence"]
    kmv_rows = kmv_summary["space_convergence"]
    spec_order = spec_degree + 1
    kmv_order = kmv_degree + 1

    spec_label = f"Spectral (p={spec_degree})"
    kmv_label = f"KMV (p={kmv_degree})"

    # Figure 1: Global field error
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4.5))
    plot_global_comparison(ax1, spec_rows, kmv_rows, spec_order, kmv_order,
                           spec_label, kmv_label)
    ax1.set_title(f"Convergência global do campo — {dim_label}")
    fig1.tight_layout()
    path1 = RESULTS / f"{output_prefix}_global_comparison.png"
    fig1.savefig(path1, bbox_inches="tight")
    print(f"Saved: {path1}")
    plt.close(fig1)

    # Figure 2: Receiver error
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4.5))
    plot_receiver_comparison(ax2, spec_summary, kmv_summary, spec_order, kmv_order,
                             spec_label, kmv_label)
    ax2.set_title(f"Convergência nos receptores — {dim_label}")
    fig2.tight_layout()
    path2 = RESULTS / f"{output_prefix}_receiver_comparison.png"
    fig2.savefig(path2, bbox_inches="tight")
    print(f"Saved: {path2}")
    plt.close(fig2)


if __name__ == "__main__":
    # 2D: both p=4
    make_comparison_figure(
        "2D",
        RESULTS / "global_selfconv_top_free_2d",
        RESULTS / "global_selfconv_top_free_2d_kmv",
        "spectral_vs_kmv_2d",
        spec_degree=4, kmv_degree=4,
    )

    # 3D: spectral p=4, KMV p=3
    make_comparison_figure(
        "3D",
        RESULTS / "global_selfconv_top_free_3d",
        RESULTS / "global_selfconv_top_free_3d_kmv",
        "spectral_vs_kmv_3d",
        spec_degree=4, kmv_degree=3,
    )
