"""Publication-quality plotting helpers for the interactive explorer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from rootfpt.explorer import ExperimentResult

ORDER_COLORS = {0: "#172033", 1: "#2563EB", 2: "#E8792E"}
ARCHITECTURE_COLORS = ("#2563EB", "#E8792E")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def architecture_figure(
    result: ExperimentResult,
    architecture_label: str,
    soil_label: str,
) -> plt.Figure:
    """Render soil water, mechanical structure, and root order without overlap."""
    _style()
    soil = result.soil
    grid = soil.grid
    x0, x1 = grid.x_limits
    z0, z1 = grid.z_limits
    figure, axis = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    image = axis.imshow(
        soil.water,
        extent=(x0, x1, z1, z0),
        origin="upper",
        cmap="Blues",
        vmin=0.08,
        vmax=0.46,
        alpha=0.78,
        interpolation="bilinear",
        aspect="equal",
    )
    horizontal, depth = grid.mesh
    if float(np.max(soil.impedance)) >= 1.2:
        axis.contour(
            horizontal,
            depth,
            soil.impedance,
            levels=[1.2],
            colors=["#8B5E3C"],
            linewidths=1.4,
        )
    anisotropy = np.sqrt(
        (soil.anisotropy_xx - 1.0) ** 2
        + 2.0 * soil.anisotropy_xz**2
        + (soil.anisotropy_zz - 1.0) ** 2
    )
    if float(np.max(anisotropy)) >= 1.0:
        axis.contour(
            horizontal,
            depth,
            anisotropy,
            levels=[1.0],
            colors=["#16865C"],
            linewidths=1.2,
            linestyles="--",
        )

    for order in (0, 1, 2):
        lines = [
            [segment.start, segment.end]
            for segment in result.architecture.segments
            if min(segment.order, 2) == order
        ]
        if lines:
            axis.add_collection(
                LineCollection(
                    lines,
                    colors=ORDER_COLORS[order],
                    linewidths=2.0 if order == 0 else 1.25,
                    capstyle="round",
                    zorder=4,
                )
            )
    axis.scatter(
        [0.0],
        [0.0],
        marker="s",
        s=42,
        color="#7C3AED",
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    view_x0, view_x1, view_z1 = x0, x1, z1
    if float(result.settings.get("domain_scale", 1.0)) > 1.0:
        root_x = result.architecture.nodes[:, 0]
        horizontal_padding = max(1.0, 0.08 * float(np.ptp(root_x)))
        view_x0 = max(x0, float(np.min(root_x)) - horizontal_padding)
        view_x1 = min(x1, float(np.max(root_x)) + horizontal_padding)
        view_z1 = min(z1, float(result.depth_bins[-1]))
    axis.set(
        xlim=(view_x0, view_x1),
        ylim=(view_z1, z0),
        xlabel="Horizontal position (cm)",
        ylabel="Depth (cm)",
    )
    axis.set_title(f"{architecture_label} in {soil_label}", pad=13, fontweight="bold")
    axis.grid(color="white", linewidth=0.7, alpha=0.45)
    colorbar = figure.colorbar(image, ax=axis, location="right", fraction=0.045, pad=0.035)
    colorbar.set_label("Volumetric water content (cm³ cm⁻³)")
    colorbar.outline.set_linewidth(0.6)

    handles = [
        Line2D([0], [0], color=ORDER_COLORS[0], lw=2.2, label="Primary order"),
        Line2D([0], [0], color=ORDER_COLORS[1], lw=1.8, label="First-order lateral"),
        Line2D([0], [0], color=ORDER_COLORS[2], lw=1.8, label="Higher order"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#7C3AED",
            markeredgecolor="white",
            markersize=7,
            label="Collar",
        ),
    ]
    if float(np.max(soil.impedance)) >= 1.2:
        handles.append(Line2D([0], [0], color="#8B5E3C", lw=1.4, label="Impedance ≥ 1.2 MPa"))
    if float(np.max(anisotropy)) >= 1.0:
        handles.append(
            Line2D([0], [0], color="#16865C", lw=1.2, ls="--", label="Anisotropic channel")
        )
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.25,
        handlelength=2.1,
    )
    return figure


def root_length_density_figure(result: ExperimentResult) -> plt.Figure:
    """Plot the depth profile on explicit bin intervals."""
    _style()
    midpoints = 0.5 * (result.depth_bins[:-1] + result.depth_bins[1:])
    figure, axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    axis.plot(
        result.root_length_density,
        midpoints,
        marker="o",
        markersize=5,
        linewidth=2,
        color="#2563EB",
    )
    axis.fill_betweenx(midpoints, 0, result.root_length_density, color="#93C5FD", alpha=0.3)
    axis.set_ylim(float(result.depth_bins[-1]), float(result.depth_bins[0]))
    axis.set_xlim(left=0)
    axis.set_xlabel("Root length per depth interval (cm cm⁻¹)")
    axis.set_ylabel("Depth-bin midpoint (cm)")
    axis.set_title("Root-length density profile", fontweight="bold")
    axis.grid(alpha=0.22)
    return figure


def comparison_figure(
    frame: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> plt.Figure:
    """Show paired-seed comparisons without implying population inference."""
    _style()
    metrics = (
        ("maximum_depth_cm", "Maximum depth (cm)"),
        ("total_root_length_cm", "Total length (cm)"),
        ("branch_count", "Branch count"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        wide = frame.pivot(index="replicate", columns="architecture", values=metric)
        for _, row in wide.iterrows():
            axis.plot([0, 1], [row["a"], row["b"]], color="#94A3B8", alpha=0.65, lw=1)
        axis.scatter(
            np.zeros(len(wide)),
            wide["a"],
            color=ARCHITECTURE_COLORS[0],
            s=32,
            zorder=3,
        )
        axis.scatter(
            np.ones(len(wide)),
            wide["b"],
            color=ARCHITECTURE_COLORS[1],
            s=32,
            zorder=3,
        )
        axis.set_xticks([0, 1], [label_a, label_b], rotation=12, ha="right")
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", alpha=0.22)
        axis.margins(x=0.35)
    figure.suptitle("Paired synthetic realizations", fontsize=13, fontweight="bold")
    return figure
