#!/usr/bin/env python3
"""Regenerate the README gallery from deterministic software outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rootfpt.explorer import labels, load_default_config, run_experiment  # noqa: E402
from rootfpt.visualization.explorer import ORDER_COLORS  # noqa: E402


def main() -> int:
    config = load_default_config()
    architecture_labels = labels(config, "root_regimes")
    architecture_names = list(architecture_labels)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "#edf4f7",
        }
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(11.5, 7.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, architecture_name in zip(axes.flat, architecture_names, strict=True):
        result = run_experiment(
            architecture_name,
            "homogeneous",
            seed=20260802,
            replicate=0,
            duration_days=5.5,
            dt_days=0.04,
            max_tips=120,
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
                        linewidths=1.7 if order == 0 else 1.15,
                        capstyle="round",
                    )
                )
        axis.scatter(
            [0],
            [0],
            marker="s",
            s=28,
            color="#7C3AED",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        axis.set_title(architecture_labels[architecture_name], fontweight="bold")
        axis.set_xlim(-6, 6)
        axis.set_ylim(12, 0)
        axis.grid(color="white", linewidth=0.7, alpha=0.8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Depth (cm)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Horizontal position (cm)")
    figure.suptitle(
        "Six synthetic architecture presets in one controlled soil",
        fontsize=15,
        fontweight="bold",
    )
    legend = [
        Line2D([0], [0], color=ORDER_COLORS[order], lw=2, label=f"order {order}")
        for order in (0, 1, 2)
    ]
    legend.append(
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color="#7C3AED",
            label="collar",
        )
    )
    figure.legend(
        legend,
        [item.get_label() for item in legend],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    destination = REPOSITORY_ROOT / "assets" / "preset_gallery.png"
    figure.savefig(destination, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(destination.relative_to(REPOSITORY_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
