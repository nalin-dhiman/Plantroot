#!/usr/bin/env python3
"""Monte Carlo verification of the ROOT-FPT fixed-effort null law.

Under a constant independent hit hazard k per unit active tip-time, all branch
schedules with the same integrated effort A have hit probability 1-exp(-k A).
This is a mathematical verification, not a biological simulation.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def schedules(total_effort: float, dt: float) -> dict[str, np.ndarray]:
    # Each array gives N(t) on a step grid and is rescaled so sum N*dt=A.
    raw = {
        "one persistent tip": np.ones(int(total_effort / dt)),
        "four simultaneous tips": np.full(int(total_effort / (4 * dt)), 4.0),
        "sixteen simultaneous tips": np.full(max(1, int(total_effort / (16 * dt))), 16.0),
        "progressive branching": np.concatenate(
            [np.ones(30), np.full(20, 2.0), np.full(15, 4.0), np.full(10, 8.0)]
        ),
    }
    out = {}
    for name, n in raw.items():
        out[name] = n * (total_effort / (n.sum() * dt))
    return out


def simulate_schedule(
    n_t: np.ndarray, k: float, dt: float, reps: int, rng: np.random.Generator
) -> float:
    # At each step no-hit probability is exp(-k*N*dt). Stop after first event.
    hits = 0
    for _ in range(reps):
        survived = True
        for n in n_t:
            if rng.random() < 1.0 - math.exp(-k * float(n) * dt):
                survived = False
                break
        hits += int(not survived)
    return hits / reps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/baselines"))
    parser.add_argument("--reps", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dt = 0.02
    k = 0.075
    effort_grid = np.linspace(1.0, 24.0, 13)
    rng = np.random.default_rng(args.seed)
    rows = []
    for effort in effort_grid:
        theory = 1.0 - math.exp(-k * effort)
        for name, n_t in schedules(float(effort), dt).items():
            estimate = simulate_schedule(n_t, k, dt, args.reps, rng)
            rows.append(
                {"effort": effort, "schedule": name, "estimate": estimate, "theory": theory}
            )

    with (args.output_dir / "fixed_effort_verification.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.plot(effort_grid, 1.0 - np.exp(-k * effort_grid), linewidth=2.2, label=r"theory $1-e^{-kA}$")
    names = list(schedules(1.0, dt).keys())
    markers = ["o", "s", "^", "D"]
    for name, marker in zip(names, markers, strict=False):
        rr = [r for r in rows if r["schedule"] == name]
        ax.scatter(
            [r["effort"] for r in rr], [r["estimate"] for r in rr], s=25, marker=marker, label=name
        )
    ax.set_xlabel("Integrated tip-time effort A")
    ax.set_ylabel("Target-hit probability")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Fixed-effort no-free-lunch baseline\nconstant independent hazard; synthetic verification"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fixed_effort_verification.png", dpi=230)
    plt.close(fig)
    max_error = max(abs(float(r["estimate"]) - float(r["theory"])) for r in rows)
    print(f"max absolute Monte Carlo error: {max_error:.4f}")


if __name__ == "__main__":
    main()
