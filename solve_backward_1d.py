#!/usr/bin/env python3
"""Finite-difference solver for a reduced one-dimensional ROOT-FPT equation.

It solves, in time-to-deadline tau,
    p_tau = D p_xx + (lambda-mu) p - lambda p^2,
with p(0,tau)=1 (target), p_x(L,tau)=0, p(x,0)=0 outside the target.
The result is a verification/illustration of the reduced independent-descendant
limit, not a calibrated root model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def solve(
    D: float, lam: float, mu: float, L: float, deadline: float, nx: int = 401, safety: float = 0.42
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, L, nx)
    dx = x[1] - x[0]
    dt = safety * dx * dx / D
    nsteps = int(np.ceil(deadline / dt))
    dt = deadline / nsteps
    p = np.zeros_like(x)
    p[0] = 1.0
    for _ in range(nsteps):
        lap = np.empty_like(p)
        lap[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / (dx * dx)
        lap[-1] = 2 * (p[-2] - p[-1]) / (dx * dx)  # reflecting far boundary
        lap[0] = 0.0
        p[1:] += dt * (D * lap[1:] + (lam - mu) * p[1:] - lam * p[1:] ** 2)
        p[0] = 1.0
        p = np.clip(p, 0.0, 1.0)
    return x, p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/backward_1d"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    D = 0.08
    mu = 0.04
    L = 6.0
    deadline = 18.0
    lams = [0.0, 0.03, 0.08, 0.16]
    rows = []
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for lam in lams:
        x, p = solve(D, lam, mu, L, deadline)
        ax.plot(x, p, label=rf"$\lambda={lam:.2f}$")
        for xx, pp in zip(x, p, strict=False):
            rows.append({"x": xx, "success_probability": pp, "branch_rate": lam})
    with (args.output_dir / "backward_1d_profiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    ax.set_xlabel("Initial distance from target")
    ax.set_ylabel("Success probability before deadline")
    ax.set_ylim(0, 1)
    ax.set_title("Reduced nonlinear backward equation\nindependent-descendant limit; synthetic")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "backward_1d_profiles.png", dpi=230)
    plt.close(fig)
    print(args.output_dir / "backward_1d_profiles.csv")


if __name__ == "__main__":
    main()
