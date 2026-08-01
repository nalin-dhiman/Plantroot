#!/usr/bin/env python3
"""Reduced ROOT-FPT demonstrator.

This is a deliberately minimal, synthetic model for software verification and
concept illustration. It is not the full ecohydraulic model and its output is
not empirical evidence.

The model evolves root tips as persistent, biased stochastic walkers in a 2-D
soil field. Tips branch through a Poisson event process and share a finite
construction budget. The full research model described in the report adds a
Richards-equation soil module, root hydraulic network, carbon dynamics,
mechanical impedance, calibration, and uncertainty quantification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MoisturePatch:
    x: float
    z: float
    amplitude: float
    sigma_x: float
    sigma_z: float


@dataclass
class ModelConfig:
    seed: int = 18
    dt: float = 0.02
    duration: float = 28.0
    domain_x_min: float = -1.2
    domain_x_max: float = 1.2
    domain_z_min: float = 0.0
    domain_z_max: float = 2.2
    base_moisture: float = 0.12
    depth_gradient: float = 0.10
    evaporation_length: float = 0.42
    speed: float = 0.085
    rotational_diffusion: float = 0.22
    gravitropism: float = 0.70
    hydrotropism: float = 3.20
    branch_rate: float = 0.11
    branch_angle_mean: float = 0.62
    branch_angle_sd: float = 0.18
    minimum_branch_age: float = 2.2
    mortality_rate: float = 0.004
    drought_mortality: float = 0.045
    moisture_survival_scale: float = 0.16
    construction_budget: float = 11.5
    branch_cost: float = 0.035
    length_cost: float = 1.0
    max_active_tips: int = 110
    target_moisture: float = 0.55
    patches: list[MoisturePatch] = field(
        default_factory=lambda: [
            MoisturePatch(-0.58, 0.76, 0.70, 0.20, 0.16),
            MoisturePatch(0.48, 1.22, 0.82, 0.24, 0.20),
            MoisturePatch(-0.08, 1.84, 0.74, 0.30, 0.17),
        ]
    )

    @classmethod
    def from_json(cls, path: Path) -> ModelConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "patches" in data:
            data["patches"] = [MoisturePatch(**p) for p in data["patches"]]
        return cls(**data)

    def to_jsonable(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class Tip:
    tip_id: int
    parent_tip_id: int | None
    x: float
    z: float
    theta: float
    age: float = 0.0
    active: bool = True
    path_x: list[float] = field(default_factory=list)
    path_z: list[float] = field(default_factory=list)

    def initialize_path(self) -> None:
        self.path_x = [self.x]
        self.path_z = [self.z]


@dataclass
class SimulationResult:
    tips: list[Tip]
    elapsed: float
    length_used: float
    branch_events: int
    hit: bool
    first_hit_time: float | None
    hit_tip_id: int | None
    hit_position: tuple[float, float] | None
    active_tip_history: list[tuple[float, int]]


class MoistureField:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def value_and_gradient(self, x: float, z: float, t: float) -> tuple[float, float, float]:
        # A low-order synthetic field: a dry surface, a weak depth trend, and
        # localized wet patches. A mild temporal decay makes the environment
        # nonstationary without pretending to solve Richards' equation.
        c = self.cfg
        value = c.base_moisture + c.depth_gradient * (1.0 - math.exp(-z / c.evaporation_length))
        gx = 0.0
        gz = c.depth_gradient * math.exp(-z / c.evaporation_length) / c.evaporation_length
        decay = math.exp(-0.010 * t)
        for p in c.patches:
            dx = x - p.x
            dz = z - p.z
            exponent = -0.5 * ((dx / p.sigma_x) ** 2 + (dz / p.sigma_z) ** 2)
            g = p.amplitude * decay * math.exp(exponent)
            value += g
            gx += g * (-dx / (p.sigma_x**2))
            gz += g * (-dz / (p.sigma_z**2))
        return float(np.clip(value, 0.0, 1.0)), gx, gz

    def grid(
        self, nx: int = 320, nz: int = 280, t: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.cfg
        xs = np.linspace(c.domain_x_min, c.domain_x_max, nx)
        zs = np.linspace(c.domain_z_min, c.domain_z_max, nz)
        w = np.empty((nz, nx), dtype=float)
        for iz, z in enumerate(zs):
            for ix, x in enumerate(xs):
                w[iz, ix] = self.value_and_gradient(float(x), float(z), t)[0]
        return xs, zs, w


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def angular_difference(target: float, current: float) -> float:
    return wrap_angle(target - current)


def simulate(cfg: ModelConfig) -> SimulationResult:
    rng = np.random.default_rng(cfg.seed)
    field_model = MoistureField(cfg)
    initial = Tip(tip_id=0, parent_tip_id=None, x=0.0, z=0.02, theta=math.pi / 2)
    initial.initialize_path()
    tips: list[Tip] = [initial]
    next_tip_id = 1
    t = 0.0
    length_used = 0.0
    branch_events = 0
    hit = False
    first_hit_time: float | None = None
    hit_tip_id: int | None = None
    hit_position: tuple[float, float] | None = None
    active_tip_history: list[tuple[float, int]] = []

    while t < cfg.duration and length_used < cfg.construction_budget:
        active_tips = [tip for tip in tips if tip.active]
        active_tip_history.append((t, len(active_tips)))
        if not active_tips:
            break

        # Shared budget: simultaneous tips divide the remaining construction
        # capacity in this reduced model.
        budget_left = cfg.construction_budget - length_used
        max_step_total = cfg.speed * cfg.dt * len(active_tips)
        step_scale = min(1.0, budget_left / max(max_step_total, 1e-12))
        newborns: list[Tip] = []

        for tip in active_tips:
            moisture, gx, gz = field_model.value_and_gradient(tip.x, tip.z, t)
            grad_norm = math.hypot(gx, gz)

            gravity_target = math.pi / 2
            gravity_turn = cfg.gravitropism * math.sin(
                angular_difference(gravity_target, tip.theta)
            )

            hydro_turn = 0.0
            if grad_norm > 1e-10:
                hydro_target = math.atan2(gz, gx)
                saturation = grad_norm / (0.25 + grad_norm)
                hydro_turn = (
                    cfg.hydrotropism
                    * saturation
                    * math.sin(angular_difference(hydro_target, tip.theta))
                )

            noise = math.sqrt(2.0 * cfg.rotational_diffusion * cfg.dt) * float(rng.normal())
            tip.theta = wrap_angle(tip.theta + (gravity_turn + hydro_turn) * cfg.dt + noise)

            # Growth slows in very dry soil but never becomes exactly zero in
            # this demonstration.
            moisture_factor = 0.35 + 0.65 * moisture
            ds = cfg.speed * moisture_factor * cfg.dt * step_scale
            tip.x += ds * math.cos(tip.theta)
            tip.z += ds * math.sin(tip.theta)
            tip.age += cfg.dt
            length_used += cfg.length_cost * ds
            tip.path_x.append(tip.x)
            tip.path_z.append(tip.z)

            if (
                tip.x < cfg.domain_x_min
                or tip.x > cfg.domain_x_max
                or tip.z < cfg.domain_z_min
                or tip.z > cfg.domain_z_max
            ):
                tip.active = False
                continue

            new_moisture, _, _ = field_model.value_and_gradient(tip.x, tip.z, t)
            if not hit and new_moisture >= cfg.target_moisture:
                hit = True
                first_hit_time = t
                hit_tip_id = tip.tip_id
                hit_position = (tip.x, tip.z)

            drought_hazard = cfg.drought_mortality * max(
                0.0, (cfg.moisture_survival_scale - new_moisture) / cfg.moisture_survival_scale
            )
            if rng.random() < 1.0 - math.exp(-(cfg.mortality_rate + drought_hazard) * cfg.dt):
                tip.active = False
                continue

            can_branch = (
                tip.age >= cfg.minimum_branch_age
                and len(active_tips) + len(newborns) < cfg.max_active_tips
                and length_used + cfg.branch_cost < cfg.construction_budget
            )
            if can_branch and rng.random() < 1.0 - math.exp(-cfg.branch_rate * cfg.dt):
                sign = -1.0 if rng.random() < 0.5 else 1.0
                offset = sign * max(
                    0.12, float(rng.normal(cfg.branch_angle_mean, cfg.branch_angle_sd))
                )
                child = Tip(
                    tip_id=next_tip_id,
                    parent_tip_id=tip.tip_id,
                    x=tip.x,
                    z=tip.z,
                    theta=wrap_angle(tip.theta + offset),
                )
                child.initialize_path()
                newborns.append(child)
                next_tip_id += 1
                branch_events += 1
                length_used += cfg.branch_cost

        tips.extend(newborns)
        t += cfg.dt

    return SimulationResult(
        tips=tips,
        elapsed=t,
        length_used=length_used,
        branch_events=branch_events,
        hit=hit,
        first_hit_time=first_hit_time,
        hit_tip_id=hit_tip_id,
        hit_position=hit_position,
        active_tip_history=active_tip_history,
    )


def save_summary(result: SimulationResult, cfg: ModelConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_status": "synthetic reduced demonstrator; not empirical evidence",
        "elapsed": result.elapsed,
        "length_used": result.length_used,
        "branch_events": result.branch_events,
        "total_tips": len(result.tips),
        "active_tips_final": sum(t.active for t in result.tips),
        "target_hit": result.hit,
        "first_hit_time": result.first_hit_time,
        "hit_tip_id": result.hit_tip_id,
        "hit_position": result.hit_position,
        "config": cfg.to_jsonable(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "root_segments.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tip_id", "parent_tip_id", "point_index", "x", "z", "active_final"])
        for tip in result.tips:
            for j, (x, z) in enumerate(zip(tip.path_x, tip.path_z, strict=False)):
                writer.writerow([tip.tip_id, tip.parent_tip_id, j, x, z, int(tip.active)])

    with (output_dir / "active_tip_history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "active_tips"])
        writer.writerows(result.active_tip_history)


def plot_architecture(result: SimulationResult, cfg: ModelConfig, output_path: Path) -> None:
    field_model = MoistureField(cfg)
    xs, zs, w = field_model.grid(t=result.elapsed)
    fig, ax = plt.subplots(figsize=(7.5, 7.3))
    image = ax.imshow(
        w,
        extent=[xs.min(), xs.max(), zs.max(), zs.min()],
        aspect="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    for tip in result.tips:
        if len(tip.path_x) < 2:
            continue
        linewidth = 1.7 if tip.parent_tip_id is None else 0.95
        ax.plot(tip.path_x, tip.path_z, color="#5d3a1a", linewidth=linewidth, alpha=0.88)
    if result.hit_position is not None:
        ax.scatter(
            [result.hit_position[0]],
            [result.hit_position[1]],
            s=90,
            marker="*",
            color="#b00020",
            zorder=5,
        )
    ax.scatter([0.0], [0.02], s=55, marker="o", color="#2e7d32", zorder=5)
    ax.set_xlim(cfg.domain_x_min, cfg.domain_x_max)
    ax.set_ylim(cfg.domain_z_max, cfg.domain_z_min)
    ax.set_xlabel("Horizontal position (arbitrary units)")
    ax.set_ylabel("Depth (arbitrary units)")
    ax.set_title("Illustrative reduced ROOT-FPT realization (synthetic)")
    cbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.04)
    cbar.set_label("Synthetic moisture index")
    ax.text(
        0.02,
        0.98,
        f"tips={len(result.tips)}; branches={result.branch_events}; "
        f"budget={result.length_used:.2f}; hit={result.hit}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.86,
            "edgecolor": "0.65",
        },
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tip_history(result: SimulationResult, output_path: Path) -> None:
    arr = np.asarray(result.active_tip_history, dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    if arr.size:
        ax.step(arr[:, 0], arr[:, 1], where="post", linewidth=1.8)
    ax.set_xlabel("Time (arbitrary units)")
    ax.set_ylabel("Number of active tips")
    ax.set_title("Illustrative branching history (synthetic)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON configuration file.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--seed", type=int, help="Override the random seed.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = ModelConfig.from_json(args.config) if args.config else ModelConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    result = simulate(cfg)
    save_summary(result, cfg, args.output_dir)
    plot_architecture(result, cfg, args.output_dir / "illustrative_root_architecture.png")
    plot_tip_history(result, args.output_dir / "illustrative_tip_history.png")
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(args.output_dir),
                "target_hit": result.hit,
                "first_hit_time": result.first_hit_time,
                "branch_events": result.branch_events,
                "total_tips": len(result.tips),
                "length_used": round(result.length_used, 4),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
