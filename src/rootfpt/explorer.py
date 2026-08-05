"""Stable helpers for interactive, synthetic root--soil experiments.

The explorer deliberately exposes a small subset of ROOT-FPT.  Every preset
and soil constructor is synthetic and uncalibrated; the returned metrics are
software outputs, not measurements or agronomic recommendations.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass, replace
from importlib import resources

import numpy as np
import pandas as pd
import yaml

from rootfpt.multiscale.agent import TipTraits
from rootfpt.multiscale.architecture import (
    Architecture,
    EmergenceResponse,
    RootType,
    SiteStatus,
    simulate_architecture,
)
from rootfpt.multiscale.soil import Grid2D, SoilState
from rootfpt.multiscale.water import hydraulic_architecture_solution

ATLAS_DURATION_DAYS = 5.5
MAX_EXPLORER_DURATION_DAYS = 30.0


@dataclass(frozen=True)
class ExperimentResult:
    """One deterministic-by-seed architecture experiment."""

    architecture_name: str
    soil_name: str
    replicate: int
    architecture: Architecture
    soil: SoilState
    metrics: dict[str, float | int]
    root_length_density: np.ndarray
    depth_bins: np.ndarray
    seeds: dict[str, int]
    settings: dict[str, float | int | list[float]]


def load_default_config() -> dict:
    """Return an independent copy of the bundled explorer configuration."""
    config_file = resources.files("rootfpt").joinpath("data/root_soil_atlas.yaml")
    with config_file.open("r", encoding="utf-8") as handle:
        return copy.deepcopy(yaml.safe_load(handle))


def labels(config: dict, section: str) -> dict[str, str]:
    """Map internal configuration keys to concise display labels."""
    return {name: str(values["label"]) for name, values in config[section].items()}


def _seed_value(master: int, *indices: int) -> int:
    return int(np.random.SeedSequence([master, *indices]).generate_state(1)[0])


def _build_grid(config: dict) -> Grid2D:
    settings = config["simulation"]
    cells = int(settings["grid_cells"])
    return Grid2D(
        cells,
        cells,
        tuple(settings["domain_x_cm"]),
        tuple(settings["domain_z_cm"]),
    )


def _configure_explorer_domain(settings: dict, duration_days: float) -> float:
    """Use a shared month-scale domain while retaining atlas grid spacing.

    The frozen atlas window is unchanged. Every extended horizon uses the same
    expanded grid so seed-matched 7-, 14-, 21-, and 30-day runs share one soil
    realization instead of regenerating it at a different array size.
    """
    if duration_days <= ATLAS_DURATION_DAYS:
        return 1.0
    scale = MAX_EXPLORER_DURATION_DAYS / ATLAS_DURATION_DAYS
    settings["domain_x_cm"] = [float(value) * scale for value in settings["domain_x_cm"]]
    settings["domain_z_cm"] = [float(value) * scale for value in settings["domain_z_cm"]]
    settings["grid_cells"] = int(math.ceil(float(settings["grid_cells"]) * scale))
    settings["root_length_depth_bins_cm"] = [
        float(value) * scale for value in settings["root_length_depth_bins_cm"]
    ]
    return scale


def _soil_from_config(name: str, grid: Grid2D, rng: np.random.Generator) -> SoilState:
    if name == "homogeneous":
        return SoilState.homogeneous(grid, water=0.32, impedance=0.28, nutrient=0.75)
    if name == "patchy_matern":
        return SoilState.matern(
            grid,
            rng=rng,
            correlation_length=1.25,
            smoothness=1.5,
            cross_correlation=-0.55,
        )
    base = SoilState.homogeneous(grid, water=0.32, impedance=0.25, nutrient=0.75)
    if name == "layered":
        return base.layered(interface_depth=3.5, lower_impedance=1.9, lower_water=0.20)
    if name == "deep_water":
        _, depth = grid.mesh
        transition = 1.0 / (1.0 + np.exp(-(depth - 6.0) / 0.65))
        water = 0.14 + 0.27 * transition
        pressure = -200.0 - 1800.0 * np.clip(0.28 - water, 0.0, None)
        conductivity = 10.0 * np.clip(water / 0.28, 0.05, 1.8) ** 3
        return replace(
            base,
            water=water,
            pressure_head=pressure,
            hydraulic_conductivity=conductivity,
            description="dry surface with persistent deep wet layer",
        )
    if name == "compacted_lens":
        return base.with_compacted_lens(
            centre=(float(rng.normal(0.0, 0.25)), 4.6),
            radii=(2.5, 0.95),
            impedance=2.45,
        )
    if name == "cracked":
        jitter = float(rng.normal(0.0, 0.035))
        first = base.with_crack(
            angle=1.18 + jitter,
            offset=0.0,
            width=0.38,
            strength=8.0,
        )
        return first.with_crack(
            angle=1.92 - jitter,
            offset=0.85,
            width=0.30,
            strength=6.0,
        )
    raise KeyError(f"unknown soil constructor: {name}")


def _tip_traits(values: dict, common: dict) -> TipTraits:
    keys = (
        "speed",
        "rotational_diffusion",
        "kappa_gravity",
        "kappa_water",
        "kappa_mechanical",
        "kappa_anisotropy",
    )
    return TipTraits(
        **{key: float(values[key]) for key in keys},
        **{key: float(value) for key, value in common.items()},
    )


def _root_types(regime: dict, common: dict) -> dict[str, RootType]:
    def make(name: str, values: dict, successor: str | None) -> RootType:
        return RootType(
            name=name,
            tip_traits=_tip_traits(values, common),
            radius=float(values["radius"]),
            branch_spacing_mean=float(values["branch_spacing"]),
            branch_spacing_shape=float(values["branch_spacing_shape"]),
            developmental_delay_mean=float(values["delay"]),
            branch_angle_mean=float(values["branch_angle"]),
            branch_angle_sd=float(values["branch_angle_sd"]),
            mortality_rate=float(values["mortality_rate"]),
            successor=successor,
        )

    primary = make("primary", regime["primary"], "lateral")
    lateral = make("lateral", regime["lateral"], "second")
    lateral_values = regime["lateral"]
    second_values = {
        "speed": 0.45,
        "rotational_diffusion": 1.0,
        "kappa_gravity": 0.10,
        "kappa_water": 0.45,
        "kappa_mechanical": lateral_values["kappa_mechanical"],
        "kappa_anisotropy": lateral_values["kappa_anisotropy"],
        "radius": 0.025,
        "branch_spacing": 1.5,
        "branch_spacing_shape": 4.0,
        "delay": 0.50,
        "branch_angle": 0.72,
        "branch_angle_sd": 0.18,
        "mortality_rate": 0.01,
    }
    return {
        "primary": primary,
        "lateral": lateral,
        "second": make("second", second_values, None),
    }


def _mechanical_arrest_proxy(architecture: Architecture, soil: SoilState) -> float:
    if not architecture.sites:
        return 0.0
    positions = np.asarray([site.position for site in architecture.sites])
    high_impedance = soil.sample("impedance", positions) >= 1.2
    if not np.any(high_impedance):
        return 0.0
    failed = np.asarray(
        [site.status in {SiteStatus.DORMANT, SiteStatus.ABORTED} for site in architecture.sites]
    )
    return float(np.mean(failed[high_impedance]))


def _boundary_contact_count(architecture: Architecture, soil: SoilState) -> int:
    """Count tips whose last recorded node lies on the rectangular boundary."""
    x0, x1 = soil.grid.x_limits
    z0, z1 = soil.grid.z_limits
    count = 0
    for path in architecture.tip_paths.values():
        if len(path) < 2:
            continue
        x, z = architecture.nodes[path[-1]]
        if (
            np.isclose(x, x0, atol=1e-9)
            or np.isclose(x, x1, atol=1e-9)
            or np.isclose(z, z0, atol=1e-9)
            or np.isclose(z, z1, atol=1e-9)
        ):
            count += 1
    return count


def _metrics(
    architecture: Architecture,
    soil: SoilState,
    config: dict,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    settings = config["simulation"]
    base = architecture.metrics()
    bins = np.asarray(settings["root_length_depth_bins_cm"], dtype=float)
    if float(settings["duration_days"]) > ATLAS_DURATION_DAYS:
        occupied_depth = float(np.max(architecture.nodes[:, 1]))
        profile_depth = min(
            float(soil.grid.z_limits[1]),
            max(12.0, 1.1 * occupied_depth),
        )
        bins = np.linspace(0.0, profile_depth, len(bins))
    density = architecture.root_length_density_by_depth(bins)
    hydraulic = config["hydraulics"]
    solution = hydraulic_architecture_solution(
        architecture,
        soil,
        collar_pressure_head=float(hydraulic["collar_pressure_head_cm"]),
        axial_conductivity_by_type={
            key: float(value) for key, value in hydraulic["axial_conductivity"].items()
        },
        radial_conductivity_by_type={
            key: float(value) for key, value in hydraulic["radial_conductivity"].items()
        },
    )
    delivery_rate = max(float(solution.collar_delivered_flow), 0.0)
    cumulative_delivery = delivery_rate * float(settings["duration_days"])
    carbon_cost = (
        float(settings["carbon_length_cost"]) * base["total_length"]
        + float(settings["carbon_branch_cost"]) * base["branch_count"]
    )
    initial_carbon = float(settings["initial_carbon"])
    carbon_remaining = max(initial_carbon - carbon_cost, 0.0)
    return (
        {
            "maximum_depth_cm": float(np.max(architecture.nodes[:, 1])),
            "horizontal_spread_cm": base["lateral_width"],
            "total_root_length_cm": base["total_length"],
            "branch_count": int(base["branch_count"]),
            "mean_tortuosity": base["tortuosity"],
            "convex_hull_area_cm2": base["convex_hull_area"],
            "mechanical_arrest_proxy": _mechanical_arrest_proxy(architecture, soil),
            "hydraulic_delivery_rate": delivery_rate,
            "cumulative_hydraulic_index": cumulative_delivery,
            "construction_cost": carbon_cost,
            "construction_remaining": carbon_remaining,
            "construction_balance_residual": abs(initial_carbon - carbon_cost - carbon_remaining),
            "construction_budget_exceeded": int(carbon_cost > initial_carbon),
            "kirchhoff_residual": solution.relative_kirchhoff_residual,
            "segment_count": len(architecture.segments),
            "allocated_tip_count": len(architecture.tip_paths),
            "tip_allocation_reached": int(
                len(architecture.tip_paths) >= int(settings["max_tips"])
            ),
            "boundary_contact_count": _boundary_contact_count(architecture, soil),
        },
        density,
        bins,
    )


def run_experiment(
    architecture_name: str,
    soil_name: str,
    *,
    seed: int = 20260802,
    replicate: int = 0,
    duration_days: float = 5.5,
    dt_days: float = 0.04,
    max_tips: int = 120,
) -> ExperimentResult:
    """Run one controlled synthetic experiment.

    The same ``seed``, ``replicate`` and inputs produce byte-identical segment
    coordinates.  Environment seeds depend only on soil and replicate, so two
    architecture presets can be compared in a paired soil realization.
    """
    if seed < 0 or replicate < 0:
        raise ValueError("seed and replicate must be non-negative")
    if not 0.25 <= duration_days <= MAX_EXPLORER_DURATION_DAYS:
        raise ValueError(
            f"duration_days must lie between 0.25 and {MAX_EXPLORER_DURATION_DAYS:g}"
        )
    if dt_days not in {0.02, 0.04, 0.08}:
        raise ValueError("dt_days must be one of 0.02, 0.04, or 0.08")
    if not 10 <= max_tips <= 160:
        raise ValueError("max_tips must lie between 10 and 160")

    config = load_default_config()
    if architecture_name not in config["root_regimes"]:
        raise KeyError(f"unknown architecture preset: {architecture_name}")
    if soil_name not in config["soils"]:
        raise KeyError(f"unknown soil constructor: {soil_name}")
    config["seed"] = int(seed)
    settings = config["simulation"]
    settings["duration_days"] = float(duration_days)
    settings["dt_days"] = float(dt_days)
    settings["max_tips"] = int(max_tips)
    domain_scale = _configure_explorer_domain(settings, duration_days)

    architecture_index = list(config["root_regimes"]).index(architecture_name)
    soil_index = list(config["soils"]).index(soil_name)
    environment_seed = _seed_value(seed, 1, soil_index, replicate)
    growth_seed = _seed_value(seed, 2, architecture_index, soil_index, replicate)
    grid = _build_grid(config)
    soil = _soil_from_config(
        soil_name,
        grid,
        np.random.default_rng(environment_seed),
    )
    response = EmergenceResponse(
        **{key: float(value) for key, value in config["emergence"].items()}
    )
    architecture = simulate_architecture(
        soil=soil,
        root_types=_root_types(
            config["root_regimes"][architecture_name],
            config["common_traits"],
        ),
        primary_type="primary",
        response=response,
        duration=float(duration_days),
        dt=float(dt_days),
        rng=np.random.default_rng(growth_seed),
        max_order=int(settings["max_order"]),
        max_tips=int(max_tips),
        carbon_factor=1.0,
    )
    metrics, density, bins = _metrics(architecture, soil, config)
    return ExperimentResult(
        architecture_name=architecture_name,
        soil_name=soil_name,
        replicate=int(replicate),
        architecture=architecture,
        soil=soil,
        metrics=metrics,
        root_length_density=density,
        depth_bins=bins,
        seeds={"environment_seed": environment_seed, "growth_seed": growth_seed},
        settings={
            "seed": int(seed),
            "duration_days": float(duration_days),
            "dt_days": float(dt_days),
            "max_tips": int(max_tips),
            "domain_scale": float(domain_scale),
            "domain_x_cm": [float(value) for value in settings["domain_x_cm"]],
            "domain_z_cm": [float(value) for value in settings["domain_z_cm"]],
            "grid_cells": int(settings["grid_cells"]),
        },
    )


def segment_frame(result: ExperimentResult) -> pd.DataFrame:
    """Return one row per simulated segment."""
    return pd.DataFrame(
        [
            {
                "segment_id": segment.segment_id,
                "parent_segment_id": segment.parent_segment_id,
                "tip_id": segment.tip_id,
                "root_type": segment.root_type,
                "order": segment.order,
                "start_x_cm": segment.start[0],
                "start_depth_cm": segment.start[1],
                "end_x_cm": segment.end[0],
                "end_depth_cm": segment.end[1],
                "radius_cm": segment.radius,
                "created_day": segment.created_time,
            }
            for segment in result.architecture.segments
        ]
    )


def result_signature(result: ExperimentResult) -> str:
    """Hash the segment table for an exact reproducibility check."""
    payload = segment_frame(result).to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def result_archive(result: ExperimentResult) -> bytes:
    """Create an in-memory, analysis-friendly ZIP export."""
    metrics = {
        "architecture": result.architecture_name,
        "soil": result.soil_name,
        "replicate": result.replicate,
        "seeds": result.seeds,
        "settings": result.settings,
        "metrics": result.metrics,
        "scope": "synthetic and uncalibrated",
    }
    depth_midpoints = 0.5 * (result.depth_bins[:-1] + result.depth_bins[1:])
    rld = pd.DataFrame(
        {
            "depth_midpoint_cm": depth_midpoints,
            "root_length_density_cm_per_cm": result.root_length_density,
        }
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metrics.json", json.dumps(metrics, indent=2) + "\n")
        archive.writestr(
            "segments.csv",
            segment_frame(result).to_csv(index=False, lineterminator="\n"),
        )
        archive.writestr(
            "root_length_density.csv",
            rld.to_csv(index=False, lineterminator="\n"),
        )
        archive.writestr(
            "README.txt",
            "ROOT-FPT Explorer export\nSynthetic and uncalibrated model output.\n",
        )
    return buffer.getvalue()


def relative_change(coarse: float, fine: float, floor: float = 1e-12) -> float:
    """Absolute symmetric relative change as a percentage."""
    scale = max(abs(coarse), abs(fine), floor)
    return 100.0 * abs(fine - coarse) / scale
