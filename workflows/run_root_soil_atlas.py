#!/usr/bin/env python3
"""Run the controlled ROOT-FPT synthetic root-soil visual atlas."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rootfpt.multiscale.agent import TipTraits
from rootfpt.multiscale.architecture import (
    Architecture,
    EmergenceResponse,
    RootType,
    SiteStatus,
    simulate_architecture,
)
from rootfpt.multiscale.soil import Grid2D, SoilState
from rootfpt.multiscale.water import (
    ReducedWaterParameters,
    ReducedWaterState,
    hydraulic_architecture_solution,
    step_reduced_water,
)

CONFIG = ROOT / "configs" / "atlas" / "root_soil_atlas.yaml"
RESULT_DIR = ROOT / "results" / "atlas"
RAW_DIR = RESULT_DIR / "raw"
TABLE_DIR = RESULT_DIR / "tables"
REPRESENTATIVE_DIR = RAW_DIR / "representatives"
REPORT_DIR = ROOT / "reports" / "atlas"


def prepare() -> None:
    for directory in (RAW_DIR, TABLE_DIR, REPRESENTATIVE_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_grid(config: dict) -> Grid2D:
    settings = config["simulation"]
    cells = int(settings["grid_cells"])
    return Grid2D(
        cells,
        cells,
        tuple(settings["domain_x_cm"]),
        tuple(settings["domain_z_cm"]),
    )


def seed_value(master: int, *indices: int) -> int:
    return int(np.random.SeedSequence([master, *indices]).generate_state(1)[0])


def soil_from_config(
    name: str,
    grid: Grid2D,
    rng: np.random.Generator,
) -> SoilState:
    """Construct a controlled soil without changing the growth equations."""
    if name == "homogeneous":
        return SoilState.homogeneous(
            grid, water=0.32, impedance=0.28, nutrient=0.75
        )
    if name == "patchy_matern":
        return SoilState.matern(
            grid,
            rng=rng,
            correlation_length=1.25,
            smoothness=1.5,
            cross_correlation=-0.55,
        )
    base = SoilState.homogeneous(
        grid, water=0.32, impedance=0.25, nutrient=0.75
    )
    if name == "layered":
        return base.layered(
            interface_depth=3.5,
            lower_impedance=1.9,
            lower_water=0.20,
        )
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
        horizontal_jitter = float(rng.normal(0.0, 0.25))
        return base.with_compacted_lens(
            centre=(horizontal_jitter, 4.6),
            radii=(2.5, 0.95),
            impedance=2.45,
        )
    if name == "cracked":
        angle_jitter = float(rng.normal(0.0, 0.035))
        first = base.with_crack(
            angle=1.18 + angle_jitter,
            offset=0.0,
            width=0.38,
            strength=8.0,
        )
        return first.with_crack(
            angle=1.92 - angle_jitter,
            offset=0.85,
            width=0.30,
            strength=6.0,
        )
    raise KeyError(name)


def tip_traits(values: dict, common: dict) -> TipTraits:
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


def root_types_from_config(regime: dict, common: dict) -> dict[str, RootType]:
    def root_type(name: str, values: dict, successor: str | None) -> RootType:
        return RootType(
            name,
            tip_traits(values, common),
            float(values["radius"]),
            float(values["branch_spacing"]),
            float(values["branch_spacing_shape"]),
            float(values["delay"]),
            float(values["branch_angle"]),
            float(values["branch_angle_sd"]),
            mortality_rate=float(values["mortality_rate"]),
            successor=successor,
        )

    primary = root_type("primary", regime["primary"], "lateral")
    lateral = root_type("lateral", regime["lateral"], "second")
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
    second = root_type("second", second_values, None)
    return {"primary": primary, "lateral": lateral, "second": second}


def emergence_from_config(values: dict) -> EmergenceResponse:
    return EmergenceResponse(**{key: float(value) for key, value in values.items()})


def rasterize_architecture(
    architecture: Architecture,
    grid: Grid2D,
) -> np.ndarray:
    density = np.zeros((grid.nz, grid.nx))
    spacing = 0.5 * min(grid.dx, grid.dz)
    for segment in architecture.segments:
        samples = max(2, int(np.ceil(segment.length / spacing)))
        fraction = (np.arange(samples) + 0.5) / samples
        start = np.asarray(segment.start)
        end = np.asarray(segment.end)
        points = start[None, :] + fraction[:, None] * (end - start)[None, :]
        ix = np.floor((points[:, 0] - grid.x_limits[0]) / grid.dx).astype(int)
        iz = np.floor((points[:, 1] - grid.z_limits[0]) / grid.dz).astype(int)
        valid = (ix >= 0) & (ix < grid.nx) & (iz >= 0) & (iz < grid.nz)
        np.add.at(
            density,
            (iz[valid], ix[valid]),
            segment.length / samples / grid.cell_area,
        )
    return density


def mechanical_arrest_proxy(architecture: Architecture, soil: SoilState) -> float:
    """Fraction of high-impedance potential sites that did not emerge."""
    if not architecture.sites:
        return 0.0
    positions = np.asarray([site.position for site in architecture.sites])
    impedance = soil.sample("impedance", positions)
    high = impedance >= 1.2
    if not np.any(high):
        return 0.0
    failed = np.asarray(
        [site.status in {SiteStatus.DORMANT, SiteStatus.ABORTED} for site in architecture.sites]
    )
    return float(np.mean(failed[high]))


def architecture_metrics(
    architecture: Architecture,
    soil: SoilState,
    config: dict,
) -> tuple[dict[str, float | int], np.ndarray]:
    settings = config["simulation"]
    base = architecture.metrics()
    depth_bins = np.asarray(settings["root_length_depth_bins_cm"], dtype=float)
    rld = architecture.root_length_density_by_depth(depth_bins)
    types = config["hydraulics"]
    solution = hydraulic_architecture_solution(
        architecture,
        soil,
        collar_pressure_head=float(types["collar_pressure_head_cm"]),
        axial_conductivity_by_type={
            key: float(value) for key, value in types["axial_conductivity"].items()
        },
        radial_conductivity_by_type={
            key: float(value) for key, value in types["radial_conductivity"].items()
        },
    )
    delivered_rate = max(float(solution.collar_delivered_flow), 0.0)
    cumulative = delivered_rate * float(settings["duration_days"])
    carbon_spent = (
        float(settings["carbon_length_cost"]) * base["total_length"]
        + float(settings["carbon_branch_cost"]) * base["branch_count"]
    )
    initial_carbon = float(settings["initial_carbon"])
    carbon_remaining = max(initial_carbon - carbon_spent, 0.0)
    carbon_balance_residual = abs(
        initial_carbon - carbon_spent - carbon_remaining
    )
    last_growth = max(
        (segment.created_time for segment in architecture.segments),
        default=0.0,
    )
    return (
        {
            "maximum_depth": float(np.max(architecture.nodes[:, 1])),
            "horizontal_spread": base["lateral_width"],
            "total_root_length": base["total_length"],
            "branch_count": int(base["branch_count"]),
            "order_0_segments": int(
                sum(segment.order == 0 for segment in architecture.segments)
            ),
            "order_1_segments": int(
                sum(segment.order == 1 for segment in architecture.segments)
            ),
            "order_2_segments": int(
                sum(segment.order >= 2 for segment in architecture.segments)
            ),
            "mean_tortuosity": base["tortuosity"],
            "convex_hull_area": base["convex_hull_area"],
            "mechanical_arrest_proxy": mechanical_arrest_proxy(architecture, soil),
            "hydraulic_delivery_rate": delivered_rate,
            "cumulative_collar_water": cumulative,
            "root_carbon_cost": carbon_spent,
            "water_per_carbon": cumulative / max(carbon_spent, 1e-12),
            "carbon_remaining": carbon_remaining,
            "carbon_budget_exceeded": int(carbon_spent > initial_carbon),
            "carbon_balance_residual": carbon_balance_residual,
            "kirchhoff_residual": solution.relative_kirchhoff_residual,
            "growth_arrested": int(
                last_growth
                < float(settings["duration_days"]) - 2.5 * float(settings["dt_days"])
            ),
            "active_tip_proxy_count": int(
                len(
                    {
                        segment.tip_id
                        for segment in architecture.segments
                        if segment.created_time
                        >= float(settings["duration_days"])
                        - 1.5 * float(settings["dt_days"])
                    }
                )
            ),
        },
        rld,
    )


def simulate_one(
    architecture_name: str,
    soil_name: str,
    replicate: int,
    config: dict,
) -> tuple[Architecture, SoilState, dict[str, float | int], np.ndarray, dict[str, int]]:
    master = int(config["seed"])
    architecture_names = list(config["root_regimes"])
    soil_names = list(config["soils"])
    architecture_index = architecture_names.index(architecture_name)
    soil_index = soil_names.index(soil_name)
    environment_seed = seed_value(master, 1, soil_index, replicate)
    growth_seed = seed_value(master, 2, architecture_index, soil_index, replicate)
    grid = build_grid(config)
    soil = soil_from_config(
        soil_name,
        grid,
        np.random.default_rng(environment_seed),
    )
    root_types = root_types_from_config(
        config["root_regimes"][architecture_name],
        config["common_traits"],
    )
    settings = config["simulation"]
    architecture = simulate_architecture(
        soil=soil,
        root_types=root_types,
        primary_type="primary",
        response=emergence_from_config(config["emergence"]),
        duration=float(settings["duration_days"]),
        dt=float(settings["dt_days"]),
        rng=np.random.default_rng(growth_seed),
        max_order=int(settings["max_order"]),
        max_tips=int(settings["max_tips"]),
        carbon_factor=1.0,
    )
    metrics, rld = architecture_metrics(architecture, soil, config)
    return (
        architecture,
        soil,
        metrics,
        rld,
        {"environment_seed": environment_seed, "growth_seed": growth_seed},
    )


def failure_row(
    architecture_name: str,
    soil_name: str,
    replicate: int,
    error: Exception,
    config: dict,
) -> dict[str, object]:
    master = int(config["seed"])
    architecture_index = list(config["root_regimes"]).index(architecture_name)
    soil_index = list(config["soils"]).index(soil_name)
    return {
        "architecture": architecture_name,
        "soil": soil_name,
        "replicate": replicate,
        "environment_seed": seed_value(master, 1, soil_index, replicate),
        "growth_seed": seed_value(
            master, 2, architecture_index, soil_index, replicate
        ),
        "failed": 1,
        "failure_reason": f"{type(error).__name__}: {error}",
    }


def choose_medoids(
    metrics: pd.DataFrame,
    rld: pd.DataFrame,
) -> pd.DataFrame:
    rld_wide = rld.pivot_table(
        index=["architecture", "soil", "replicate"],
        columns="depth_bin",
        values="root_length_density",
    ).add_prefix("rld_")
    merged = metrics.merge(
        rld_wide.reset_index(),
        on=["architecture", "soil", "replicate"],
        how="left",
    )
    feature_columns = [
        "maximum_depth",
        "horizontal_spread",
        "total_root_length",
        "branch_count",
        "mean_tortuosity",
        "cumulative_collar_water",
        *[column for column in merged if column.startswith("rld_")],
    ]
    rows = []
    for (architecture, soil), values in merged.groupby(["architecture", "soil"]):
        eligible = values[values["failed"] == 0].dropna(subset=feature_columns)
        if eligible.empty:
            rows.append(
                {
                    "architecture": architecture,
                    "soil": soil,
                    "replicate": -1,
                    "medoid_distance": np.nan,
                }
            )
            continue
        matrix = eligible[feature_columns].to_numpy(float)
        target = np.median(matrix, axis=0)
        scale = np.median(np.abs(matrix - target), axis=0)
        fallback = np.std(matrix, axis=0)
        scale = np.where(scale > 1e-12, scale, fallback)
        scale = np.where(scale > 1e-12, scale, 1.0)
        distance = np.linalg.norm((matrix - target) / scale, axis=1)
        minimum = float(np.min(distance))
        candidates = eligible.iloc[np.flatnonzero(np.isclose(distance, minimum))]
        selected = candidates.sort_values("replicate").iloc[0]
        rows.append(
            {
                "architecture": architecture,
                "soil": soil,
                "replicate": int(selected["replicate"]),
                "medoid_distance": minimum,
            }
        )
    return pd.DataFrame(rows)


def water_assay(
    architecture: Architecture,
    soil: SoilState,
    config: dict,
) -> tuple[np.ndarray, dict[str, float]]:
    settings = config["water_assay"]
    density = rasterize_architecture(architecture, soil.grid)
    state = ReducedWaterState(soil.grid, soil.water.copy())
    parameters = ReducedWaterParameters(
        float(settings["redistribution_diffusivity"]),
        float(settings["residual_water"]),
        float(settings["saturation_water"]),
        float(settings["uptake_coefficient"]),
    )
    steps = round(float(settings["duration_days"]) / float(settings["dt_days"]))
    for _ in range(steps):
        step_reduced_water(
            state,
            parameters,
            root_length_density=density,
            dt=float(settings["dt_days"]),
            infiltration=float(settings["infiltration_per_day"]),
            evaporation=float(settings["evaporation_per_day"]),
        )
    return state.water, {
        "dynamic_root_water_gain": state.cumulative_root_gain,
        "maximum_water_balance_residual": state.maximum_balance_residual,
    }


def store_representative(
    architecture_name: str,
    soil_name: str,
    replicate: int,
    architecture: Architecture,
    soil: SoilState,
    config: dict,
) -> tuple[list[dict], dict]:
    hydraulic = config["hydraulics"]
    solution = hydraulic_architecture_solution(
        architecture,
        soil,
        collar_pressure_head=float(hydraulic["collar_pressure_head_cm"]),
        axial_conductivity_by_type={
            key: float(value)
            for key, value in hydraulic["axial_conductivity"].items()
        },
        radial_conductivity_by_type={
            key: float(value)
            for key, value in hydraulic["radial_conductivity"].items()
        },
    )
    segment_rows = []
    for segment, axial, radial in zip(
        architecture.segments,
        solution.axial_flows,
        solution.radial_flows,
        strict=True,
    ):
        segment_rows.append(
            {
                "architecture": architecture_name,
                "soil": soil_name,
                "replicate": replicate,
                "segment_id": segment.segment_id,
                "start_node": segment.start_node,
                "end_node": segment.end_node,
                "tip_id": segment.tip_id,
                "root_type": segment.root_type,
                "order": segment.order,
                "start_x": segment.start[0],
                "start_z": segment.start[1],
                "end_x": segment.end[0],
                "end_z": segment.end[1],
                "radius": segment.radius,
                "created_time": segment.created_time,
                "axial_flow": float(axial),
                "radial_flow": float(radial),
            }
        )
    final_water, water_metrics = water_assay(architecture, soil, config)
    key = f"{architecture_name}__{soil_name}"
    anisotropy = np.sqrt(
        (soil.anisotropy_xx - 1.0) ** 2
        + 2.0 * soil.anisotropy_xz**2
        + (soil.anisotropy_zz - 1.0) ** 2
    )
    np.savez_compressed(
        REPRESENTATIVE_DIR / f"{key}.npz",
        initial_water=soil.water,
        final_water=final_water,
        impedance=soil.impedance,
        anisotropy=anisotropy,
    )
    return segment_rows, {
        "architecture": architecture_name,
        "soil": soil_name,
        "replicate": replicate,
        **water_metrics,
    }


def write_reports(
    config: dict,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    medoids: pd.DataFrame,
    water_summary: pd.DataFrame,
) -> None:
    settings = config["simulation"]
    failures = int(metrics["failed"].sum())
    exceeded = int(metrics["carbon_budget_exceeded"].fillna(0).sum())
    max_kirchhoff = float(metrics["kirchhoff_residual"].max())
    max_water = float(water_summary["maximum_water_balance_residual"].max())
    (REPORT_DIR / "atlas_methods.md").write_text(
        f"""# ROOT-FPT visual root-soil atlas: methods

All systems are **synthetic and uncalibrated**. Six trait presets and six soil
constructors use one unchanged `simulate_architecture` equation. The presets
change only documented speed, turning, tropism, sensing, radius, spacing,
delay, angle and mortality parameters.

Each of the 36 root-by-soil conditions contains
{settings['replicates']} independent growth realizations in the common
{settings['domain_x_cm']} by {settings['domain_z_cm']} cm domain for
{settings['duration_days']} days at `{settings['dt_days']}` day resolution.
The initial carbon accounting budget is {settings['initial_carbon']} for every
run. Environment seeds depend only on soil and replicate, so the exact soil
realization is paired across all six root regimes. Growth seeds are separate
and recorded. Failed or visually unattractive realizations are never removed.

The displayed realization is selected automatically within each condition.
The medoid minimizes robustly standardized Euclidean distance to the component
wise multivariate median of maximum depth, spread, length, branch count,
tortuosity, depth-binned root-length density and cumulative hydraulic
delivery. No image is inspected during selection.

Hydraulic delivery uses the existing sparse graph solver. Cumulative delivery
is the final-architecture collar rate multiplied by the controlled assay
duration; it is a synthetic functional index. Dynamic depletion figures use
the conservative reduced-water solver on the selected medoid architecture.

`mechanical_arrest_proxy` is not a discrete axis-arrest event. It is the
fraction of potential lateral sites in soil with impedance at least 1.2 MPa
that ended dormant or aborted. ROOT-FPT currently represents axial mechanical
effects mainly through continuous growth slowing, so this proxy is labelled
explicitly.

Carbon is a common accounting control, not an active feedback in this atlas.
`carbon_budget_exceeded` is reported rather than silently truncating roots.
Water per carbon divides cumulative collar delivery by accounted construction
and branch-initiation cost.
"""
    )
    strongest = (
        summary.groupby("architecture")["maximum_depth_mean"].agg(["min", "max"])
    )
    largest_span = float((strongest["max"] - strongest["min"]).max())
    across_regimes = summary.groupby("soil")["maximum_depth_mean"].agg(["min", "max"])
    largest_regime_span = float(
        (across_regimes["max"] - across_regimes["min"]).max()
    )
    branch_across_regimes = summary.groupby("soil")["branch_count_mean"].agg(
        ["min", "max"]
    )
    largest_branch_span = float(
        (branch_across_regimes["max"] - branch_across_regimes["min"]).max()
    )
    (REPORT_DIR / "atlas_results.md").write_text(
        f"""# ROOT-FPT visual root-soil atlas: results

The frozen atlas contains {len(metrics)} realizations across
{metrics['architecture'].nunique()} synthetic root regimes and
{metrics['soil'].nunique()} controlled soils. It includes {len(medoids)}
algorithmically selected medoids.

- Simulation failures retained in the raw table: {failures}.
- Carbon-accounting budget exceedances: {exceeded}.
- Maximum hydraulic Kirchhoff residual: {max_kirchhoff:.3e}.
- Maximum medoid water-balance residual: {max_water:.3e}.
- Largest across-soil span in architecture-level mean maximum depth:
  {largest_span:.3f} cm.
- Largest same-soil span in mean maximum depth across trait regimes:
  {largest_regime_span:.3f} cm.
- Largest same-soil span in mean branch count across trait regimes:
  {largest_branch_span:.3f}.

The atlas demonstrates software capability and controlled sensitivity, not
species prediction or agronomic superiority. Morphology and function are
reported separately; visual extent is not interpreted as performance.
"""
    )


def main() -> None:
    prepare()
    config = yaml.safe_load(CONFIG.read_text())
    settings = config["simulation"]
    metric_rows: list[dict] = []
    rld_rows: list[dict] = []
    depth_bins = np.asarray(settings["root_length_depth_bins_cm"], dtype=float)
    for soil_name in config["soils"]:
        for architecture_name in config["root_regimes"]:
            failures = 0
            for replicate in range(int(settings["replicates"])):
                try:
                    _, _, metrics, rld, seeds = simulate_one(
                        architecture_name,
                        soil_name,
                        replicate,
                        config,
                    )
                    metric_rows.append(
                        {
                            "architecture": architecture_name,
                            "soil": soil_name,
                            "replicate": replicate,
                            **seeds,
                            "failed": 0,
                            "failure_reason": "",
                            **metrics,
                        }
                    )
                    rld_rows.extend(
                        {
                            "architecture": architecture_name,
                            "soil": soil_name,
                            "replicate": replicate,
                            "depth_bin": index,
                            "depth_midpoint": 0.5
                            * (depth_bins[index] + depth_bins[index + 1]),
                            "root_length_density": float(value),
                        }
                        for index, value in enumerate(rld)
                    )
                except Exception as error:  # retained as declared failed runs
                    failures += 1
                    metric_rows.append(
                        failure_row(
                            architecture_name,
                            soil_name,
                            replicate,
                            error,
                            config,
                        )
                    )
            print(
                json.dumps(
                    {
                        "architecture": architecture_name,
                        "soil": soil_name,
                        "replicates": settings["replicates"],
                        "failures": failures,
                    }
                ),
                flush=True,
            )
    metrics = pd.DataFrame(metric_rows)
    rld = pd.DataFrame(rld_rows)
    metrics.to_csv(RAW_DIR / "replicate_metrics.csv", index=False)
    rld.to_csv(RAW_DIR / "replicate_root_length_density.csv", index=False)
    paired_seed_audit = (
        metrics.groupby(["soil", "replicate"])["environment_seed"]
        .nunique()
        .rename("unique_environment_seeds_across_roots")
        .reset_index()
    )
    paired_seed_audit["paired_seed_pass"] = (
        paired_seed_audit["unique_environment_seeds_across_roots"] == 1
    )
    paired_seed_audit.to_csv(TABLE_DIR / "paired_seed_audit.csv", index=False)

    medoids = choose_medoids(metrics, rld)
    medoids.to_csv(TABLE_DIR / "medoid_selection.csv", index=False)
    segment_rows = []
    water_rows = []
    for row in medoids.itertuples(index=False):
        if row.replicate < 0:
            continue
        architecture, soil, _, _, _ = simulate_one(
            row.architecture,
            row.soil,
            int(row.replicate),
            config,
        )
        segments, water = store_representative(
            row.architecture,
            row.soil,
            int(row.replicate),
            architecture,
            soil,
            config,
        )
        segment_rows.extend(segments)
        water_rows.append(water)
    pd.DataFrame(segment_rows).to_csv(
        RAW_DIR / "representative_segments.csv", index=False
    )
    water_summary = pd.DataFrame(water_rows)
    water_summary.to_csv(TABLE_DIR / "representative_water_balance.csv", index=False)

    numeric = [
        "maximum_depth",
        "horizontal_spread",
        "total_root_length",
        "branch_count",
        "order_0_segments",
        "order_1_segments",
        "order_2_segments",
        "mean_tortuosity",
        "convex_hull_area",
        "mechanical_arrest_proxy",
        "cumulative_collar_water",
        "water_per_carbon",
        "kirchhoff_residual",
        "carbon_balance_residual",
    ]
    valid = metrics[metrics["failed"] == 0]
    summary = valid.groupby(["architecture", "soil"])[numeric].agg(
        ["mean", "std", "median", "min", "max"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    counts = metrics.groupby(["architecture", "soil"]).agg(
        replicates=("replicate", "size"),
        failures=("failed", "sum"),
        growth_arrest_frequency=("growth_arrested", "mean"),
        carbon_budget_exceedance_frequency=("carbon_budget_exceeded", "mean"),
    )
    summary = summary.merge(counts.reset_index(), on=["architecture", "soil"])
    summary = summary.merge(
        water_summary,
        on=["architecture", "soil"],
        how="left",
        suffixes=("", "_medoid"),
    )
    summary.to_csv(TABLE_DIR / "condition_summary.csv", index=False)

    order_counts = valid.groupby(["architecture", "soil"])[
        ["order_0_segments", "order_1_segments", "order_2_segments"]
    ].agg(["mean", "std"])
    order_counts.columns = ["_".join(column) for column in order_counts.columns]
    order_counts.reset_index().to_csv(
        TABLE_DIR / "branch_order_counts.csv", index=False
    )
    rld.groupby(["architecture", "soil", "depth_bin", "depth_midpoint"])[
        "root_length_density"
    ].agg(["mean", "std", "median"]).reset_index().to_csv(
        TABLE_DIR / "root_length_density_by_depth.csv", index=False
    )
    metrics[metrics["failed"] == 1][
        [
            "architecture",
            "soil",
            "replicate",
            "environment_seed",
            "growth_seed",
            "failure_reason",
        ]
    ].to_csv(TABLE_DIR / "failures.csv", index=False)
    write_reports(config, metrics, summary, medoids, water_summary)
    manifest = {
        "title": config["title"],
        "synthetic_uncalibrated": True,
        "configuration": str(CONFIG.relative_to(ROOT)),
        "configuration_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "master_seed": int(config["seed"]),
        "replicates": len(metrics),
        "conditions": len(summary),
        "medoids": len(medoids),
        "failures_retained": int(metrics["failed"].sum()),
        "environment_pairing": "seed depends only on soil and replicate",
        "simulator": "rootfpt.multiscale.architecture.simulate_architecture",
        "manual_visual_selection": False,
    }
    (RAW_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "replicates": len(metrics),
                "conditions": len(summary),
                "medoids": len(medoids),
                "failures": int(metrics["failed"].sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
