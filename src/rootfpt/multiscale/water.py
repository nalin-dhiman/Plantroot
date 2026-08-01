"""Conservative reduced soil-water coupling and hydraulic benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rootfpt.hydraulics import (
    HydraulicSegment,
    HydraulicSolution,
    solve_hydraulic_network,
)
from rootfpt.multiscale.architecture import Architecture
from rootfpt.multiscale.soil import Grid2D, SoilState


@dataclass(frozen=True)
class ReducedWaterParameters:
    redistribution_diffusivity: float
    residual_water: float
    saturation_water: float
    uptake_coefficient: float

    def __post_init__(self) -> None:
        if self.redistribution_diffusivity < 0 or self.uptake_coefficient < 0:
            raise ValueError("water rates must be nonnegative")
        if not 0 <= self.residual_water < self.saturation_water <= 1:
            raise ValueError("invalid residual/saturation water contents")


@dataclass
class ReducedWaterState:
    grid: Grid2D
    water: np.ndarray
    cumulative_root_gain: float = 0.0
    cumulative_infiltration: float = 0.0
    cumulative_evaporation: float = 0.0
    cumulative_drainage: float = 0.0
    maximum_balance_residual: float = 0.0
    time: float = 0.0

    @property
    def storage(self) -> float:
        return float(np.sum(self.water) * self.grid.cell_area)


def step_reduced_water(
    state: ReducedWaterState,
    parameters: ReducedWaterParameters,
    *,
    root_length_density: np.ndarray,
    dt: float,
    infiltration: float = 0.0,
    evaporation: float = 0.0,
) -> dict[str, float]:
    """Advance a no-flux grid water balance.

    Water content is volumetric; the two-dimensional domain has unit
    out-of-plane thickness. ``infiltration`` and ``evaporation`` are boundary
    depths per day.
    """
    if dt <= 0 or root_length_density.shape != state.water.shape:
        raise ValueError("invalid water update")
    before = state.storage
    water = state.water.copy()
    diffusivity = parameters.redistribution_diffusivity
    flux_x = -diffusivity * np.diff(water, axis=1) / state.grid.dx
    flux_z = -diffusivity * np.diff(water, axis=0) / state.grid.dz
    divergence = np.zeros_like(water)
    divergence[:, :-1] -= flux_x / state.grid.dx
    divergence[:, 1:] += flux_x / state.grid.dx
    divergence[:-1, :] -= flux_z / state.grid.dz
    divergence[1:, :] += flux_z / state.grid.dz
    water += dt * divergence

    infiltration_volume = (
        max(0.0, infiltration) * dt * (state.grid.x_limits[1] - state.grid.x_limits[0])
    )
    evaporation_demand = max(0.0, evaporation) * dt / state.grid.dz
    infiltration_increment = max(0.0, infiltration) * dt / state.grid.dz
    water[0] += infiltration_increment
    available_top = np.maximum(water[0] - parameters.residual_water, 0.0)
    evaporation_removed = np.minimum(available_top, evaporation_demand)
    water[0] -= evaporation_removed
    evaporation_volume = float(np.sum(evaporation_removed) * state.grid.cell_area)

    availability = np.clip(
        (water - parameters.residual_water)
        / (parameters.saturation_water - parameters.residual_water),
        0.0,
        1.0,
    )
    uptake_demand = (
        parameters.uptake_coefficient * np.maximum(root_length_density, 0.0) * availability * dt
    )
    uptake = np.minimum(uptake_demand, np.maximum(water - parameters.residual_water, 0.0))
    water -= uptake
    root_gain = float(np.sum(uptake) * state.grid.cell_area)

    excess = np.maximum(water - parameters.saturation_water, 0.0)
    drainage = float(np.sum(excess) * state.grid.cell_area)
    water -= excess
    water = np.maximum(water, parameters.residual_water)
    state.water = water
    state.cumulative_root_gain += root_gain
    state.cumulative_infiltration += infiltration_volume
    state.cumulative_evaporation += evaporation_volume
    state.cumulative_drainage += drainage
    state.time += dt
    after = state.storage
    residual = after - before - infiltration_volume + evaporation_volume + root_gain + drainage
    state.maximum_balance_residual = max(state.maximum_balance_residual, abs(residual))
    return {
        "storage": after,
        "root_gain": root_gain,
        "infiltration": infiltration_volume,
        "evaporation": evaporation_volume,
        "drainage": drainage,
        "balance_residual": residual,
    }


def hydraulic_architecture_solution(
    architecture: Architecture,
    soil: SoilState,
    *,
    collar_pressure_head: float,
    axial_conductivity_by_type: dict[str, float],
    radial_conductivity_by_type: dict[str, float],
) -> HydraulicSolution:
    """Map deposited architecture segments to the validated hydraulic graph."""
    edges = []
    for segment in architecture.segments:
        midpoint = 0.5 * (np.asarray(segment.start) + np.asarray(segment.end))
        soil_pressure = float(soil.sample("pressure_head", midpoint[None, :])[0])
        total_soil_head = soil_pressure - midpoint[1]
        axial = axial_conductivity_by_type[segment.root_type]
        radial = (
            2.0
            * np.pi
            * segment.radius
            * segment.length
            * radial_conductivity_by_type[segment.root_type]
        )
        edges.append(
            HydraulicSegment(
                segment.start_node,
                segment.end_node,
                segment.length,
                axial,
                radial,
                total_soil_head,
            )
        )
    return solve_hydraulic_network(
        edges,
        collar_node=0,
        collar_potential=collar_pressure_head - architecture.nodes[0, 1],
    )


def geometric_potential_uptake(
    architecture: Architecture,
    soil: SoilState,
    *,
    collar_pressure_head: float,
    radial_conductivity_by_type: dict[str, float],
) -> float:
    """Independent-segment uptake that ignores axial hydraulic limitation."""
    potential = 0.0
    for segment in architecture.segments:
        midpoint = 0.5 * (np.asarray(segment.start) + np.asarray(segment.end))
        soil_pressure = float(soil.sample("pressure_head", midpoint[None, :])[0])
        radial = (
            2.0
            * np.pi
            * segment.radius
            * segment.length
            * radial_conductivity_by_type[segment.root_type]
        )
        potential += radial * max(soil_pressure - collar_pressure_head, 0.0)
    return potential


def m31_analytical_pressure(
    depth: np.ndarray,
    *,
    length: float = 50.0,
    radius: float = 0.2,
    axial_conductivity: float = 0.0432,
    radial_conductivity: float = 1.73e-4,
    soil_pressure: float = -200.0,
    collar_pressure: float = -1000.0,
) -> np.ndarray:
    """Published collaborative benchmark M3.1 analytical solution."""
    depth = np.asarray(depth, dtype=float)
    coefficient = 2.0 * np.pi * radius * radial_conductivity / axial_conductivity
    root = np.sqrt(coefficient)
    coordinate_tip = -length
    matrix = np.array(
        [
            [1.0, 1.0],
            [
                root * np.exp(root * coordinate_tip),
                -root * np.exp(-root * coordinate_tip),
            ],
        ]
    )
    constants = np.linalg.solve(matrix, np.array([collar_pressure - soil_pressure, -1.0]))
    coordinate = -depth
    return (
        soil_pressure
        + constants[0] * np.exp(root * coordinate)
        + constants[1] * np.exp(-root * coordinate)
    )


def run_m31_hydraulic_benchmark(segments: int) -> dict[str, float | np.ndarray]:
    """Discretize benchmark M3.1 with total hydraulic head."""
    if segments < 2:
        raise ValueError("benchmark requires at least two segments")
    length = 50.0
    radius = 0.2
    axial = 0.0432
    radial = 1.73e-4
    soil_pressure = -200.0
    collar_pressure = -1000.0
    segment_length = length / segments
    edges = []
    for index in range(segments):
        midpoint_depth = (index + 0.5) * segment_length
        edges.append(
            HydraulicSegment(
                index,
                index + 1,
                segment_length,
                axial,
                2.0 * np.pi * radius * radial * segment_length,
                soil_pressure - midpoint_depth,
            )
        )
    solution = solve_hydraulic_network(
        edges,
        collar_node=0,
        collar_potential=collar_pressure,
    )
    depth = np.linspace(0.0, length, segments + 1)
    numerical_pressure = solution.potentials + depth
    analytical_pressure = m31_analytical_pressure(depth)
    relative_l2 = float(
        np.linalg.norm(numerical_pressure - analytical_pressure)
        / np.linalg.norm(analytical_pressure)
    )
    return {
        "depth": depth,
        "numerical_pressure": numerical_pressure,
        "analytical_pressure": analytical_pressure,
        "relative_l2_error": relative_l2,
        "maximum_absolute_error": float(np.max(np.abs(numerical_pressure - analytical_pressure))),
        "kirchhoff_residual": solution.relative_kirchhoff_residual,
        "collar_flow": solution.collar_delivered_flow,
    }
