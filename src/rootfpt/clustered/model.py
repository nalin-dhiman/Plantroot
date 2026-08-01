"""Exact raster quadrature for clustered discovery and finite-capacity reward."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammainc, gammaincc

ClusterFamily = Literal["thomas", "matern"]
CapacityFamily = Literal["deterministic", "exponential", "gamma"]


@dataclass(frozen=True)
class Grid2D:
    """Cell-centred square quadrature grid."""

    lower: float = -3.0
    upper: float = 3.0
    resolution: int = 192

    def __post_init__(self) -> None:
        if self.upper <= self.lower or self.resolution < 32:
            raise ValueError("invalid grid")

    @property
    def dx(self) -> float:
        return (self.upper - self.lower) / self.resolution

    @property
    def cell_area(self) -> float:
        return self.dx**2

    @property
    def area(self) -> float:
        return (self.upper - self.lower) ** 2

    @property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        axis = self.lower + (np.arange(self.resolution) + 0.5) * self.dx
        return np.meshgrid(axis, axis)


@dataclass(frozen=True)
class NetworkGeometry:
    """Embedded tree segments with root-path and deployment metadata."""

    segments: np.ndarray
    path_starts: np.ndarray
    deployment_times: np.ndarray

    def __post_init__(self) -> None:
        segments = np.asarray(self.segments, dtype=float)
        starts = np.asarray(self.path_starts, dtype=float)
        times = np.asarray(self.deployment_times, dtype=float)
        if segments.ndim != 3 or segments.shape[1:] != (2, 2):
            raise ValueError("segments must have shape (n, 2, 2)")
        if starts.shape != (len(segments),) or times.shape != (len(segments),):
            raise ValueError("metadata must match segment count")
        if np.any(starts < 0) or np.any(times < 0):
            raise ValueError("path starts and deployment times must be nonnegative")
        if np.any(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) <= 0):
            raise ValueError("segments must have positive length")

    @property
    def total_length(self) -> float:
        return float(np.linalg.norm(self.segments[:, 1] - self.segments[:, 0], axis=1).sum())


@dataclass(frozen=True)
class EqualBudget:
    """Material ledger for one extend or split-branch action."""

    step_budget: float = 1.0
    length_cost: float = 1.0
    maintenance_cost: float = 0.1
    residence_time: float = 1.0
    branch_initiation_cost: float = 0.08

    def __post_init__(self) -> None:
        if (
            min(
                self.step_budget,
                self.length_cost,
                self.residence_time,
            )
            <= 0
        ):
            raise ValueError("budget, length cost and residence time must be positive")
        if self.maintenance_cost < 0 or self.branch_initiation_cost < 0:
            raise ValueError("costs must be nonnegative")
        if self.branch_initiation_cost >= self.step_budget:
            raise ValueError("branch initiation must be below the step budget")

    @property
    def effective_length_cost(self) -> float:
        return self.length_cost + self.maintenance_cost * self.residence_time

    @property
    def extension_length(self) -> float:
        return self.step_budget / self.effective_length_cost

    @property
    def branch_total_length(self) -> float:
        return (self.step_budget - self.branch_initiation_cost) / self.effective_length_cost

    def spent(self, action: Literal["extend", "branch"]) -> float:
        if action == "extend":
            return self.effective_length_cost * self.extension_length
        if action == "branch":
            return (
                self.branch_initiation_cost + self.effective_length_cost * self.branch_total_length
            )
        raise ValueError(f"unknown action: {action}")


def action_networks(
    budget: EqualBudget,
    *,
    branch_angle: float,
    base_length: float = 0.6,
    action_deployment_time: float = 0.15,
) -> dict[str, NetworkGeometry]:
    """Return a base, one-axis extension, and symmetric split at equal material."""
    if not 0 < branch_angle < math.pi:
        raise ValueError("branch_angle must be between zero and pi")
    base = np.asarray([[[0.0, -base_length], [0.0, 0.0]]])
    base_geometry = NetworkGeometry(base, np.asarray([0.0]), np.asarray([0.0]))
    le = budget.extension_length
    extension = np.concatenate(
        [base, np.asarray([[[0.0, 0.0], [0.0, le]]])],
        axis=0,
    )
    extension_geometry = NetworkGeometry(
        extension,
        np.asarray([0.0, base_length]),
        np.asarray([0.0, action_deployment_time]),
    )
    daughter_length = budget.branch_total_length / 2.0
    half = branch_angle / 2.0
    left = [-math.sin(half) * daughter_length, math.cos(half) * daughter_length]
    right = [math.sin(half) * daughter_length, math.cos(half) * daughter_length]
    branch = np.concatenate(
        [
            base,
            np.asarray(
                [
                    [[0.0, 0.0], left],
                    [[0.0, 0.0], right],
                ]
            ),
        ],
        axis=0,
    )
    branch_geometry = NetworkGeometry(
        branch,
        np.asarray([0.0, base_length, base_length]),
        np.asarray([0.0, action_deployment_time, action_deployment_time]),
    )
    return {
        "base": base_geometry,
        "extend": extension_geometry,
        "branch": branch_geometry,
    }


def _distance_and_projection(
    x: np.ndarray,
    y: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vector = end - start
    denominator = float(vector @ vector)
    projection = ((x - start[0]) * vector[0] + (y - start[1]) * vector[1]) / denominator
    projection = np.clip(projection, 0.0, 1.0)
    distance_squared = (x - (start[0] + projection * vector[0])) ** 2 + (
        y - (start[1] + projection * vector[1])
    ) ** 2
    return distance_squared, projection


def rasterize_network(
    geometry: NetworkGeometry,
    grid: Grid2D,
    search_radius: float,
    *,
    supersample: int = 4,
) -> np.ndarray:
    """Return subcell-averaged tube coverage fractions on grid cells."""
    if search_radius <= 0:
        raise ValueError("search_radius must be positive")
    if supersample < 1:
        raise ValueError("supersample must be positive")
    x_center, y_center = grid.coordinates
    coverage = np.zeros_like(x_center, dtype=float)
    offsets = ((np.arange(supersample) + 0.5) / supersample - 0.5) * grid.dx
    for x_offset in offsets:
        for y_offset in offsets:
            x = x_center + x_offset
            y = y_center + y_offset
            best = np.full_like(x, np.inf, dtype=float)
            for segment in geometry.segments:
                distance_squared, _ = _distance_and_projection(
                    x,
                    y,
                    segment[0],
                    segment[1],
                )
                best = np.minimum(best, distance_squared)
            coverage += best <= search_radius**2
    return coverage / supersample**2


def kernel_on_grid(
    grid: Grid2D,
    sigma: float,
    family: ClusterFamily,
) -> np.ndarray:
    """Return a normalized offspring density sampled on convolution offsets."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    offsets = (np.arange(2 * grid.resolution - 1) - (grid.resolution - 1)) * grid.dx
    x, y = np.meshgrid(offsets, offsets)
    if family == "thomas":
        kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    elif family == "matern":
        kernel = ((x**2 + y**2) <= sigma**2).astype(float)
    else:
        raise ValueError(f"unknown cluster family: {family}")
    normalization = float(kernel.sum() * grid.cell_area)
    if normalization <= 0:
        raise ValueError("kernel is unresolved on grid")
    return kernel / normalization


def cluster_probability_map(
    search_mask: np.ndarray,
    *,
    grid: Grid2D,
    sigma: float,
    family: ClusterFamily,
) -> np.ndarray:
    """Compute ``p_G(z)=integral_S f_sigma(x-z) dx`` by zero-padded convolution."""
    if search_mask.shape != (grid.resolution, grid.resolution):
        raise ValueError("search mask does not match grid")
    kernel = kernel_on_grid(grid, sigma, family)
    probability = (
        fftconvolve(
            search_mask.astype(float),
            kernel,
            mode="same",
        )
        * grid.cell_area
    )
    return np.clip(probability, 0.0, 1.0)


def hydraulic_volume_map(
    geometry: NetworkGeometry,
    grid: Grid2D,
    *,
    deadline: float,
    hydraulic_timescale: float,
    axial_conductivity: float = 1.0,
    radial_conductance: float = 0.7,
    access_radius: float | None = None,
) -> np.ndarray:
    """Path-resistance delivery volume assigned to each possible cluster centre.

    A cluster uses the best-delivering segment within ``access_radius``. Axial
    conductance along the collar-to-contact path is ``K_x / path_length`` and
    the one-segment circuit flow is ``4*g_r*g_x/(4*g_x+g_r)``. Remaining
    delivery time is deadline minus deployment and a path-proportional
    transport delay. With no access radius, every segment is eligible.
    """
    if min(deadline, axial_conductivity, radial_conductance) <= 0:
        raise ValueError("deadline and conductances must be positive")
    if hydraulic_timescale < 0:
        raise ValueError("hydraulic_timescale must be nonnegative")
    if access_radius is not None and access_radius <= 0:
        raise ValueError("access_radius must be positive")
    x, y = grid.coordinates
    best_volume = np.zeros_like(x, dtype=float)
    for segment, path_start, deployment in zip(
        geometry.segments,
        geometry.path_starts,
        geometry.deployment_times,
        strict=True,
    ):
        distance_squared, projection = _distance_and_projection(
            x,
            y,
            segment[0],
            segment[1],
        )
        length = float(np.linalg.norm(segment[1] - segment[0]))
        path = np.maximum(path_start + projection * length, grid.dx / 2.0)
        gx = axial_conductivity / path
        flow = 4.0 * radial_conductance * gx / (4.0 * gx + radial_conductance)
        delay = hydraulic_timescale * path
        remaining = np.maximum(0.0, deadline - deployment - delay)
        volume = flow * remaining
        eligible = (
            np.ones_like(distance_squared, dtype=bool)
            if access_radius is None
            else distance_squared <= access_radius**2
        )
        best_volume = np.maximum(best_volume, np.where(eligible, volume, 0.0))
    return best_volume


def capacity_truncated_mean(
    volume: np.ndarray | float,
    *,
    family: CapacityFamily,
    mean_capacity: float,
    gamma_shape: float = 2.0,
) -> np.ndarray:
    """Return ``E[min(R, volume)]`` for three capacity families."""
    if mean_capacity <= 0:
        raise ValueError("mean_capacity must be positive")
    values = np.maximum(np.asarray(volume, dtype=float), 0.0)
    if family == "deterministic":
        return np.minimum(values, mean_capacity)
    if family == "exponential":
        return mean_capacity * (-np.expm1(-values / mean_capacity))
    if family == "gamma":
        if gamma_shape <= 0:
            raise ValueError("gamma_shape must be positive")
        scale = mean_capacity / gamma_shape
        cutoff = values / scale
        return mean_capacity * gammainc(gamma_shape + 1.0, cutoff) + values * gammaincc(
            gamma_shape, cutoff
        )
    raise ValueError(f"unknown capacity family: {family}")


def cluster_metrics(
    probability_map: np.ndarray,
    *,
    grid: Grid2D,
    parent_intensity: float,
    mean_offspring: float,
    capacity_mean_map: np.ndarray | None = None,
    search_area: float | None = None,
) -> dict[str, float]:
    """Integrate exact parent-cluster discovery and marked reward quantities."""
    if parent_intensity < 0 or mean_offspring < 0:
        raise ValueError("intensities must be nonnegative")
    if probability_map.shape != (grid.resolution, grid.resolution):
        raise ValueError("probability map does not match grid")
    hit = -np.expm1(-mean_offspring * probability_map)
    distinct = parent_intensity * float(hit.sum() * grid.cell_area)
    result = {
        "expected_distinct_clusters": distinct,
        "probability_any_cluster": -math.expm1(-distinct),
    }
    if search_area is not None:
        result["expected_micro_site_contacts"] = parent_intensity * mean_offspring * search_area
    if capacity_mean_map is not None:
        if capacity_mean_map.shape != probability_map.shape:
            raise ValueError("capacity map does not match probability map")
        result["expected_delivered_resource"] = parent_intensity * float(
            (hit * capacity_mean_map).sum() * grid.cell_area
        )
    return result
