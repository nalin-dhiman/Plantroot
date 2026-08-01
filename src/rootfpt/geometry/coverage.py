"""Raster-converged search-tube geometry and Poisson coverage tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rootfpt.metrics.intervals import wilson_interval


@dataclass(frozen=True)
class RectDomain:
    x_min: float
    x_max: float
    z_min: float
    z_max: float

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.z_max - self.z_min)


def polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("points must have shape (n>=2, 2)")
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _distance_squared_to_segments(
    x: np.ndarray,
    z: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    best = np.full(np.broadcast_shapes(x.shape, z.shape), np.inf, dtype=float)
    for start, end in zip(points[:-1], points[1:], strict=True):
        vx, vz = end - start
        denominator = vx * vx + vz * vz
        if denominator == 0:
            distance = (x - start[0]) ** 2 + (z - start[1]) ** 2
        else:
            projection = ((x - start[0]) * vx + (z - start[1]) * vz) / denominator
            projection = np.clip(projection, 0.0, 1.0)
            distance = (x - (start[0] + projection * vx)) ** 2 + (
                z - (start[1] + projection * vz)
            ) ** 2
        best = np.minimum(best, distance)
    return best


def points_in_tube(x: np.ndarray, z: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray:
    """Return exact segment-distance membership for points in a polyline tube."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    return _distance_squared_to_segments(np.asarray(x), np.asarray(z), points) <= radius**2


def graph_tube_metrics(
    *,
    segments: np.ndarray,
    radius: float,
    domain: RectDomain,
    resolution: int,
) -> dict[str, float]:
    """Approximate union coverage for an arbitrary set of line segments."""
    segments = np.asarray(segments, dtype=float)
    if segments.size == 0:
        return {
            "unique_search_tube_coverage": 0.0,
            "path_overlap": 0.0,
            "path_overlap_fraction": 0.0,
        }
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError("segments must have shape (n, 2, 2)")
    dx = (domain.x_max - domain.x_min) / resolution
    dz = (domain.z_max - domain.z_min) / resolution
    xs = domain.x_min + (np.arange(resolution) + 0.5) * dx
    zs = domain.z_min + (np.arange(resolution) + 0.5) * dz
    x_grid, z_grid = np.meshgrid(xs, zs)
    mask = np.zeros_like(x_grid, dtype=bool)
    total_length = 0.0
    for segment in segments:
        mask |= points_in_tube(x_grid, z_grid, segment, radius)
        total_length += math.dist(segment[0], segment[1])
    unique = float(mask.sum() * dx * dz)
    nominal = 2.0 * radius * total_length + math.pi * radius**2
    overlap = max(0.0, nominal - unique)
    return {
        "unique_search_tube_coverage": unique,
        "path_overlap": overlap,
        "path_overlap_fraction": overlap / nominal if nominal > 0 else 0.0,
    }


def intensity(x: np.ndarray, z: np.ndarray, family: str, base: float) -> np.ndarray:
    """Evaluate a declared Poisson intensity family."""
    if family == "homogeneous":
        return np.full(np.broadcast_shapes(x.shape, z.shape), base, dtype=float)
    if family == "inhomogeneous":
        bump = np.exp(-((x - 0.72) ** 2 / 0.32**2 + (z - 0.60) ** 2 / 0.28**2) / 2.0)
        return base * (0.35 + 1.9 * bump)
    raise ValueError(f"unknown intensity family: {family}")


def raster_metrics(
    *,
    points: np.ndarray,
    radius: float,
    domain: RectDomain,
    resolution: int,
    intensity_family: str,
    intensity_base: float,
) -> dict[str, float | int | str]:
    """Compute unique and weighted coverage on a cell-centred raster."""
    if resolution < 16:
        raise ValueError("resolution must be at least 16")
    dx = (domain.x_max - domain.x_min) / resolution
    dz = (domain.z_max - domain.z_min) / resolution
    xs = domain.x_min + (np.arange(resolution) + 0.5) * dx
    zs = domain.z_min + (np.arange(resolution) + 0.5) * dz
    x_grid, z_grid = np.meshgrid(xs, zs)
    mask = points_in_tube(x_grid, z_grid, points, radius)
    cell_area = dx * dz
    unique_area = float(mask.sum() * cell_area)
    total_length = polyline_length(points)
    nominal_area = 2.0 * radius * total_length + math.pi * radius**2
    overlap_area = max(0.0, nominal_area - unique_area)
    lambda_grid = intensity(x_grid, z_grid, intensity_family, intensity_base)
    weighted_unique_coverage = float((lambda_grid * mask).sum() * cell_area)
    return {
        "family": intensity_family,
        "resolution": resolution,
        "total_root_length": total_length,
        "nominal_tube_area": nominal_area,
        "unique_tube_area": unique_area,
        "overlap_area": overlap_area,
        "overlap_fraction": overlap_area / nominal_area,
        "weighted_unique_coverage": weighted_unique_coverage,
        "theoretical_hit_probability": -math.expm1(-weighted_unique_coverage),
    }


def _sample_inhomogeneous_points(
    rng: np.random.Generator,
    domain: RectDomain,
    family: str,
    base: float,
) -> tuple[np.ndarray, np.ndarray]:
    if family == "homogeneous":
        upper = base
    elif family == "inhomogeneous":
        upper = 2.25 * base
    else:
        raise ValueError(f"unknown intensity family: {family}")
    candidates = rng.poisson(upper * domain.area)
    x = rng.uniform(domain.x_min, domain.x_max, candidates)
    z = rng.uniform(domain.z_min, domain.z_max, candidates)
    keep = rng.random(candidates) < intensity(x, z, family, base) / upper
    return x[keep], z[keep]


def empirical_poisson_hit_probability(
    *,
    points: np.ndarray,
    radius: float,
    domain: RectDomain,
    intensity_family: str,
    intensity_base: float,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[int, float, float, float]:
    """Generate Poisson point patterns and test geometric tube contact."""
    hits = 0
    for _ in range(replicates):
        x, z = _sample_inhomogeneous_points(rng, domain, intensity_family, intensity_base)
        if x.size and bool(points_in_tube(x, z, points, radius).any()):
            hits += 1
    low, high = wilson_interval(hits, replicates)
    return hits, hits / replicates, low, high


def verify_poisson_coverage(
    *,
    points: np.ndarray,
    radius: float,
    domain: RectDomain,
    resolutions: tuple[int, ...],
    families: tuple[str, ...],
    intensity_base: float,
    replicates: int,
    master_seed: int,
    probability_tolerance: float,
    area_relative_tolerance: float,
) -> pd.DataFrame:
    """Run raster convergence and independent Poisson point simulations."""
    sequences = np.random.SeedSequence(master_seed).spawn(len(families))
    rows: list[dict[str, float | int | str | bool]] = []
    for family, sequence in zip(families, sequences, strict=True):
        family_metrics = [
            raster_metrics(
                points=points,
                radius=radius,
                domain=domain,
                resolution=resolution,
                intensity_family=family,
                intensity_base=intensity_base,
            )
            for resolution in resolutions
        ]
        final = family_metrics[-1]
        hits, estimate, low, high = empirical_poisson_hit_probability(
            points=points,
            radius=radius,
            domain=domain,
            intensity_family=family,
            intensity_base=intensity_base,
            replicates=replicates,
            rng=np.random.default_rng(sequence),
        )
        theoretical = float(final["theoretical_hit_probability"])
        probability_error = abs(estimate - theoretical)
        for index, metrics in enumerate(family_metrics):
            previous = family_metrics[index - 1] if index else None
            area_change = (
                abs(
                    float(metrics["unique_tube_area"])
                    - float(previous["unique_tube_area"])
                )
                / float(metrics["unique_tube_area"])
                if previous is not None
                else math.nan
            )
            row = dict(metrics)
            row.update(
                {
                    "replicates": replicates,
                    "successes": hits,
                    "empirical_hit_probability": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "probability_absolute_error": probability_error,
                    "area_relative_change": area_change,
                    "probability_tolerance": probability_tolerance,
                    "area_relative_tolerance": area_relative_tolerance,
                    "probability_passed": (low <= theoretical <= high)
                    or probability_error <= probability_tolerance,
                    "resolution_passed": index == len(family_metrics) - 1
                    and area_change <= area_relative_tolerance,
                    "seed_spawn_key": ".".join(str(item) for item in sequence.spawn_key),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
