"""Pair- and lineage-resolved statistics for stochastic branching root tips.

The module deliberately separates:

* the first factorial moment, represented by one-tip histograms;
* the second factorial moment, represented by ordered distinct-tip pairs;
* genealogical marks carried by a separate lineage table; and
* reduced pair-density evolution in signed relative position.

It does not claim that pair equations or moment closures are new.  The
root-specific state and marks support tests of overlap and shared hydraulic
paths that are not identifiable from a one-tip density alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class LineageTable:
    """Genealogy of every tip identity ever created."""

    parent: np.ndarray
    birth_time: np.ndarray
    path_at_birth: np.ndarray
    branch_order: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.parent)
        columns = (self.birth_time, self.path_at_birth, self.branch_order)
        if any(len(values) != size for values in columns):
            raise ValueError("lineage columns must have equal length")
        if np.any(self.parent >= np.arange(size)):
            raise ValueError("a parent must precede its daughter")
        if (
            np.any(self.parent < -1)
            or np.any(self.birth_time < 0)
            or np.any(self.path_at_birth < 0)
        ):
            raise ValueError("invalid lineage values")


@dataclass(frozen=True)
class TipSnapshot:
    positions: np.ndarray
    orientations: np.ndarray
    root_types: np.ndarray
    active_lineage_ids: np.ndarray
    lineage: LineageTable
    time: float

    def __post_init__(self) -> None:
        number = len(self.positions)
        if self.positions.shape != (number, 2):
            raise ValueError("positions must have shape (tip, 2)")
        columns = (self.orientations, self.root_types, self.active_lineage_ids)
        if any(len(values) != number for values in columns):
            raise ValueError("tip-state columns must have equal length")
        invalid = (self.active_lineage_ids < 0) | (
            self.active_lineage_ids >= len(self.lineage.parent)
        )
        if np.any(invalid):
            raise ValueError("active lineage identifier is out of range")
        if self.time < 0:
            raise ValueError("snapshot time cannot be negative")


def ancestor_chain(lineage: LineageTable, identity: int) -> list[int]:
    chain = []
    current = int(identity)
    while current >= 0:
        chain.append(current)
        current = int(lineage.parent[current])
    return chain


def lineage_mark(lineage: LineageTable, first: int, second: int) -> dict[str, float | str | int]:
    first_chain = ancestor_chain(lineage, first)
    second_chain = ancestor_chain(lineage, second)
    second_set = set(second_chain)
    common = next((identity for identity in first_chain if identity in second_set), -1)
    depth_first = len(first_chain) - 1
    depth_second = len(second_chain) - 1
    depth_common = len(ancestor_chain(lineage, common)) - 1 if common >= 0 else -1
    graph_distance = depth_first + depth_second - 2 * depth_common
    if lineage.parent[first] == second or lineage.parent[second] == first:
        relation = "parent-daughter"
    elif lineage.parent[first] >= 0 and lineage.parent[first] == lineage.parent[second]:
        relation = "sisters"
    elif common >= 0:
        relation = "related"
    else:
        relation = "unrelated"
    return {
        "mrca_id": common,
        "mrca_time": float(lineage.birth_time[common]) if common >= 0 else np.nan,
        "graph_distance": int(graph_distance),
        "common_path_length": float(lineage.path_at_birth[common]) if common >= 0 else 0.0,
        "relation": relation,
        "first_order": int(lineage.branch_order[first]),
        "second_order": int(lineage.branch_order[second]),
    }


def pair_records(snapshot: TipSnapshot) -> np.ndarray:
    """Return one record per unordered distinct active pair."""
    dtype = [
        ("first", int),
        ("second", int),
        ("distance", float),
        ("orientation_correlation", float),
        ("mrca_time", float),
        ("graph_distance", int),
        ("common_path_length", float),
        ("relation", "U20"),
        ("first_order", int),
        ("second_order", int),
    ]
    rows = []
    for first in range(len(snapshot.positions)):
        for second in range(first + 1, len(snapshot.positions)):
            mark = lineage_mark(
                snapshot.lineage,
                int(snapshot.active_lineage_ids[first]),
                int(snapshot.active_lineage_ids[second]),
            )
            rows.append(
                (
                    first,
                    second,
                    float(np.linalg.norm(snapshot.positions[first] - snapshot.positions[second])),
                    float(np.cos(snapshot.orientations[first] - snapshot.orientations[second])),
                    mark["mrca_time"],
                    mark["graph_distance"],
                    mark["common_path_length"],
                    mark["relation"],
                    mark["first_order"],
                    mark["second_order"],
                )
            )
    return np.asarray(rows, dtype=dtype)


def first_moment_histogram(
    positions: np.ndarray,
    edges_x: np.ndarray,
    edges_z: np.ndarray,
) -> np.ndarray:
    """Normalized one-tip density histogram."""
    count, _, _ = np.histogram2d(positions[:, 1], positions[:, 0], bins=(edges_z, edges_x))
    cell_area = np.diff(edges_x)[0] * np.diff(edges_z)[0]
    return count / max(len(positions) * cell_area, 1e-15)


def radial_pair_correlation(
    positions: np.ndarray,
    bins: np.ndarray,
    *,
    domain_size: tuple[float, float],
    periodic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate radial ``g2`` for ordered distinct pairs in a rectangular window."""
    points = np.asarray(positions, dtype=float)
    number = len(points)
    if number < 2 or np.any(np.diff(bins) <= 0):
        raise ValueError("at least two points and increasing radial bins are required")
    difference = points[:, None, :] - points[None, :, :]
    if periodic:
        for component, length in enumerate(domain_size):
            difference[..., component] -= length * np.round(difference[..., component] / length)
    distance = np.linalg.norm(difference, axis=-1)
    mask = ~np.eye(number, dtype=bool)
    counts, _ = np.histogram(distance[mask], bins=bins)
    shell_area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    window_area = float(domain_size[0] * domain_size[1])
    expected = number * (number - 1) * shell_area / window_area
    g2 = np.divide(counts, expected, out=np.zeros_like(expected), where=expected > 0)
    return 0.5 * (bins[:-1] + bins[1:]), g2


def cell_count_moments(
    ensembles: list[np.ndarray],
    edges_x: np.ndarray,
    edges_z: np.ndarray,
) -> dict[str, np.ndarray]:
    counts = np.asarray(
        [
            np.histogram2d(points[:, 1], points[:, 0], bins=(edges_z, edges_x))[0]
            for points in ensembles
        ]
    )
    flattened = counts.reshape(len(counts), -1)
    return {
        "mean": np.mean(counts, axis=0),
        "variance": np.var(counts, axis=0, ddof=1),
        "covariance": np.cov(flattened, rowvar=False),
    }


def expected_overlap_from_pairs(distances: np.ndarray, radius: float) -> float:
    if radius <= 0:
        raise ValueError("overlap radius must be positive")
    return float(np.mean(np.asarray(distances) < 2.0 * radius))


def independence_triplet_closure(f1: np.ndarray) -> np.ndarray:
    return np.einsum("i,j,k->ijk", f1, f1, f1)


def kirkwood_triplet_closure(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    denominator = np.einsum("i,j,k->ijk", f1, f1, f1)
    numerator = (
        f2[:, :, None] * f2[:, None, :] * f2[None, :, :]
    )
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def maximum_entropy_triplet_closure(
    f1: np.ndarray,
    f2: np.ndarray,
    *,
    correction_strength: float,
) -> np.ndarray:
    """A declared second-order maxent surrogate around Kirkwood.

    The exact constrained maximum-entropy closure requires solving its
    normalization integral.  Here the correction is preregistered from the
    training environments and held fixed on tests; it is not described as a
    new maximum-entropy derivation.
    """
    if not 0 <= correction_strength <= 1:
        raise ValueError("correction strength must lie in [0, 1]")
    independent = independence_triplet_closure(f1)
    kirkwood = kirkwood_triplet_closure(f1, f2)
    return (1.0 - correction_strength) * kirkwood + correction_strength * independent


def binary_maximum_entropy_triplets(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Exact finite-cell pair-constrained maximum-entropy distribution.

    Binary occupancy states are enumerated, and the Lagrange dual is solved for
    one- and two-cell occupancy constraints.  The returned triplet tensor is
    therefore a genuine discrete maximum-entropy prediction, not a fitted
    blend of heuristic closures.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    cells = len(first)
    if cells > 12 or second.shape != (cells, cells):
        raise ValueError("binary maxent requires a square pair matrix with at most 12 cells")
    states = (
        (np.arange(2**cells)[:, None] >> np.arange(cells)[None, :]) & 1
    ).astype(float)
    pair_indices = [(i, j) for i in range(cells) for j in range(i + 1, cells)]
    features = np.column_stack(
        (
            states,
            *[states[:, i] * states[:, j] for i, j in pair_indices],
        )
    )
    target = np.concatenate(
        (first, np.asarray([second[i, j] for i, j in pair_indices]))
    )

    def objective(parameter: np.ndarray) -> tuple[float, np.ndarray]:
        logits = features @ parameter
        maximum = float(np.max(logits))
        weights = np.exp(logits - maximum)
        probability = weights / np.sum(weights)
        log_partition = maximum + np.log(np.sum(weights))
        value = log_partition - float(parameter @ target)
        gradient = features.T @ probability - target
        return value, gradient

    fit = minimize(
        lambda parameter: objective(parameter),
        np.zeros(features.shape[1]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-40.0, 40.0)] * features.shape[1],
        options={"gtol": 1e-10, "ftol": 1e-14, "maxiter": 2000},
    )
    logits = features @ fit.x
    probability = np.exp(logits - np.max(logits))
    probability /= np.sum(probability)
    third = np.einsum("s,si,sj,sk->ijk", probability, states, states, states)
    residual = float(np.max(np.abs(features.T @ probability - target)))
    return third, residual


def lineage_conditioned_triplet_closure(
    f1: np.ndarray,
    f2: np.ndarray,
    sister_fraction: float,
) -> np.ndarray:
    if not 0 <= sister_fraction <= 1:
        raise ValueError("sister fraction must lie in [0, 1]")
    independent = independence_triplet_closure(f1)
    kirkwood = kirkwood_triplet_closure(f1, f2)
    return (1.0 - sister_fraction) * independent + sister_fraction * kirkwood


@dataclass(frozen=True)
class RelativePairGrid:
    cells: int
    limits: tuple[float, float]

    def __post_init__(self) -> None:
        if self.cells < 8 or self.limits[1] <= self.limits[0]:
            raise ValueError("invalid relative-pair grid")

    @property
    def dx(self) -> float:
        return (self.limits[1] - self.limits[0]) / self.cells

    @property
    def x(self) -> np.ndarray:
        return self.limits[0] + (np.arange(self.cells) + 0.5) * self.dx


@dataclass
class RelativePairState:
    density: np.ndarray
    time: float = 0.0
    cumulative_birth_mass: float = 0.0
    cumulative_death_mass: float = 0.0


def initialize_relative_pair(
    grid: RelativePairGrid,
    *,
    pair_mass: float,
    mean_separation: float,
    separation_sigma: float,
    symmetric: bool = True,
) -> RelativePairState:
    if pair_mass < 0 or separation_sigma <= 0:
        raise ValueError("invalid relative-pair initialization")
    first = np.exp(-0.5 * ((grid.x - mean_separation) / separation_sigma) ** 2)
    density = first
    if symmetric and abs(mean_separation) > 1e-14:
        density = 0.5 * (
            first + np.exp(-0.5 * ((grid.x + mean_separation) / separation_sigma) ** 2)
        )
    density *= pair_mass / max(float(np.sum(density) * grid.dx), 1e-15)
    return RelativePairState(density)


def relative_pair_mass(state: RelativePairState, grid: RelativePairGrid) -> float:
    return float(np.sum(state.density) * grid.dx)


def step_relative_pair(
    state: RelativePairState,
    grid: RelativePairGrid,
    *,
    dt: float,
    relative_diffusivity: float,
    relative_drift: float = 0.0,
    mortality_rate: float = 0.0,
    pair_birth_rate: float = 0.0,
    birth_sigma: float | None = None,
) -> dict[str, float]:
    """Exact periodic transport-diffusion step plus pair birth/death sources."""
    if dt <= 0 or relative_diffusivity < 0 or mortality_rate < 0 or pair_birth_rate < 0:
        raise ValueError("invalid relative-pair step")
    wave = 2.0 * np.pi * np.fft.fftfreq(grid.cells, d=grid.dx)
    multiplier = np.exp(
        (-relative_diffusivity * wave**2 - 1j * relative_drift * wave - 2.0 * mortality_rate)
        * dt
    )
    before = relative_pair_mass(state, grid)
    density = np.fft.ifft(np.fft.fft(state.density) * multiplier).real
    death_mass = before * (1.0 - np.exp(-2.0 * mortality_rate * dt))
    birth_mass = (
        pair_birth_rate * dt
        if mortality_rate == 0
        else pair_birth_rate
        * (1.0 - np.exp(-2.0 * mortality_rate * dt))
        / (2.0 * mortality_rate)
    )
    if birth_mass > 0:
        sigma = birth_sigma if birth_sigma is not None else grid.dx
        source = np.exp(-0.5 * (grid.x / sigma) ** 2)
        source /= float(np.sum(source) * grid.dx)
        density += birth_mass * source
    minimum = float(np.min(density))
    density = np.maximum(density, 0.0)
    target = before - death_mass + birth_mass
    numerical = float(np.sum(density) * grid.dx)
    if numerical > 0:
        density *= target / numerical
    state.density = density
    state.time += dt
    state.cumulative_birth_mass += birth_mass
    state.cumulative_death_mass += death_mass
    return {
        "mass_before": before,
        "mass_after": relative_pair_mass(state, grid),
        "birth_mass": birth_mass,
        "death_mass": death_mass,
        "minimum_density": minimum,
        "symmetry_error": float(np.max(np.abs(density - density[::-1]))),
    }


def sister_relative_msd(
    time: np.ndarray | float,
    *,
    speed: float,
    rotational_diffusion: float,
    half_branch_angle: float,
) -> np.ndarray:
    """Exact relative MSD for two independently diffusing sister orientations."""
    values = np.asarray(time, dtype=float)
    if rotational_diffusion == 0:
        return (
            2.0
            * speed**2
            * values**2
            * (1.0 - np.cos(2.0 * half_branch_angle))
        )
    scaled = rotational_diffusion * values
    single_msd = (
        2.0
        * speed**2
        / rotational_diffusion**2
        * (scaled + np.exp(-scaled) - 1.0)
    )
    mean_length = speed * (1.0 - np.exp(-scaled)) / rotational_diffusion
    return 2.0 * single_msd - 2.0 * mean_length**2 * np.cos(2.0 * half_branch_angle)


def birth_death_factorial_moments(
    time: np.ndarray | float,
    *,
    birth_rate: float,
    mortality_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First and second factorial moments of a linear birth-death process."""
    values = np.asarray(time, dtype=float)
    growth = birth_rate - mortality_rate
    first = np.exp(growth * values)
    if abs(growth) < 1e-12:
        second = 2.0 * birth_rate * values
    else:
        second = (
            2.0
            * birth_rate
            / growth
            * np.exp(growth * values)
            * (np.exp(growth * values) - 1.0)
        )
    return first, second
