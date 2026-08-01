"""Sparse axial/radial hydraulic graph with explicit sign conventions."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve


class SingularHydraulicSystem(RuntimeError):
    """Raised when a hydraulic component has no potential reference."""


@dataclass(frozen=True)
class HydraulicSegment:
    """One root segment.

    ``axial_conductivity / length`` is the axial conductance. The declared
    ``radial_conductance`` is the total conductance of the segment; its influx
    ``G_r (psi_soil - (psi_i + psi_j)/2)`` is divided equally between its two
    endpoint balance equations.
    """

    start: int
    end: int
    length: float
    axial_conductivity: float
    radial_conductance: float = 0.0
    soil_potential: float = 0.0

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("hydraulic segment endpoints must differ")
        if self.length <= 0:
            raise ValueError("hydraulic segment length must be positive")
        if self.axial_conductivity < 0 or self.radial_conductance < 0:
            raise ValueError("hydraulic conductances must be nonnegative")

    @property
    def axial_conductance(self) -> float:
        return self.axial_conductivity / self.length


@dataclass(frozen=True)
class HydraulicSolution:
    nodes: tuple[int, ...]
    potentials: np.ndarray
    axial_flows: np.ndarray
    radial_flows: np.ndarray
    collar_delivered_flow: float
    relative_kirchhoff_residual: float
    dissipation: float

    def potential(self, node: int) -> float:
        return float(self.potentials[self.nodes.index(node)])


def maturation_multiplier(
    age: float,
    maturation_time: float,
    *,
    minimum_fraction: float = 0.05,
) -> float:
    """Monotone saturating multiplier for age-dependent conductance."""
    if age < 0 or maturation_time <= 0 or not 0 <= minimum_fraction <= 1:
        raise ValueError("invalid maturation parameters")
    return minimum_fraction + (1.0 - minimum_fraction) * (
        1.0 - math.exp(-age / maturation_time)
    )


def _unanchored_components(
    segments: tuple[HydraulicSegment, ...],
    nodes: tuple[int, ...],
    fixed_nodes: set[int],
) -> list[set[int]]:
    adjacency = {node: set() for node in nodes}
    radial_nodes: set[int] = set()
    for segment in segments:
        if segment.axial_conductance > 0:
            adjacency[segment.start].add(segment.end)
            adjacency[segment.end].add(segment.start)
        if segment.radial_conductance > 0:
            radial_nodes.update((segment.start, segment.end))
    unseen = set(nodes)
    unanchored = []
    while unseen:
        start = unseen.pop()
        component = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            new = adjacency[current] & unseen
            unseen -= new
            component |= new
            frontier.extend(new)
        if not (component & fixed_nodes or component & radial_nodes):
            unanchored.append(component)
    return unanchored


def solve_hydraulic_network(
    segments: Iterable[HydraulicSegment],
    *,
    collar_node: int,
    collar_potential: float | None = None,
    collar_flux: float | None = None,
    fixed_potentials: dict[int, float] | None = None,
    nodes: Iterable[int] | None = None,
) -> HydraulicSolution:
    """Solve nodal water potentials with sparse linear algebra.

    Axial flow is positive from ``segment.start`` to ``segment.end``. Radial
    flow is positive from soil into root. Collar-delivered flow is positive
    out of the root network. ``collar_flux`` follows that delivered-positive
    convention.
    """
    edges = tuple(segments)
    if (collar_potential is None) == (collar_flux is None):
        raise ValueError("specify exactly one of collar_potential or collar_flux")
    node_set = set(nodes or ())
    node_set.add(collar_node)
    for edge in edges:
        node_set.update((edge.start, edge.end))
    ordered = tuple(sorted(node_set))
    index = {node: position for position, node in enumerate(ordered)}
    fixed = dict(fixed_potentials or {})
    if collar_potential is not None:
        if collar_node in fixed and not math.isclose(
            fixed[collar_node], collar_potential
        ):
            raise ValueError("conflicting collar potentials")
        fixed[collar_node] = float(collar_potential)
    unknown_fixed = set(fixed) - node_set
    if unknown_fixed:
        raise ValueError(f"fixed potential nodes not in graph: {unknown_fixed}")
    unanchored = _unanchored_components(edges, ordered, set(fixed))
    if unanchored:
        raise SingularHydraulicSystem(
            f"components without Dirichlet or radial reference: {unanchored}"
        )

    size = len(ordered)
    matrix = lil_matrix((size, size), dtype=float)
    rhs = np.zeros(size)
    for edge in edges:
        i, j = index[edge.start], index[edge.end]
        gx = edge.axial_conductance
        gr = edge.radial_conductance
        matrix[i, i] += gx + 0.25 * gr
        matrix[j, j] += gx + 0.25 * gr
        matrix[i, j] += -gx + 0.25 * gr
        matrix[j, i] += -gx + 0.25 * gr
        rhs[i] += 0.5 * gr * edge.soil_potential
        rhs[j] += 0.5 * gr * edge.soil_potential
    original_matrix = matrix.tocsr()
    original_rhs = rhs.copy()
    if collar_flux is not None:
        rhs[index[collar_node]] -= float(collar_flux)
    balance_rhs = rhs.copy()

    for node, value in fixed.items():
        position = index[node]
        column = matrix[:, position].toarray().ravel()
        rhs -= column * value
        matrix[:, position] = 0.0
        matrix[position, :] = 0.0
        matrix[position, position] = 1.0
        rhs[position] = value
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        try:
            potentials = np.asarray(spsolve(matrix.tocsr(), rhs), dtype=float)
        except MatrixRankWarning as exc:
            raise SingularHydraulicSystem("sparse hydraulic matrix is singular") from exc
    if not np.isfinite(potentials).all():
        raise SingularHydraulicSystem("hydraulic solution is not finite")

    axial = np.asarray(
        [
            edge.axial_conductance
            * (potentials[index[edge.start]] - potentials[index[edge.end]])
            for edge in edges
        ]
    )
    radial = np.asarray(
        [
            edge.radial_conductance
            * (
                edge.soil_potential
                - 0.5
                * (potentials[index[edge.start]] + potentials[index[edge.end]])
            )
            for edge in edges
        ]
    )
    original_residual = original_matrix @ potentials - original_rhs
    collar_delivered = -float(original_residual[index[collar_node]])
    residual = original_matrix @ potentials - balance_rhs
    free = np.ones(size, dtype=bool)
    for node in fixed:
        free[index[node]] = False
    numerator = float(np.linalg.norm(residual[free]))
    denominator = max(
        float(np.linalg.norm((original_matrix @ potentials)[free])),
        float(np.linalg.norm(balance_rhs[free])),
        1e-15,
    )
    relative_residual = numerator / denominator
    dissipation = float(
        sum(
            edge.axial_conductance
            * (potentials[index[edge.start]] - potentials[index[edge.end]]) ** 2
            + edge.radial_conductance
            * (
                edge.soil_potential
                - 0.5
                * (potentials[index[edge.start]] + potentials[index[edge.end]])
            )
            ** 2
            for edge in edges
        )
    )
    return HydraulicSolution(
        ordered,
        potentials,
        axial,
        radial,
        collar_delivered,
        relative_residual,
        dissipation,
    )
