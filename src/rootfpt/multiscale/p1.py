"""Polarization-retaining P1/telegraph closure of the root-tip kinetic equation.

For a two-dimensional orientation density, define the zeroth moment ``n`` and
first moment ``m``.  The declared P1 closure sets the traceless nematic moment
to zero, so ``integral(p p f) dp = n I/2``.  The resulting equations are

``partial_t n + v div(m) = births - deaths``

``partial_t m + v grad(n)/2 = -D_r m + n F/2 + polarized births - deaths``.

No transport or relaxation coefficient is fitted.  In a homogeneous,
unforced medium, eliminating ``m`` gives the telegraph equation

``partial_tt n + D_r partial_t n = (v**2/2) Laplacian(n)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import iv

from rootfpt.multiscale.soil import Grid2D


@dataclass
class P1State:
    """P1 moments with axes ``type,z,x`` and ``type,z,x,component``."""

    density: np.ndarray
    polarization: np.ndarray
    root_length_density: np.ndarray
    cumulative_bottom_passage: np.ndarray
    time: float = 0.0


@dataclass(frozen=True)
class P1Parameters:
    speeds: np.ndarray
    rotational_diffusion: np.ndarray
    mortality: np.ndarray
    branching_rate: np.ndarray
    transition: np.ndarray
    daughter_polarization: np.ndarray
    force: np.ndarray
    root_decay: float = 0.0

    def validate(self, number_types: int, grid: Grid2D) -> None:
        vectors = (
            self.speeds,
            self.rotational_diffusion,
            self.mortality,
            self.branching_rate,
            self.daughter_polarization,
        )
        if any(np.shape(vector) != (number_types,) for vector in vectors):
            raise ValueError("P1 type vectors have inconsistent shape")
        if np.shape(self.transition) != (number_types, number_types):
            raise ValueError("P1 transition matrix has inconsistent shape")
        if np.shape(self.force) not in {
            (number_types, 2),
            (number_types, grid.nz, grid.nx, 2),
        }:
            raise ValueError("P1 force must be type-constant or defined on the grid")
        if any(np.any(np.asarray(vector) < 0) for vector in vectors):
            raise ValueError("P1 speeds, rates and inheritance must be nonnegative")
        if np.any(self.transition < 0) or self.root_decay < 0:
            raise ValueError("P1 transition and decay values must be nonnegative")
        active = self.branching_rate > 0
        if np.any(np.abs(np.sum(self.transition, axis=1)[active] - 1.0) > 1e-10):
            raise ValueError("active P1 transition rows must sum to one")


def initialize_p1_state(
    grid: Grid2D,
    *,
    count_by_type: np.ndarray,
    centre: tuple[float, float],
    spatial_sigma: float,
    mean_angle: float | np.ndarray | None = None,
    angle_concentration: float | np.ndarray = 0.0,
) -> P1State:
    """Initialize a Gaussian density and its exact von-Mises first moment."""
    counts = np.asarray(count_by_type, dtype=float)
    if counts.ndim != 1 or np.any(counts < 0) or spatial_sigma <= 0:
        raise ValueError("invalid P1 initialization")
    horizontal, depth = grid.mesh
    spatial = np.exp(
        -0.5
        * (
            ((horizontal - centre[0]) / spatial_sigma) ** 2
            + ((depth - centre[1]) / spatial_sigma) ** 2
        )
    )
    spatial /= float(np.sum(spatial) * grid.cell_area)
    density = counts[:, None, None] * spatial[None, :, :]
    polarization = np.zeros((*density.shape, 2))
    if mean_angle is not None:
        angles = np.broadcast_to(np.asarray(mean_angle, dtype=float), counts.shape)
        concentration = np.broadcast_to(
            np.asarray(angle_concentration, dtype=float), counts.shape
        )
        if np.any(concentration < 0):
            raise ValueError("angle concentration must be nonnegative")
        denominator = iv(0, concentration)
        magnitude = np.divide(
            iv(1, concentration),
            denominator,
            out=np.zeros_like(concentration),
            where=denominator > 0,
        )
        direction = np.column_stack((np.cos(angles), np.sin(angles)))
        polarization = (
            density[:, :, :, None]
            * magnitude[:, None, None, None]
            * direction[:, None, None, :]
        )
    return P1State(
        density=density,
        polarization=polarization,
        root_length_density=np.zeros_like(density),
        cumulative_bottom_passage=np.zeros(len(counts)),
    )


def p1_mass(state: P1State, grid: Grid2D) -> np.ndarray:
    return np.sum(state.density, axis=(1, 2)) * grid.cell_area


def p1_orientation_density(
    state: P1State,
    theta: np.ndarray,
    grid: Grid2D,
) -> np.ndarray:
    """Return the un-clipped P1 angular reconstruction integrated over space."""
    theta = np.asarray(theta, dtype=float)
    direction = np.column_stack((np.cos(theta), np.sin(theta)))
    total_n = np.sum(state.density, axis=(1, 2)) * grid.cell_area
    total_m = np.sum(state.polarization, axis=(1, 2)) * grid.cell_area
    return (
        total_n[:, None] + 2.0 * np.einsum("tc,kc->tk", total_m, direction)
    ) / (2.0 * np.pi)


def p1_msd(time: np.ndarray | float, speed: float, rotational_diffusion: float) -> np.ndarray:
    """Exact second moment of the homogeneous P1/telegraph closure in 2-D."""
    values = np.asarray(time, dtype=float)
    if rotational_diffusion == 0:
        return (speed * values) ** 2
    scaled = rotational_diffusion * values
    return (
        2.0
        * speed**2
        / rotational_diffusion**2
        * (scaled + np.exp(-scaled) - 1.0)
    )


def p1_characteristic_speed(speed: np.ndarray | float) -> np.ndarray:
    """Finite signal speed of the homogeneous two-dimensional P1 system."""
    return np.asarray(speed, dtype=float) / np.sqrt(2.0)


def _flux_x(state: np.ndarray, speed: float) -> np.ndarray:
    flux = np.zeros_like(state)
    flux[..., 0] = speed * state[..., 1]
    flux[..., 1] = 0.5 * speed * state[..., 0]
    return flux


def _flux_z(state: np.ndarray, speed: float) -> np.ndarray:
    flux = np.zeros_like(state)
    flux[..., 0] = speed * state[..., 2]
    flux[..., 2] = 0.5 * speed * state[..., 0]
    return flux


def _rusanov(left: np.ndarray, right: np.ndarray, speed: float, axis: str) -> np.ndarray:
    physical = _flux_x if axis == "x" else _flux_z
    wavespeed = speed / np.sqrt(2.0)
    return 0.5 * (physical(left, speed) + physical(right, speed)) - 0.5 * wavespeed * (
        right - left
    )


def _boundary_ghost(interior: np.ndarray, boundary: str, component: int) -> np.ndarray:
    if boundary == "outflow":
        return np.zeros_like(interior)
    if boundary == "reflecting":
        ghost = interior.copy()
        ghost[..., component] *= -1.0
        return ghost
    raise ValueError(f"unknown P1 boundary {boundary!r}")


def step_p1(
    state: P1State,
    grid: Grid2D,
    parameters: P1Parameters,
    *,
    dt: float,
    x_boundary: str = "periodic",
    z_boundary: str = "periodic",
) -> dict[str, np.ndarray]:
    """Advance one conservative P1 finite-volume step with a Rusanov flux."""
    if dt <= 0:
        raise ValueError("P1 time step must be positive")
    number_types = state.density.shape[0]
    parameters.validate(number_types, grid)
    if state.polarization.shape != (*state.density.shape, 2):
        raise ValueError("P1 polarization has inconsistent shape")
    max_speed = float(np.max(parameters.speeds))
    if max_speed > 0:
        cfl = dt * max_speed / np.sqrt(2.0) * (1.0 / grid.dx + 1.0 / grid.dz)
        if cfl > 0.98:
            raise ValueError(f"P1 CFL condition violated ({cfl:.3f} > 0.98)")

    old_mass = p1_mass(state, grid)
    updated_density = np.empty_like(state.density)
    updated_polarization = np.empty_like(state.polarization)
    bottom_passage = np.zeros(number_types)
    force = np.broadcast_to(
        parameters.force[:, None, None, :]
        if parameters.force.shape == (number_types, 2)
        else parameters.force,
        (number_types, grid.nz, grid.nx, 2),
    )

    for root_type in range(number_types):
        speed = float(parameters.speeds[root_type])
        conserved = np.concatenate(
            (state.density[root_type, :, :, None], state.polarization[root_type]),
            axis=-1,
        )
        if x_boundary == "periodic":
            flux_x_right = _rusanov(conserved, np.roll(conserved, -1, axis=1), speed, "x")
            flux_x_left = np.roll(flux_x_right, 1, axis=1)
        else:
            right = np.roll(conserved, -1, axis=1)
            left = np.roll(conserved, 1, axis=1)
            right[:, -1] = _boundary_ghost(conserved[:, -1], x_boundary, 1)
            left[:, 0] = _boundary_ghost(conserved[:, 0], x_boundary, 1)
            flux_x_right = _rusanov(conserved, right, speed, "x")
            flux_x_left = _rusanov(left, conserved, speed, "x")

        if z_boundary == "periodic":
            flux_z_bottom = _rusanov(conserved, np.roll(conserved, -1, axis=0), speed, "z")
            flux_z_top = np.roll(flux_z_bottom, 1, axis=0)
        else:
            below = np.roll(conserved, -1, axis=0)
            above = np.roll(conserved, 1, axis=0)
            below[-1] = _boundary_ghost(conserved[-1], z_boundary, 2)
            above[0] = _boundary_ghost(conserved[0], z_boundary, 2)
            flux_z_bottom = _rusanov(conserved, below, speed, "z")
            flux_z_top = _rusanov(above, conserved, speed, "z")
            if z_boundary == "outflow":
                bottom_passage[root_type] = (
                    dt * float(np.sum(np.maximum(flux_z_bottom[-1, :, 0], 0.0))) * grid.dx
                )

        candidate = conserved - dt * (
            (flux_x_right - flux_x_left) / grid.dx
            + (flux_z_bottom - flux_z_top) / grid.dz
        )
        mortality = float(parameters.mortality[root_type])
        candidate[..., 0] *= np.exp(-mortality * dt)
        relaxation_rate = float(parameters.rotational_diffusion[root_type]) + mortality
        relaxation = np.exp(-relaxation_rate * dt)
        candidate[..., 1:] *= relaxation
        source_factor = (1.0 - relaxation) / relaxation_rate if relaxation_rate > 0 else dt
        candidate[..., 1:] += (
            0.5 * source_factor * candidate[..., 0, None] * force[root_type]
        )
        updated_density[root_type] = candidate[..., 0]
        updated_polarization[root_type] = candidate[..., 1:]

    birth_density = np.zeros_like(updated_density)
    birth_polarization = np.zeros_like(updated_polarization)
    for parent in range(number_types):
        rate = float(parameters.branching_rate[parent])
        for daughter in range(number_types):
            weight = rate * float(parameters.transition[parent, daughter])
            if weight <= 0:
                continue
            birth_density[daughter] += weight * updated_density[parent]
            birth_polarization[daughter] += (
                weight
                * float(parameters.daughter_polarization[parent])
                * updated_polarization[parent]
            )
    updated_density += dt * birth_density
    updated_polarization += dt * birth_polarization
    minimum_density = float(np.min(updated_density))
    limited_cells = 0
    for root_type in range(number_types):
        negative = updated_density[root_type] < 0.0
        limited_cells += int(np.sum(negative))
        if not np.any(negative):
            continue
        # P1 is not globally positivity preserving near closure-dependent open
        # boundaries.  Project onto the nonnegative density cone while
        # preserving the candidate zeroth moment.  The polarization is then
        # projected onto the realizability cone |m| <= n/sqrt(2).
        target_mass = float(np.sum(updated_density[root_type]))
        clipped = np.maximum(updated_density[root_type], 0.0)
        clipped_sum = float(np.sum(clipped))
        if target_mass <= 0 or clipped_sum <= 0:
            clipped.fill(0.0)
            updated_polarization[root_type].fill(0.0)
        else:
            clipped *= target_mass / clipped_sum
        updated_density[root_type] = clipped
        magnitude = np.linalg.norm(updated_polarization[root_type], axis=-1)
        limit = clipped / np.sqrt(2.0)
        factor = np.minimum(
            1.0,
            np.divide(
                limit,
                magnitude,
                out=np.ones_like(limit),
                where=magnitude > 0,
            ),
        )
        updated_polarization[root_type] *= factor[..., None]
    state.root_length_density += dt * parameters.speeds[:, None, None] * state.density
    state.root_length_density *= np.exp(-parameters.root_decay * dt)
    state.density = updated_density
    state.polarization = updated_polarization
    state.cumulative_bottom_passage += bottom_passage
    state.time += dt
    return {
        "mass_before": old_mass,
        "mass_after": p1_mass(state, grid),
        "birth_mass": np.sum(dt * birth_density, axis=(1, 2)) * grid.cell_area,
        "bottom_passage": bottom_passage,
        "minimum_density": np.asarray([minimum_density]),
        "limited_cells": np.asarray([limited_cells]),
    }
