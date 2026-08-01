"""Orientation- and age-resolved kinetic solver and its derived diffusion limit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rootfpt.multiscale.soil import Grid2D


@dataclass(frozen=True)
class KineticGrid:
    spatial: Grid2D
    ntheta: int
    nage: int
    age_step: float

    def __post_init__(self) -> None:
        if self.ntheta < 8 or self.nage < 2 or self.age_step <= 0:
            raise ValueError("invalid kinetic grid")

    @property
    def dtheta(self) -> float:
        return 2.0 * np.pi / self.ntheta

    @property
    def theta(self) -> np.ndarray:
        return -np.pi + (np.arange(self.ntheta) + 0.5) * self.dtheta

    @property
    def age(self) -> np.ndarray:
        return (np.arange(self.nage) + 0.5) * self.age_step


@dataclass
class KineticState:
    """Density has axes ``type, age, z, x, orientation``."""

    density: np.ndarray
    root_length_density: np.ndarray
    cumulative_bottom_passage: np.ndarray
    time: float = 0.0


@dataclass(frozen=True)
class KineticParameters:
    speeds: np.ndarray
    rotational_diffusion: np.ndarray
    mortality: np.ndarray
    branching_rate: np.ndarray
    transition: np.ndarray
    branch_angle: np.ndarray
    branch_concentration: np.ndarray
    root_decay: float = 0.0

    def validate(self, number_types: int) -> None:
        vectors = (
            self.speeds,
            self.rotational_diffusion,
            self.mortality,
            self.branching_rate,
            self.branch_angle,
            self.branch_concentration,
        )
        if any(np.shape(vector) != (number_types,) for vector in vectors):
            raise ValueError("kinetic type vectors have inconsistent shape")
        if np.shape(self.transition) != (number_types, number_types):
            raise ValueError("transition matrix has inconsistent shape")
        if np.any(np.asarray(vectors[:4]) < 0) or self.root_decay < 0:
            raise ValueError("rates and speeds must be nonnegative")
        if np.any(self.transition < 0):
            raise ValueError("transition probabilities must be nonnegative")
        row_sums = np.sum(self.transition, axis=1)
        active = self.branching_rate > 0
        if np.any(np.abs(row_sums[active] - 1.0) > 1e-10):
            raise ValueError("active branching transition rows must sum to one")


@dataclass(frozen=True)
class DiffusionCoefficients:
    diffusivity: np.ndarray
    drift: np.ndarray


def derived_diffusion_coefficients(
    speeds: np.ndarray,
    rotational_diffusion: np.ndarray,
    force: np.ndarray | None = None,
) -> DiffusionCoefficients:
    """Fast-reorientation closure in 2-D.

    With ``dtheta = p_perp·F dt + sqrt(2 Dr)dW``, the leading moments give
    ``D_eff = v²/(2 Dr)`` and weak-bias drift ``u = v F/(2 Dr)``.
    """
    speeds = np.asarray(speeds, dtype=float)
    rotational_diffusion = np.asarray(rotational_diffusion, dtype=float)
    if np.any(rotational_diffusion <= 0) or np.any(speeds < 0):
        raise ValueError("positive rotational diffusion and nonnegative speed required")
    diffusivity = speeds**2 / (2.0 * rotational_diffusion)
    if force is None:
        drift = np.zeros((len(speeds), 2))
    else:
        force = np.asarray(force, dtype=float)
        if force.shape != (len(speeds), 2):
            raise ValueError("force must have shape (type, 2)")
        drift = speeds[:, None] * force / (2.0 * rotational_diffusion[:, None])
    return DiffusionCoefficients(diffusivity, drift)


def initialize_kinetic_state(
    grid: KineticGrid,
    *,
    number_types: int,
    count_by_type: np.ndarray,
    centre: tuple[float, float],
    spatial_sigma: float,
    mean_angle: float | None = None,
    angle_concentration: float = 0.0,
) -> KineticState:
    counts = np.asarray(count_by_type, dtype=float)
    if counts.shape != (number_types,) or np.any(counts < 0) or spatial_sigma <= 0:
        raise ValueError("invalid kinetic initialization")
    horizontal, depth = grid.spatial.mesh
    spatial = np.exp(
        -0.5
        * (
            ((horizontal - centre[0]) / spatial_sigma) ** 2
            + ((depth - centre[1]) / spatial_sigma) ** 2
        )
    )
    spatial /= float(np.sum(spatial) * grid.spatial.cell_area)
    if mean_angle is None or angle_concentration == 0:
        angular = np.full(grid.ntheta, 1.0 / (2.0 * np.pi))
    else:
        angular = np.exp(angle_concentration * np.cos(grid.theta - mean_angle))
        angular /= float(np.sum(angular) * grid.dtheta)
    density = np.zeros(
        (
            number_types,
            grid.nage,
            grid.spatial.nz,
            grid.spatial.nx,
            grid.ntheta,
        )
    )
    for root_type, count in enumerate(counts):
        density[root_type, 0] = count * spatial[:, :, None] * angular[None, None, :] / grid.age_step
    return KineticState(
        density,
        np.zeros((number_types, grid.spatial.nz, grid.spatial.nx)),
        np.zeros(number_types),
    )


def kinetic_tip_density(state: KineticState, grid: KineticGrid) -> np.ndarray:
    return np.sum(state.density, axis=(1, 4)) * grid.age_step * grid.dtheta


def kinetic_orientation_density(state: KineticState, grid: KineticGrid) -> np.ndarray:
    return np.sum(state.density, axis=(1, 2, 3)) * grid.age_step * grid.spatial.cell_area


def kinetic_mass(state: KineticState, grid: KineticGrid) -> np.ndarray:
    return (
        np.sum(state.density, axis=(1, 2, 3, 4))
        * grid.age_step
        * grid.dtheta
        * grid.spatial.cell_area
    )


def _upwind_periodic(
    values: np.ndarray,
    velocity: np.ndarray,
    dt_over_dx: float,
    axis: int,
) -> np.ndarray:
    shape = [1] * values.ndim
    shape[-1] = len(velocity)
    speed = velocity.reshape(shape)
    backward = values - np.roll(values, 1, axis=axis)
    forward = np.roll(values, -1, axis=axis) - values
    advection = np.maximum(speed, 0.0) * backward + np.minimum(speed, 0.0) * forward
    return values - dt_over_dx * advection


def _upwind_absorbing_z(
    values: np.ndarray,
    velocity: np.ndarray,
    dt_over_dz: float,
    z_axis: int,
) -> np.ndarray:
    shape = [1] * values.ndim
    shape[-1] = len(velocity)
    speed = velocity.reshape(shape)
    previous = np.roll(values, 1, axis=z_axis)
    following = np.roll(values, -1, axis=z_axis)
    index_top = [slice(None)] * values.ndim
    index_top[z_axis] = 0
    previous[tuple(index_top)] = 0.0
    index_bottom = [slice(None)] * values.ndim
    index_bottom[z_axis] = -1
    following[tuple(index_bottom)] = 0.0
    backward = values - previous
    forward = following - values
    return values - dt_over_dz * (
        np.maximum(speed, 0.0) * backward + np.minimum(speed, 0.0) * forward
    )


def _orientation_kernel(grid: KineticGrid, angle: float, concentration: float) -> np.ndarray:
    offsets = np.arange(grid.ntheta) * grid.dtheta
    if concentration <= 1e-12:
        return np.full(grid.ntheta, 1.0 / (2.0 * np.pi))
    kernel = 0.5 * (
        np.exp(concentration * np.cos(offsets - angle))
        + np.exp(concentration * np.cos(offsets + angle))
    )
    return kernel / float(np.sum(kernel) * grid.dtheta)


def step_kinetic(
    state: KineticState,
    grid: KineticGrid,
    parameters: KineticParameters,
    *,
    dt: float,
    angular_velocity: np.ndarray | None = None,
    z_boundary: str = "periodic",
    terminal_age_bin: bool = True,
    spatial_method: str = "upwind",
) -> dict[str, np.ndarray]:
    """Advance one conservative cohort step.

    ``dt`` must equal the age-bin width. Spatial streaming and angular drift
    use positive finite-volume upwinding; rotational diffusion is an exact
    periodic Fourier semigroup.
    """
    density = state.density
    number_types = density.shape[0]
    parameters.validate(number_types)
    if not np.isclose(dt, grid.age_step):
        raise ValueError("cohort solver requires dt equal to age_step")
    theta = grid.theta
    bottom_flux = np.zeros(number_types)
    transported = density.copy()
    for root_type in range(number_types):
        vx = parameters.speeds[root_type] * np.cos(theta)
        vz = parameters.speeds[root_type] * np.sin(theta)
        if spatial_method == "spectral":
            if z_boundary != "periodic":
                raise ValueError("spectral streaming requires periodic boundaries")
            wave_x = 2.0 * np.pi * np.fft.fftfreq(grid.spatial.nx, d=grid.spatial.dx)
            wave_z = 2.0 * np.pi * np.fft.fftfreq(grid.spatial.nz, d=grid.spatial.dz)
            kx, kz = np.meshgrid(wave_x, wave_z)
            phase = np.exp(
                -1j * dt * (kx[:, :, None] * vx[None, None, :] + kz[:, :, None] * vz[None, None, :])
            )
            spectrum = np.fft.fft2(transported[root_type], axes=(1, 2))
            transported[root_type] = np.fft.ifft2(
                spectrum * phase[None, :, :, :],
                axes=(1, 2),
            ).real
        elif spatial_method == "upwind":
            transported[root_type] = _upwind_periodic(
                transported[root_type],
                vx,
                dt / grid.spatial.dx,
                axis=2,
            )
            if z_boundary == "periodic":
                transported[root_type] = _upwind_periodic(
                    transported[root_type],
                    vz,
                    dt / grid.spatial.dz,
                    axis=1,
                )
            elif z_boundary == "absorbing":
                boundary_density = transported[root_type, :, -1, :, :]
                bottom_flux[root_type] = (
                    dt
                    * float(np.sum(boundary_density * np.maximum(vz[None, None, :], 0.0)))
                    * grid.age_step
                    * grid.dtheta
                    * grid.spatial.dx
                )
                transported[root_type] = _upwind_absorbing_z(
                    transported[root_type],
                    vz,
                    dt / grid.spatial.dz,
                    z_axis=1,
                )
            else:
                raise ValueError(f"unknown z boundary {z_boundary!r}")
        else:
            raise ValueError(f"unknown spatial method {spatial_method!r}")

        target_sum = float(np.sum(transported[root_type])) * np.exp(
            -parameters.mortality[root_type] * dt
        )
        if angular_velocity is not None:
            omega = np.asarray(angular_velocity[root_type])
            if omega.shape != (
                grid.spatial.nz,
                grid.spatial.nx,
                grid.ntheta,
            ):
                raise ValueError("angular velocity has inconsistent shape")
            positive_flux = np.maximum(omega, 0.0) * transported[root_type]
            negative_flux = np.minimum(omega, 0.0) * transported[root_type]
            divergence = (
                positive_flux
                - np.roll(positive_flux, 1, axis=-1)
                + np.roll(negative_flux, -1, axis=-1)
                - negative_flux
            ) / grid.dtheta
            transported[root_type] -= dt * divergence
        modes = np.fft.fftfreq(grid.ntheta, d=grid.dtheta) * 2.0 * np.pi
        spectrum = np.fft.fft(transported[root_type], axis=-1)
        spectrum *= np.exp(-parameters.rotational_diffusion[root_type] * modes**2 * dt)
        transported[root_type] = np.fft.ifft(spectrum, axis=-1).real
        transported[root_type] *= np.exp(-parameters.mortality[root_type] * dt)
        transported[root_type] = np.maximum(transported[root_type], 0.0)
        numerical_sum = float(np.sum(transported[root_type]))
        if numerical_sum > 0:
            transported[root_type] *= target_sum / numerical_sum
    transported = np.maximum(transported, 0.0)

    births = np.zeros((number_types, grid.spatial.nz, grid.spatial.nx, grid.ntheta))
    for parent in range(number_types):
        if parameters.branching_rate[parent] <= 0:
            continue
        parent_orientation = (
            np.sum(transported[parent], axis=0) * grid.age_step * parameters.branching_rate[parent]
        )
        kernel = _orientation_kernel(
            grid,
            parameters.branch_angle[parent],
            parameters.branch_concentration[parent],
        )
        daughter_orientation = (
            np.fft.ifft(
                np.fft.fft(parent_orientation, axis=-1) * np.fft.fft(kernel, axis=-1),
                axis=-1,
            ).real
            * grid.dtheta
        )
        for daughter in range(number_types):
            births[daughter] += parameters.transition[parent, daughter] * daughter_orientation

    aged = np.zeros_like(transported)
    aged[:, 1:] = transported[:, :-1]
    if terminal_age_bin:
        aged[:, -1] += transported[:, -1]
    aged[:, 0] = births
    state.density = aged
    tip_density_before_age = np.sum(transported, axis=(1, 4)) * grid.age_step * grid.dtheta
    state.root_length_density += dt * parameters.speeds[:, None, None] * tip_density_before_age
    state.root_length_density *= np.exp(-parameters.root_decay * dt)
    state.cumulative_bottom_passage += bottom_flux
    state.time += dt
    return {
        "births": np.sum(births, axis=(1, 2, 3)) * grid.spatial.cell_area * grid.dtheta * dt,
        "bottom_passage": bottom_flux,
        "mass": kinetic_mass(state, grid),
    }


@dataclass
class DiffusionState:
    density: np.ndarray
    root_length_density: np.ndarray
    time: float = 0.0


def initialize_diffusion_state(
    grid: Grid2D,
    *,
    count_by_type: np.ndarray,
    centre: tuple[float, float],
    spatial_sigma: float,
) -> DiffusionState:
    horizontal, depth = grid.mesh
    spatial = np.exp(
        -0.5
        * (
            ((horizontal - centre[0]) / spatial_sigma) ** 2
            + ((depth - centre[1]) / spatial_sigma) ** 2
        )
    )
    spatial /= float(np.sum(spatial) * grid.cell_area)
    density = np.asarray(count_by_type)[:, None, None] * spatial[None, :, :]
    return DiffusionState(density, np.zeros_like(density))


def step_diffusion(
    state: DiffusionState,
    grid: Grid2D,
    *,
    dt: float,
    diffusivity: np.ndarray,
    drift: np.ndarray,
    birth: np.ndarray,
    mortality: np.ndarray,
    speeds: np.ndarray,
    root_decay: float = 0.0,
) -> None:
    """Periodic conservative drift-diffusion update."""
    number_types = state.density.shape[0]
    diffusivity_values = np.asarray(diffusivity, dtype=float)
    if diffusivity_values.shape == (number_types,):
        diffusivity_values = diffusivity_values[:, None, None]
    diffusivity = np.broadcast_to(diffusivity_values, (number_types, grid.nz, grid.nx))
    if drift.shape == (number_types, 2):
        drift = np.broadcast_to(drift[:, None, None, :], (number_types, grid.nz, grid.nx, 2))
    if drift.shape != (number_types, grid.nz, grid.nx, 2):
        raise ValueError("drift has inconsistent shape")
    updated = np.empty_like(state.density)
    for root_type in range(number_types):
        values = state.density[root_type]
        laplacian = (
            np.roll(values, 1, axis=1) - 2.0 * values + np.roll(values, -1, axis=1)
        ) / grid.dx**2 + (
            np.roll(values, 1, axis=0) - 2.0 * values + np.roll(values, -1, axis=0)
        ) / grid.dz**2
        candidate = values + dt * diffusivity[root_type] * laplacian
        vx = drift[root_type, :, :, 0]
        vz = drift[root_type, :, :, 1]
        flux_x = np.maximum(vx, 0.0) * values + np.minimum(vx, 0.0) * np.roll(values, -1, axis=1)
        flux_z = np.maximum(vz, 0.0) * values + np.minimum(vz, 0.0) * np.roll(values, -1, axis=0)
        candidate -= dt * (
            (flux_x - np.roll(flux_x, 1, axis=1)) / grid.dx
            + (flux_z - np.roll(flux_z, 1, axis=0)) / grid.dz
        )
        candidate += dt * (birth[root_type] - mortality[root_type]) * values
        updated[root_type] = np.maximum(candidate, 0.0)
    state.root_length_density += dt * speeds[:, None, None] * state.density
    state.root_length_density *= np.exp(-root_decay * dt)
    state.density = updated
    state.time += dt


def evolve_diffusion_constant_periodic(
    state: DiffusionState,
    grid: Grid2D,
    *,
    duration: float,
    diffusivity: np.ndarray,
    drift: np.ndarray,
    net_growth: np.ndarray,
    speeds: np.ndarray,
) -> None:
    """Exact Fourier evolution for constant periodic drift-diffusion coefficients."""
    number_types = state.density.shape[0]
    diffusivity = np.asarray(diffusivity, dtype=float)
    drift = np.asarray(drift, dtype=float)
    net_growth = np.asarray(net_growth, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    if (
        duration <= 0
        or diffusivity.shape != (number_types,)
        or drift.shape != (number_types, 2)
        or net_growth.shape != (number_types,)
        or speeds.shape != (number_types,)
    ):
        raise ValueError("invalid constant diffusion evolution")
    wave_x = 2.0 * np.pi * np.fft.fftfreq(grid.nx, d=grid.dx)
    wave_z = 2.0 * np.pi * np.fft.fftfreq(grid.nz, d=grid.dz)
    kx, kz = np.meshgrid(wave_x, wave_z)
    initial = state.density.copy()
    for root_type in range(number_types):
        exponent = (
            -diffusivity[root_type] * (kx**2 + kz**2)
            - 1j * (drift[root_type, 0] * kx + drift[root_type, 1] * kz)
            + net_growth[root_type]
        )
        spectrum = np.fft.fft2(initial[root_type])
        state.density[root_type] = np.fft.ifft2(spectrum * np.exp(exponent * duration)).real
        state.density[root_type] = np.maximum(state.density[root_type], 0.0)
        integral_factor = np.empty_like(exponent, dtype=complex)
        regular = np.abs(exponent) > 1e-12
        integral_factor[regular] = np.expm1(exponent[regular] * duration) / exponent[regular]
        integral_factor[~regular] = duration
        accumulated = np.fft.ifft2(speeds[root_type] * spectrum * integral_factor).real
        state.root_length_density[root_type] += np.maximum(
            accumulated,
            0.0,
        )
    state.time += duration


def normalized_l1(first: np.ndarray, second: np.ndarray, cell_measure: float = 1.0) -> float:
    denominator = max(
        float(np.sum(np.abs(first)) * cell_measure),
        float(np.sum(np.abs(second)) * cell_measure),
        1e-15,
    )
    return float(np.sum(np.abs(first - second)) * cell_measure / denominator)


def density_moments(density: np.ndarray, grid: Grid2D) -> dict[str, float]:
    mass = float(np.sum(density) * grid.cell_area)
    if mass <= 0:
        return {"mass": 0.0, "mean_x": np.nan, "mean_z": np.nan, "variance": np.nan}
    horizontal, depth = grid.mesh
    mean_x = float(np.sum(horizontal * density) * grid.cell_area / mass)
    mean_z = float(np.sum(depth * density) * grid.cell_area / mass)
    variance = float(
        np.sum(((horizontal - mean_x) ** 2 + (depth - mean_z) ** 2) * density)
        * grid.cell_area
        / mass
    )
    return {"mass": mass, "mean_x": mean_x, "mean_z": mean_z, "variance": variance}
