"""Persistent active root-tip dynamics in a heterogeneous soil state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from rootfpt.multiscale.soil import SoilState


class DevelopmentState(StrEnum):
    ELONGATING = "elongating"
    DORMANT = "dormant"
    ARRESTED = "arrested"
    DEAD = "dead"


@dataclass
class TipState:
    """Complete state of one biological root tip."""

    position: np.ndarray
    orientation: float
    age: float
    root_type: str
    development_state: DevelopmentState
    radius: float
    memory: np.ndarray
    status: DevelopmentState


@dataclass(frozen=True)
class TipTraits:
    """Root-type traits in centimetres and days."""

    speed: float = 1.0
    rotational_diffusion: float = 1.0
    kappa_gravity: float = 0.0
    kappa_water: float = 0.0
    kappa_nutrient: float = 0.0
    kappa_mechanical: float = 0.0
    kappa_anisotropy: float = 0.0
    circumnutation_amplitude: float = 0.0
    circumnutation_frequency: float = 0.0
    water_half_speed: float = 0.16
    impedance_half_speed: float = 1.5
    oxygen_half_speed: float = 0.2
    nutrient_half_speed: float = 0.1
    carbon_factor: float = 1.0
    noise_impedance_sensitivity: float = 0.0
    noise_pore_sensitivity: float = 0.0

    @property
    def persistence_time(self) -> float:
        return np.inf if self.rotational_diffusion == 0 else 1.0 / self.rotational_diffusion

    @property
    def persistence_length(self) -> float:
        return self.speed * self.persistence_time


@dataclass(frozen=True)
class AgentEnsemble:
    times: np.ndarray
    positions: np.ndarray
    orientations: np.ndarray
    speeds: np.ndarray
    first_passage_times: np.ndarray
    active: np.ndarray

    @property
    def final_positions(self) -> np.ndarray:
        return self.positions[-1]


def orientation_correlation(time: np.ndarray | float, rotational_diffusion: float) -> np.ndarray:
    """Exact 2-D orientation autocorrelation for angular Brownian motion."""
    return np.exp(-rotational_diffusion * np.asarray(time, dtype=float))


def free_walk_msd(
    time: np.ndarray | float,
    speed: float,
    rotational_diffusion: float,
) -> np.ndarray:
    """Exact mean-squared displacement of a 2-D active Brownian walker."""
    values = np.asarray(time, dtype=float)
    if rotational_diffusion == 0:
        return (speed * values) ** 2
    scaled = rotational_diffusion * values
    return 2.0 * speed**2 / rotational_diffusion**2 * (scaled + np.exp(-scaled) - 1.0)


def _soil_response(
    soil: SoilState,
    positions: np.ndarray,
    angles: np.ndarray,
    time: float,
    traits: TipTraits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    perpendicular = np.stack((-np.sin(angles), np.cos(angles)), axis=-1)
    water = soil.sample("water", positions)
    impedance = soil.sample("impedance", positions)
    oxygen = soil.sample("oxygen", positions)
    nutrient = soil.sample("nutrient", positions)
    speed_factor = (
        water
        / (water + traits.water_half_speed)
        * traits.impedance_half_speed
        / (impedance + traits.impedance_half_speed)
        * oxygen
        / (oxygen + traits.oxygen_half_speed)
        * nutrient
        / (nutrient + traits.nutrient_half_speed)
        * traits.carbon_factor
    )
    speed = traits.speed * np.clip(speed_factor, 0.0, 1.0)
    force = np.zeros_like(positions)
    force[:, 1] += traits.kappa_gravity
    force += traits.kappa_water * soil.gradient("water", positions)
    force += traits.kappa_nutrient * soil.gradient("nutrient", positions)
    force -= traits.kappa_mechanical * soil.gradient("impedance", positions)
    tensor = soil.anisotropy_tensor(positions)
    tensor_direction = np.einsum("nij,nj->ni", tensor, direction)
    force += traits.kappa_anisotropy * tensor_direction
    angular_velocity = np.einsum("ni,ni->n", perpendicular, force)
    angular_velocity += traits.circumnutation_amplitude * np.sin(
        traits.circumnutation_frequency * time
    )
    anisotropy_strength = np.linalg.norm(tensor_direction - direction, axis=1)
    rotational_diffusion = traits.rotational_diffusion * (
        1.0
        + traits.noise_impedance_sensitivity * impedance
        - traits.noise_pore_sensitivity * anisotropy_strength
    )
    return speed, angular_velocity, np.clip(rotational_diffusion, 1e-8, None)


def simulate_tip_ensemble(
    *,
    number: int,
    duration: float,
    dt: float,
    traits: TipTraits,
    rng: np.random.Generator,
    soil: SoilState | None = None,
    initial_position: tuple[float, float] | np.ndarray = (0.0, 0.0),
    initial_orientation: float | np.ndarray = np.pi / 2.0,
    random_initial_orientation: bool = False,
    target_depth: float | None = None,
    boundary: str = "none",
    record_history: bool = True,
) -> AgentEnsemble:
    """Vectorized Euler–Maruyama ensemble with no translational Brownian noise."""
    if number <= 0 or duration <= 0 or dt <= 0:
        raise ValueError("number, duration, and dt must be positive")
    steps = int(np.ceil(duration / dt))
    dt = duration / steps
    times = np.linspace(0.0, duration, steps + 1)
    stored_steps = steps + 1 if record_history else 2
    positions = np.empty((stored_steps, number, 2), dtype=float)
    orientations = np.empty((stored_steps, number), dtype=float)
    speeds = np.empty((steps if record_history else 1, number), dtype=float)
    supplied_positions = np.asarray(initial_position, dtype=float)
    if supplied_positions.shape not in {(2,), (number, 2)}:
        raise ValueError("initial_position must have shape (2,) or (number, 2)")
    positions[0] = supplied_positions
    if random_initial_orientation:
        orientations[0] = rng.uniform(-np.pi, np.pi, size=number)
    else:
        initial_angles = np.asarray(initial_orientation, dtype=float)
        if initial_angles.shape not in {(), (number,)}:
            raise ValueError("initial_orientation must be scalar or have shape (number,)")
        orientations[0] = initial_angles
    current_positions = positions[0].copy()
    current_angles = orientations[0].copy()
    speed_sum = np.zeros(number)
    first_passage = np.full(number, np.nan)
    active = np.ones(number, dtype=bool)

    for step in range(steps):
        angles = current_angles
        current = current_positions
        if soil is None:
            local_speed = np.full(number, traits.speed)
            angular_velocity = np.full(
                number,
                traits.circumnutation_amplitude
                * np.sin(traits.circumnutation_frequency * times[step]),
            )
            local_dr = np.full(number, traits.rotational_diffusion)
        else:
            local_speed, angular_velocity, local_dr = _soil_response(
                soil, current, angles, times[step], traits
            )
        local_speed = local_speed * active
        if record_history:
            speeds[step] = local_speed
        else:
            speed_sum += local_speed
        noise = np.sqrt(2.0 * local_dr * dt) * rng.normal(size=number)
        updated_angles = (angles + angular_velocity * dt + noise + np.pi) % (2.0 * np.pi) - np.pi
        displacement = (
            local_speed[:, None]
            * np.stack((np.cos(updated_angles), np.sin(updated_angles)), axis=-1)
            * dt
        )
        updated_positions = current + displacement
        if soil is not None and boundary != "none":
            x0, x1 = soil.grid.x_limits
            z0, z1 = soil.grid.z_limits
            if boundary == "periodic":
                updated_positions[:, 0] = x0 + (updated_positions[:, 0] - x0) % (x1 - x0)
                updated_positions[:, 1] = z0 + (updated_positions[:, 1] - z0) % (z1 - z0)
            elif boundary == "reflect":
                left = updated_positions[:, 0] < x0
                right = updated_positions[:, 0] > x1
                top = updated_positions[:, 1] < z0
                bottom = updated_positions[:, 1] > z1
                updated_positions[left, 0] = 2.0 * x0 - updated_positions[left, 0]
                updated_positions[right, 0] = 2.0 * x1 - updated_positions[right, 0]
                updated_positions[top, 1] = 2.0 * z0 - updated_positions[top, 1]
                updated_positions[bottom, 1] = 2.0 * z1 - updated_positions[bottom, 1]
                updated_angles[left | right] = np.pi - updated_angles[left | right]
                updated_angles[top | bottom] = -updated_angles[top | bottom]
            else:
                raise ValueError(f"unknown boundary {boundary!r}")
        if target_depth is not None:
            crossed = active & (updated_positions[:, 1] >= target_depth)
            first_passage[crossed] = times[step + 1]
            active[crossed] = False
        current_positions = updated_positions
        current_angles = updated_angles
        if record_history:
            positions[step + 1] = updated_positions
            orientations[step + 1] = updated_angles
    if not record_history:
        positions[1] = current_positions
        orientations[1] = current_angles
        speeds[0] = speed_sum / steps
        times = np.asarray([0.0, duration])
    return AgentEnsemble(times, positions, orientations, speeds, first_passage, active)
