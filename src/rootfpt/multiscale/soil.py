"""Unit-documented heterogeneous soil fields on a regular two-dimensional grid."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter

FIELD_UNITS = {
    "water": "cm3 cm-3",
    "pressure_head": "cm",
    "hydraulic_conductivity": "cm d-1",
    "impedance": "MPa",
    "porosity": "cm3 cm-3",
    "anisotropy_xx": "1",
    "anisotropy_xz": "1",
    "anisotropy_zz": "1",
    "nutrient": "relative concentration",
    "oxygen": "relative concentration",
}


@dataclass(frozen=True)
class Grid2D:
    """Cell-centred grid with horizontal coordinate ``x`` and positive depth ``z``."""

    nx: int
    nz: int
    x_limits: tuple[float, float]
    z_limits: tuple[float, float]

    def __post_init__(self) -> None:
        if self.nx < 3 or self.nz < 3:
            raise ValueError("grid requires at least three cells per direction")
        if self.x_limits[1] <= self.x_limits[0] or self.z_limits[1] <= self.z_limits[0]:
            raise ValueError("grid limits must increase")

    @property
    def dx(self) -> float:
        return (self.x_limits[1] - self.x_limits[0]) / self.nx

    @property
    def dz(self) -> float:
        return (self.z_limits[1] - self.z_limits[0]) / self.nz

    @property
    def x(self) -> np.ndarray:
        return self.x_limits[0] + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def z(self) -> np.ndarray:
        return self.z_limits[0] + (np.arange(self.nz) + 0.5) * self.dz

    @property
    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.z)

    @property
    def cell_area(self) -> float:
        return self.dx * self.dz


def _constant(grid: Grid2D, value: float) -> np.ndarray:
    return np.full((grid.nz, grid.nx), value, dtype=float)


def _standardize(values: np.ndarray, mean: float, standard_deviation: float) -> np.ndarray:
    centred = values - float(np.mean(values))
    scale = float(np.std(centred))
    if scale <= 1e-14:
        return np.full_like(values, mean)
    return mean + standard_deviation * centred / scale


def matern_random_field(
    grid: Grid2D,
    *,
    rng: np.random.Generator,
    mean: float,
    standard_deviation: float,
    correlation_length: float,
    smoothness: float = 1.5,
) -> np.ndarray:
    """Generate a periodic Matérn-like Gaussian field by spectral filtering.

    The spectrum is proportional to
    ``(kappa**2 + |k|**2)**(-(nu + d/2))`` in two dimensions. The requested
    mean and pointwise standard deviation are imposed after filtering.
    """
    if standard_deviation < 0 or correlation_length <= 0 or smoothness <= 0:
        raise ValueError("invalid Matérn parameters")
    white = rng.normal(size=(grid.nz, grid.nx))
    kx = 2.0 * np.pi * np.fft.fftfreq(grid.nx, d=grid.dx)
    kz = 2.0 * np.pi * np.fft.fftfreq(grid.nz, d=grid.dz)
    wave_x, wave_z = np.meshgrid(kx, kz)
    kappa = np.sqrt(2.0 * smoothness) / correlation_length
    spectral_amplitude = (kappa**2 + wave_x**2 + wave_z**2) ** (-0.5 * (smoothness + 1.0))
    spectral_amplitude[0, 0] = 0.0
    values = np.fft.ifft2(np.fft.fft2(white) * spectral_amplitude).real
    return _standardize(values, mean, standard_deviation)


@dataclass(frozen=True)
class SoilState:
    """Modular soil state.

    Length is in centimetres and time in days. Field units are exposed through
    :data:`FIELD_UNITS`; values are never silently rescaled.
    """

    grid: Grid2D
    water: np.ndarray
    pressure_head: np.ndarray
    hydraulic_conductivity: np.ndarray
    impedance: np.ndarray
    porosity: np.ndarray
    anisotropy_xx: np.ndarray
    anisotropy_xz: np.ndarray
    anisotropy_zz: np.ndarray
    nutrient: np.ndarray
    oxygen: np.ndarray
    correlation_length: float
    description: str

    def __post_init__(self) -> None:
        target = (self.grid.nz, self.grid.nx)
        for name in FIELD_UNITS:
            values = np.asarray(getattr(self, name))
            if values.shape != target or not np.isfinite(values).all():
                raise ValueError(f"{name} must be a finite array with shape {target}")
        if self.correlation_length < 0:
            raise ValueError("correlation length cannot be negative")
        if np.any(self.hydraulic_conductivity < 0):
            raise ValueError("hydraulic conductivity cannot be negative")
        if np.any((self.porosity <= 0) | (self.porosity > 1)):
            raise ValueError("porosity must lie in (0, 1]")

    @classmethod
    def homogeneous(
        cls,
        grid: Grid2D,
        *,
        water: float = 0.28,
        pressure_head: float = -200.0,
        hydraulic_conductivity: float = 10.0,
        impedance: float = 0.5,
        porosity: float = 0.42,
        nutrient: float = 0.5,
        oxygen: float = 1.0,
    ) -> SoilState:
        return cls(
            grid=grid,
            water=_constant(grid, water),
            pressure_head=_constant(grid, pressure_head),
            hydraulic_conductivity=_constant(grid, hydraulic_conductivity),
            impedance=_constant(grid, impedance),
            porosity=_constant(grid, porosity),
            anisotropy_xx=_constant(grid, 1.0),
            anisotropy_xz=_constant(grid, 0.0),
            anisotropy_zz=_constant(grid, 1.0),
            nutrient=_constant(grid, nutrient),
            oxygen=_constant(grid, oxygen),
            correlation_length=0.0,
            description="homogeneous",
        )

    @classmethod
    def matern(
        cls,
        grid: Grid2D,
        *,
        rng: np.random.Generator,
        correlation_length: float,
        smoothness: float = 1.5,
        cross_correlation: float = -0.55,
    ) -> SoilState:
        """Generate cross-correlated water and impedance Matérn fields."""
        if not -0.99 < cross_correlation < 0.99:
            raise ValueError("cross-correlation must be strictly between -0.99 and 0.99")
        first = matern_random_field(
            grid,
            rng=rng,
            mean=0.0,
            standard_deviation=1.0,
            correlation_length=correlation_length,
            smoothness=smoothness,
        )
        second = matern_random_field(
            grid,
            rng=rng,
            mean=0.0,
            standard_deviation=1.0,
            correlation_length=correlation_length,
            smoothness=smoothness,
        )
        correlated = cross_correlation * first + np.sqrt(1.0 - cross_correlation**2) * second
        correlated = _standardize(correlated, 0.0, 1.0)
        water = np.clip(0.28 + 0.055 * first, 0.08, 0.46)
        impedance = np.clip(0.8 + 0.32 * correlated, 0.05, 2.5)
        nutrient_noise = matern_random_field(
            grid,
            rng=rng,
            mean=0.5,
            standard_deviation=0.14,
            correlation_length=correlation_length,
            smoothness=smoothness,
        )
        pressure = -200.0 - 1800.0 * np.clip(0.28 - water, 0.0, None)
        conductivity = 10.0 * np.clip(water / 0.28, 0.05, 1.8) ** 3
        porosity = np.clip(0.46 - 0.08 * impedance, 0.22, 0.48)
        return cls(
            grid,
            water,
            pressure,
            conductivity,
            impedance,
            porosity,
            _constant(grid, 1.0),
            _constant(grid, 0.0),
            _constant(grid, 1.0),
            np.clip(nutrient_noise, 0.0, 1.0),
            np.clip(0.75 + 0.5 * (porosity - 0.35), 0.2, 1.0),
            correlation_length,
            f"cross-correlated Matérn nu={smoothness:g}",
        )

    def layered(
        self,
        *,
        interface_depth: float,
        lower_impedance: float | None = None,
        lower_water: float | None = None,
        anisotropy_angle: float | None = None,
        anisotropy_strength: float = 0.0,
    ) -> SoilState:
        _, depth = self.grid.mesh
        mask = depth >= interface_depth
        impedance = self.impedance.copy()
        water = self.water.copy()
        if lower_impedance is not None:
            impedance[mask] = lower_impedance
        if lower_water is not None:
            water[mask] = lower_water
        axx = self.anisotropy_xx.copy()
        axz = self.anisotropy_xz.copy()
        azz = self.anisotropy_zz.copy()
        if anisotropy_angle is not None:
            c, s = np.cos(anisotropy_angle), np.sin(anisotropy_angle)
            axx[mask] = 1.0 + anisotropy_strength * c * c
            axz[mask] = anisotropy_strength * c * s
            azz[mask] = 1.0 + anisotropy_strength * s * s
        pressure = -200.0 - 1800.0 * np.clip(0.28 - water, 0.0, None)
        return replace(
            self,
            water=water,
            pressure_head=pressure,
            impedance=impedance,
            anisotropy_xx=axx,
            anisotropy_xz=axz,
            anisotropy_zz=azz,
            description=f"{self.description}; layer at z={interface_depth:g} cm",
        )

    def with_compacted_lens(
        self,
        *,
        centre: tuple[float, float],
        radii: tuple[float, float],
        impedance: float,
    ) -> SoilState:
        horizontal, depth = self.grid.mesh
        mask = ((horizontal - centre[0]) / radii[0]) ** 2 + (
            (depth - centre[1]) / radii[1]
        ) ** 2 <= 1.0
        values = self.impedance.copy()
        values[mask] = impedance
        return replace(self, impedance=values, description=f"{self.description}; compacted lens")

    def with_crack(
        self,
        *,
        angle: float,
        offset: float = 0.0,
        width: float = 0.2,
        strength: float = 5.0,
    ) -> SoilState:
        horizontal, depth = self.grid.mesh
        direction = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        distance = np.abs(normal[0] * horizontal + normal[1] * depth - offset)
        mask = distance <= width / 2.0
        axx, axz, azz = (
            self.anisotropy_xx.copy(),
            self.anisotropy_xz.copy(),
            self.anisotropy_zz.copy(),
        )
        axx[mask] = 1.0 + strength * direction[0] ** 2
        axz[mask] = strength * direction[0] * direction[1]
        azz[mask] = 1.0 + strength * direction[1] ** 2
        impedance = self.impedance.copy()
        impedance[mask] *= 0.15
        porosity = self.porosity.copy()
        porosity[mask] = np.minimum(0.65, porosity[mask] + 0.15)
        return replace(
            self,
            impedance=impedance,
            porosity=porosity,
            anisotropy_xx=axx,
            anisotropy_xz=axz,
            anisotropy_zz=azz,
            description=f"{self.description}; explicit crack",
        )

    def _fractional_indices(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(positions, dtype=float)
        ix = (points[..., 0] - self.grid.x_limits[0]) / self.grid.dx - 0.5
        iz = (points[..., 1] - self.grid.z_limits[0]) / self.grid.dz - 0.5
        return (
            np.clip(ix, 0.0, self.grid.nx - 1.000001),
            np.clip(iz, 0.0, self.grid.nz - 1.000001),
        )

    def sample(self, name: str, positions: np.ndarray) -> np.ndarray:
        if name not in FIELD_UNITS:
            raise KeyError(f"unknown soil field {name!r}")
        values = np.asarray(getattr(self, name))
        ix, iz = self._fractional_indices(positions)
        x0, z0 = np.floor(ix).astype(int), np.floor(iz).astype(int)
        x1 = np.minimum(x0 + 1, self.grid.nx - 1)
        z1 = np.minimum(z0 + 1, self.grid.nz - 1)
        tx, tz = ix - x0, iz - z0
        return (
            (1.0 - tx) * (1.0 - tz) * values[z0, x0]
            + tx * (1.0 - tz) * values[z0, x1]
            + (1.0 - tx) * tz * values[z1, x0]
            + tx * tz * values[z1, x1]
        )

    def gradient(self, name: str, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(getattr(self, name))
        derivative_z, derivative_x = np.gradient(values, self.grid.dz, self.grid.dx)
        copy = replace(self, **{name: derivative_x})
        grad_x = copy.sample(name, positions)
        copy = replace(self, **{name: derivative_z})
        grad_z = copy.sample(name, positions)
        return np.stack((grad_x, grad_z), axis=-1)

    def anisotropy_tensor(self, positions: np.ndarray) -> np.ndarray:
        xx = self.sample("anisotropy_xx", positions)
        xz = self.sample("anisotropy_xz", positions)
        zz = self.sample("anisotropy_zz", positions)
        tensor = np.empty(np.shape(xx) + (2, 2), dtype=float)
        tensor[..., 0, 0] = xx
        tensor[..., 0, 1] = xz
        tensor[..., 1, 0] = xz
        tensor[..., 1, 1] = zz
        return tensor

    def empirical_cross_correlation(self) -> float:
        return float(np.corrcoef(self.water.ravel(), self.impedance.ravel())[0, 1])

    def smoothed(self, sigma_cells: float) -> SoilState:
        """Return a diagnostic refinement control with every scalar field smoothed."""
        updates = {
            name: gaussian_filter(np.asarray(getattr(self, name)), sigma_cells, mode="wrap")
            for name in FIELD_UNITS
        }
        return replace(self, **updates, description=f"{self.description}; smoothed")
