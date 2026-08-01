"""Developmentally explicit lateral-site placement and emergence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from rootfpt.tips import TipState, wrap_angle


class LateralStatus(StrEnum):
    SCHEDULED = "scheduled"
    DORMANT = "dormant"
    ABORTED = "aborted"
    EMERGED = "emerged"


@dataclass
class LateralSite:
    site_id: int
    parent_tip_id: int
    parent_segment_id: int
    parent_arc_length: float
    position: np.ndarray
    parent_orientation: float
    root_order: int
    initiation_time: float
    emergence_time: float
    daughter_angle: float
    status: LateralStatus


@dataclass
class LateralDevelopment:
    mean_spacing: float
    spacing_shape: float
    mean_emergence_delay: float
    abortion_probability: float
    dormancy_probability: float
    daughter_angle_mean: float
    daughter_angle_sd: float
    maximum_order: int
    rng: np.random.Generator
    sites: list[LateralSite] = field(default_factory=list)
    _next_site_id: int = 0

    def __post_init__(self) -> None:
        if self.mean_spacing <= 0 or self.spacing_shape <= 0:
            raise ValueError("branch spacing parameters must be positive")
        if self.mean_emergence_delay < 0:
            raise ValueError("emergence delay must be nonnegative")
        if self.abortion_probability + self.dormancy_probability > 1.0:
            raise ValueError("abortion and dormancy probabilities exceed one")

    def sample_spacing(self) -> float:
        scale = self.mean_spacing / self.spacing_shape
        return float(self.rng.gamma(self.spacing_shape, scale))

    def initialize_tip(self, tip: TipState) -> None:
        tip.next_lateral_arc = tip.arc_length + self.sample_spacing()

    def register_growth(
        self,
        *,
        tip: TipState,
        segment_id: int,
        start_position: np.ndarray,
        end_position: np.ndarray,
        previous_arc_length: float,
        time: float,
    ) -> list[LateralSite]:
        """Place every potential site crossed by the newly grown segment."""
        new_sites: list[LateralSite] = []
        grown = tip.arc_length - previous_arc_length
        while (
            tip.root_order < self.maximum_order
            and tip.next_lateral_arc <= tip.arc_length + 1e-12
        ):
            fraction = (
                (tip.next_lateral_arc - previous_arc_length) / grown if grown > 0 else 1.0
            )
            fraction = float(np.clip(fraction, 0.0, 1.0))
            position = start_position + fraction * (end_position - start_position)
            outcome = float(self.rng.random())
            if outcome < self.abortion_probability:
                status = LateralStatus.ABORTED
            elif outcome < self.abortion_probability + self.dormancy_probability:
                status = LateralStatus.DORMANT
            else:
                status = LateralStatus.SCHEDULED
            delay = (
                float(self.rng.gamma(2.0, self.mean_emergence_delay / 2.0))
                if self.mean_emergence_delay > 0
                else 0.0
            )
            sign = -1.0 if self.rng.random() < 0.5 else 1.0
            angle = sign * max(
                0.05,
                float(self.rng.normal(self.daughter_angle_mean, self.daughter_angle_sd)),
            )
            site = LateralSite(
                site_id=self._next_site_id,
                parent_tip_id=tip.tip_id,
                parent_segment_id=segment_id,
                parent_arc_length=tip.next_lateral_arc,
                position=position,
                parent_orientation=tip.orientation,
                root_order=tip.root_order + 1,
                initiation_time=time,
                emergence_time=time + delay,
                daughter_angle=angle,
                status=status,
            )
            self._next_site_id += 1
            self.sites.append(site)
            new_sites.append(site)
            tip.next_lateral_arc += self.sample_spacing()
        return new_sites

    def due_sites(self, time: float) -> list[LateralSite]:
        return [
            site
            for site in self.sites
            if site.status == LateralStatus.SCHEDULED and site.emergence_time <= time
        ]

    def daughter_orientation(self, site: LateralSite, wet_side_turn: float = 0.0) -> float:
        return wrap_angle(site.parent_orientation + site.daughter_angle + wet_side_turn)


@dataclass(frozen=True)
class MarkovBranchingApproximation:
    """Reduced apex-branching surrogate used only for PDE comparisons."""

    branch_rate: float
    mortality_rate: float

    def event_probabilities(self, dt: float) -> tuple[float, float]:
        if dt <= 0 or self.branch_rate < 0 or self.mortality_rate < 0:
            raise ValueError("invalid Markov branching parameters")
        return (
            -math.expm1(-self.branch_rate * dt),
            -math.expm1(-self.mortality_rate * dt),
        )

