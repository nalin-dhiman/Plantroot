"""Closed finite-carbon accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CarbonLedger:
    initial_budget: float
    length_cost: float
    branch_initiation_cost: float
    maintenance_cost: float
    sensing_cost: float
    construction_spent: float = 0.0
    branching_spent: float = 0.0
    maintenance_spent: float = 0.0
    sensing_spent: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.initial_budget,
            self.length_cost,
            self.branch_initiation_cost,
            self.maintenance_cost,
            self.sensing_cost,
        )
        if self.initial_budget <= 0 or any(value < 0 for value in values[1:]):
            raise ValueError("budget must be positive and costs nonnegative")

    @property
    def total_spent(self) -> float:
        return (
            self.construction_spent
            + self.branching_spent
            + self.maintenance_spent
            + self.sensing_spent
        )

    @property
    def remaining(self) -> float:
        value = self.initial_budget - self.total_spent
        return max(0.0, value) if value > -1e-12 else value

    @property
    def closure_error(self) -> float:
        return abs(self.initial_budget - (self.remaining + self.total_spent))

    def _charge(self, category: str, amount: float) -> bool:
        if amount < 0:
            raise ValueError("charge must be nonnegative")
        if amount > self.remaining + 1e-12:
            return False
        setattr(self, category, getattr(self, category) + min(amount, self.remaining))
        return True

    def charge_maintenance(self, living_length: float, dt: float) -> bool:
        return self._charge("maintenance_spent", self.maintenance_cost * living_length * dt)

    def charge_sensing(self, active_tips: int, dt: float) -> bool:
        return self._charge("sensing_spent", self.sensing_cost * active_tips * dt)

    def affordable_growth(self, requested_length: float) -> float:
        if requested_length < 0:
            raise ValueError("requested_length must be nonnegative")
        if self.length_cost == 0:
            return requested_length
        return min(requested_length, self.remaining / self.length_cost)

    def charge_growth(self, length: float) -> bool:
        return self._charge("construction_spent", self.length_cost * length)

    def charge_branch(self) -> bool:
        return self._charge("branching_spent", self.branch_initiation_cost)

