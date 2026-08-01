from __future__ import annotations

from rootfpt.carbon import CarbonLedger


def test_carbon_ledger_closes_and_rejects_overspend() -> None:
    ledger = CarbonLedger(
        initial_budget=2.0,
        length_cost=1.0,
        branch_initiation_cost=0.2,
        maintenance_cost=0.1,
        sensing_cost=0.05,
    )
    assert ledger.charge_growth(0.7)
    assert ledger.charge_branch()
    assert ledger.charge_maintenance(1.0, 0.5)
    assert ledger.charge_sensing(2, 0.5)
    before = ledger.total_spent
    assert not ledger.charge_growth(2.0)
    assert ledger.total_spent == before
    assert ledger.closure_error < 1e-12
    assert ledger.total_spent <= ledger.initial_budget

