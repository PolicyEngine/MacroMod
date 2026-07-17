"""Tests for pe_population_impact (population-level reform scoring).

Fast tests stub out policyengine entirely (no 20s model import, no data
download); the slow test runs the real UK microsimulation and needs
HUGGING_FACE_TOKEN for the first-ever dataset download.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from policyengine_macro import core


# ---------------------------------------------------------------------------
# Fast, fully mocked
# ---------------------------------------------------------------------------

class _FakeSim:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_reform = kwargs.get("policy") is not None

    def run(self):
        pass


def _fake_decile_rows(**_kwargs):
    rows = [
        types.SimpleNamespace(
            decile=i,
            baseline_mean=100.0 * i,
            reform_mean=100.0 * i - 10.0,
            relative_change=-0.1 / i,
            count_better_off=0.0,
            count_worse_off=1000.0 * i,
        )
        for i in range(1, 11)
    ]
    return types.SimpleNamespace(outputs=rows)


@pytest.fixture
def fake_pe(monkeypatch):
    """Stub policyengine modules + the dataset/baseline machinery."""
    pe_mod = types.ModuleType("policyengine")
    pe_mod.uk = types.SimpleNamespace(model="uk-model")
    pe_mod.us = types.SimpleNamespace(model="us-model")

    core_mod = types.ModuleType("policyengine.core")
    core_mod.Simulation = _FakeSim
    outputs_mod = types.ModuleType("policyengine.outputs")
    decile_mod = types.ModuleType("policyengine.outputs.decile_impact")
    decile_mod.calculate_decile_impacts = _fake_decile_rows

    monkeypatch.setitem(sys.modules, "policyengine", pe_mod)
    monkeypatch.setitem(sys.modules, "policyengine.core", core_mod)
    monkeypatch.setitem(sys.modules, "policyengine.outputs", outputs_mod)
    monkeypatch.setitem(
        sys.modules, "policyengine.outputs.decile_impact", decile_mod
    )
    monkeypatch.setattr(core, "_import_pe", lambda: pe_mod)

    fake_ds = types.SimpleNamespace(
        name="fake-frs-2026",
        year=2026,
        data=types.SimpleNamespace(household=[{}] * 500),
    )
    baseline = _FakeSim(dataset=fake_ds)
    monkeypatch.setattr(
        core, "_pe_pop_baseline", lambda country, year, dataset: (fake_ds, baseline)
    )

    # Baseline gov_balance £578bn, reform £584.5bn -> +£6.5bn budgetary
    sums = {
        (False, "gov_balance"): 578.0e9,
        (True, "gov_balance"): 584.5e9,
        (False, "household_net_income"): 1_500.0e9,
        (True, "household_net_income"): 1_494.0e9,
        (False, "household_tax"): 3_000.0e9,
        (True, "household_tax"): 3_010.0e9,
        (False, "household_benefits"): 1_000.0e9,
        (True, "household_benefits"): 998.0e9,
    }
    monkeypatch.setattr(
        core, "_pe_pop_sum", lambda sim, var: sums[(sim.is_reform, var)]
    )
    return pe_mod


def test_population_impact_uk_shape(fake_pe):
    res = core.pe_population_impact(
        "uk", reform={"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}
    )
    assert res["country"] == "uk"
    assert res["currency"] == "GBP"
    assert res["year"] == 2026
    assert res["n_households"] == 500
    assert res["budgetary_impact_bn"] == pytest.approx(6.5)
    assert "raises" in res["headline"]
    assert res["household_net_income_change_bn"] == pytest.approx(-6.0)
    assert len(res["decile_impacts"]) == 10
    row = res["decile_impacts"][0]
    assert {"decile", "avg_income_change", "relative_change_pct",
            "count_better_off", "count_worse_off"} <= set(row)
    assert res["winners"] == 0
    assert res["losers"] == sum(1000 * i for i in range(1, 11))
    json.dumps(res)


def test_population_impact_us_uses_tax_minus_benefits(fake_pe):
    res = core.pe_population_impact(
        "us", reform={"gov.irs.credits.ctc.amount.base[0].amount": 3000}
    )
    # d_tax = +10bn, d_benefits = -2bn -> +12bn
    assert res["budgetary_impact_bn"] == pytest.approx(12.0)
    assert res["currency"] == "USD"


def test_population_impact_validation():
    with pytest.raises(ValueError):
        core.pe_population_impact("uk", reform={})
    with pytest.raises(ValueError):
        core.pe_population_impact("fr", reform={"x": 1})


# ---------------------------------------------------------------------------
# Slow, real microsimulation (needs data; first run needs HUGGING_FACE_TOKEN)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not (os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")),
    reason="needs HUGGING_FACE_TOKEN for the UK population microdata",
)
def test_population_impact_uk_real_basic_rate():
    res = core.pe_population_impact(
        "uk", reform={"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}, year=2026
    )
    # 1p on the basic rate raises roughly £6-7bn/yr (measured £6.46bn).
    assert 4.0 < res["budgetary_impact_bn"] < 9.0, res["budgetary_impact_bn"]
    assert res["household_net_income_change_bn"] < 0
    assert res["n_households"] > 10_000
    assert res["losers"] > res["winners"]
    json.dumps(res)
