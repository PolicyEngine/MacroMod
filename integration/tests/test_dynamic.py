"""Tests for the macro -> micro EconomicAssumptions overlay (issue #11).

Engine-free tests validate parameter paths against policyengine-uk's shipped
YAML and the compounding fallback; unit tests exercise the overlay
construction on synthetic OG payloads. The one engine-dependent test is the
CRITICAL empirical check that overriding the DERIVED index
gov.economic_assumptions.indices.obr.average_earnings actually moves
aggregate employment income (the derived indices are built once at system
load; a yoy_growth override would silently do nothing) — slow-marked and
skipped where policyengine does not import or the private microdata token
is absent, per the repo's conventions.
"""

from __future__ import annotations

import json
import os

import pytest

from policyengine_macro import assumptions, core
from policyengine_macro.assumptions import (
    EARNINGS_INDEX_PARAM,
    EconomicAssumptions,
    compound_index,
    load_yoy_growth,
    yoy_growth_yaml_candidates,
)


def _synthetic_og(w_reform=0.99, l_reform=0.995, start_year=2026):
    base = {"r": 0.05, "w": 1.00, "Y": 2.0, "K": 6.0, "L": 1.00,
            "C": 1.4, "I": 0.4, "G": 0.2, "tax_revenue": 0.6, "debt": 1.8}
    ref = dict(base, w=w_reform, L=l_reform, r=0.051)
    return {
        "start_year": start_year,
        "baseline_steady_state_model_units": base,
        "reform_steady_state_model_units": ref,
    }


# ---------------------------------------------------------------------------
# Engine-free: path validation against the shipped YAML
# ---------------------------------------------------------------------------

_YAML = yoy_growth_yaml_candidates()


@pytest.mark.skipif(not _YAML, reason="no policyengine-uk yoy_growth.yaml "
                    "found (engine not installed, no uv wheel cache)")
def test_yoy_growth_paths_exist_in_shipped_yaml():
    import yaml

    tree = yaml.safe_load(_YAML[0].read_text())
    obr = tree["obr"]
    for series in ("average_earnings", "consumer_price_index", "rpi"):
        assert series in obr, f"yoy_growth.obr.{series} missing"
        assert obr[series]["values"], series


@pytest.mark.skipif(not _YAML, reason="no policyengine-uk yoy_growth.yaml")
def test_load_yoy_growth_average_earnings():
    growth = load_yoy_growth("average_earnings")
    # Known outturn values from the 2.88.x wheel.
    assert growth[2009] == pytest.approx(0.018)
    assert growth[2026] == pytest.approx(0.034)
    assert min(growth) == 2009


def test_compound_index_toy_series():
    """The fallback reproduces a hand-computed 3-year cumulative index."""
    idx = compound_index({2020: 0.05, 2021: 0.02, 2022: 0.03})
    # Base 1.0 at the earliest year (its own growth is NOT applied,
    # mirroring create_economic_assumption_indices).
    assert idx[2020] == 1.0
    assert idx[2021] == pytest.approx(1.02)
    assert idx[2022] == pytest.approx(round(1.02 * 1.03, 5))
    # Growth held at the last listed value afterwards, through 2039.
    assert idx[2023] == pytest.approx(round(idx[2022] * 1.03, 5))
    assert max(idx) == 2039


# ---------------------------------------------------------------------------
# Unit: EconomicAssumptions construction and overlay emission
# ---------------------------------------------------------------------------

def test_from_og_result_factors():
    ea = EconomicAssumptions.from_og_result(_synthetic_og())
    assert ea.earnings_factor == pytest.approx(0.99)
    assert ea.labour_supply_factor == pytest.approx(0.995)
    assert ea.interest_rate_baseline == 0.05
    assert ea.interest_rate_reform == 0.051
    assert ea.start_year == 2026
    assert any("no transition dynamics" in n for n in ea.notes)
    json.dumps(ea.model_dump())


def test_to_parameter_reform_multiplies_baseline_index():
    ea = EconomicAssumptions.from_og_result(_synthetic_og())
    base = {y: 1.0 + 0.03 * (y - 2025) for y in range(2025, 2040)}
    overlay = ea.to_parameter_reform(base)
    # Only the average-earnings derived index, dated, base * factor.
    assert set(overlay) == {EARNINGS_INDEX_PARAM}
    dated = overlay[EARNINGS_INDEX_PARAM]
    assert set(dated) == {f"{y}-01-01" for y in range(2026, 2036)}
    for y in range(2026, 2036):
        assert dated[f"{y}-01-01"] == pytest.approx(base[y] * 0.99)


def test_to_parameter_reform_null_og_is_identity():
    """Double-counting invariant: a no-op macro result emits an EMPTY
    overlay, so dynamic scoring reduces exactly to static scoring."""
    ea = EconomicAssumptions.from_og_result(
        _synthetic_og(w_reform=1.00, l_reform=1.00)
    )
    assert ea.earnings_factor == 1.0
    assert ea.to_parameter_reform({2026: 1.5}) == {}


def test_to_parameter_reform_errors_outside_base_years():
    ea = EconomicAssumptions.from_og_result(_synthetic_og())
    with pytest.raises(ValueError, match="no baseline index values"):
        ea.to_parameter_reform({1990: 1.0})


def test_collision_guard_rejects_economic_assumption_reforms():
    """A user reform touching gov.economic_assumptions.* is refused before
    any heavy import (no oguk/policyengine needed to hit the error)."""
    with pytest.raises(ValueError, match="gov.economic_assumptions"):
        core.dynamic_population_reform_impact(
            reform={EARNINGS_INDEX_PARAM: 1.0}
        )


def test_dynamic_is_uk_only():
    with pytest.raises(ValueError, match="UK-only"):
        core.dynamic_population_reform_impact(
            country="us", reform={"x": 1.0}
        )
    with pytest.raises(ValueError, match="UK-only"):
        core.score_reform("us", {"x": 1.0}, model="og+microsim")


# ---------------------------------------------------------------------------
# Wiring: dynamic scoring end to end with mocked OG + microsim
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_dynamic(monkeypatch):
    calls = {}

    def fake_og(reform, start_year, max_iter, baseline_cache=True):
        calls["og"] = {"reform": reform, "start_year": start_year}
        return _synthetic_og(start_year=start_year)

    def fake_micro(country, reform, year, dataset=None):
        calls["micro"] = {"reform": reform, "year": year}
        return {
            "currency": "GBP", "budgetary_impact_bn": 5.0,
            "budgetary_impact_basis": "change in gov_balance",
            "headline": "The reform raises £5.0bn/year in 2026.",
            "decile_impacts": [], "winners": 0, "losers": 0,
        }

    monkeypatch.setattr(core, "og_score_reform", fake_og)
    monkeypatch.setattr(core, "pe_population_impact", fake_micro)
    monkeypatch.setattr(
        assumptions, "baseline_index_values",
        lambda series="average_earnings": (
            {y: 1.0 for y in range(2026, 2040)}, "yaml-compound"
        ),
    )
    return calls


def test_dynamic_merges_overlay_into_reform(fake_dynamic):
    reform = {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}
    res = core.dynamic_population_reform_impact(reform=reform, year=2026)
    merged = fake_dynamic["micro"]["reform"]
    assert EARNINGS_INDEX_PARAM in merged
    assert merged["gov.hmrc.income_tax.rates.uk[0].rate"] == 0.21
    assert merged[EARNINGS_INDEX_PARAM]["2026-01-01"] == pytest.approx(0.99)
    assert res["score"]["model"] == "og+microsim"
    assert res["economic_assumptions"]["earnings_factor"] == pytest.approx(0.99)
    assert res["reform"] == reform  # user reform reported without the overlay
    assert any("hours change" in c for c in res["caveats"])
    json.dumps(res)


def test_score_reform_routes_og_microsim(fake_dynamic):
    res = core.score_reform(
        "uk", {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21},
        model="og+microsim",
    )
    assert res["score"]["model"] == "og+microsim"
    assert fake_dynamic["og"]["start_year"] == 2026


# ---------------------------------------------------------------------------
# Engine-dependent: the CRITICAL derived-index override check
# ---------------------------------------------------------------------------

def _pe_engine_skip_reason():
    try:
        import policyengine as pe  # noqa: F401
    except Exception as e:  # broad: pydantic mismatches raise non-ImportError
        return f"policyengine not importable: {type(e).__name__}: {e}"
    if not (os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")):
        return "needs HUGGING_FACE_TOKEN for the UK population microdata"
    return None


_PE_SKIP = _pe_engine_skip_reason()


@pytest.mark.slow
@pytest.mark.skipif(_PE_SKIP is not None, reason=_PE_SKIP or "")
def test_derived_index_override_actually_bites():
    """Empirical proof the overlay does something: overriding the DERIVED
    average-earnings index by x0.99 must lower aggregate employment income
    by ~1% vs stock. If this fails, the overlay silently does nothing
    (e.g. only yoy_growth overrides work) — STOP and report, do not ship.
    """
    import policyengine as pe
    from policyengine.core import Simulation
    from policyengine.outputs.aggregate import Aggregate, AggregateType

    year = 2026
    base_idx, _src = assumptions.baseline_index_values("average_earnings")
    ds, base_sim = core._pe_pop_baseline("uk", year, None)
    reform = {
        EARNINGS_INDEX_PARAM: {
            f"{y}-01-01": round(base_idx[y] * 0.99, 6)
            for y in range(year, year + 3)
        }
    }
    ref_sim = Simulation(
        dataset=ds,
        tax_benefit_model_version=pe.uk.model,
        policy=reform,
    )
    ref_sim.run()

    def _sum_emp(sim):
        agg = Aggregate(
            simulation=sim, variable="employment_income",
            aggregate_type=AggregateType.SUM, entity="person",
        )
        agg.run()
        return float(agg.result)

    ratio = _sum_emp(ref_sim) / _sum_emp(base_sim)
    assert 0.985 < ratio < 0.995, (
        f"derived-index override did not bite: employment income ratio "
        f"{ratio:.4f} (expected ~0.99). Do NOT ship the overlay."
    )
