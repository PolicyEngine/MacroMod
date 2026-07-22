"""Tests for the macro -> micro EconomicAssumptions overlay (issue #11).

The overlay is applied by DIRECT INPUT SCALING of the reform simulation's
employment-income arrays via the engine's Dynamic(simulation_modifier=...)
hook — NOT by parameter overrides on the uprating indices.

EMPIRICAL RECORD (2026-07-20, production engine via the deployed MCP
server): derived-index overrides are DEAD on pre-built per-year datasets.
Two population_reform_impact calls with a reform overriding
gov.economic_assumptions.indices.obr.average_earnings for 2026 — first
x0.99 (value 1.66561), then a drastic 0.84 (~half the compounded baseline)
— BOTH returned exactly zero everywhere: £0.0bn budgetary impact, 0
winners, 0 losers, all deciles 0.0. Root cause: the per-year datasets
(enhanced_frs_2023_24-year-2026) are pre-uprated at dataset BUILD time
(policyengine tax_benefit_models/uk/datasets.py create_datasets), and
PolicyEngineUKLatest.run feeds the stored input arrays straight into
UKSingleYearDataset, so simulation-time uprating parameters are never
consulted for input variables. Hence the input-scaling mechanism.

Fast tests are engine-free (synthetic OG payloads, fake microsim); the
engine-gated test proving the NEW mechanism bites is slow-marked and
skipped where policyengine does not import or the microdata token is
absent, per the repo's conventions.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from policyengine_macro import core
from policyengine_macro.assumptions import (
    SCALED_INPUT_VARIABLES,
    EconomicAssumptions,
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
# Unit: EconomicAssumptions construction
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


def test_caveats_report_unallocated_hours():
    ea = EconomicAssumptions.from_og_result(_synthetic_og())
    caveats = ea.caveat_strings()
    assert any("-0.50%" in c for c in caveats)
    assert any("self-employment" in c for c in caveats)


# ---------------------------------------------------------------------------
# Unit: the input-scaling modifier against a fake engine simulation
# ---------------------------------------------------------------------------

class _FakeHolder:
    def __init__(self, arrays: dict):
        self.arrays = arrays  # period -> np.ndarray

    def get_known_periods(self):
        return list(self.arrays)

    def get_array(self, period):
        return self.arrays[period]

    def delete_arrays(self, period):
        del self.arrays[period]


class _FakeMicrosim:
    """Mimics the policyengine_uk.Microsimulation surface the modifier uses
    (get_holder / set_input, the policyengine_core Simulation API)."""

    def __init__(self, holders: dict):
        self.holders = {name: _FakeHolder(arrs)
                        for name, arrs in holders.items()}
        self.set_calls: list[tuple] = []

    def get_holder(self, name):
        return self.holders.setdefault(name, _FakeHolder({}))

    def set_input(self, name, period, values):
        self.set_calls.append((name, period))
        self.holders[name].arrays[period] = values


def test_modifier_scales_before_lsr_inputs():
    ea = EconomicAssumptions.from_og_result(_synthetic_og())
    sim = _FakeMicrosim({
        "employment_income_before_lsr": {2026: np.array([100.0, 200.0])},
        "employment_income": {},
    })
    ea.input_scaling_modifier()(sim)
    scaled = sim.holders["employment_income_before_lsr"].arrays[2026]
    assert scaled == pytest.approx([99.0, 198.0])
    # Only the populated variable was touched, once.
    assert sim.set_calls == [("employment_income_before_lsr", 2026)]


def test_modifier_falls_back_to_employment_income():
    ea = EconomicAssumptions.from_og_result(_synthetic_og())
    sim = _FakeMicrosim({
        "employment_income": {2026: np.array([50.0])},
    })
    ea.input_scaling_modifier()(sim)
    assert sim.holders["employment_income"].arrays[2026] == pytest.approx(
        [49.5]
    )


def test_modifier_refuses_when_no_input_is_populated():
    """Never return a silently static result as a dynamic one."""
    ea = EconomicAssumptions.from_og_result(_synthetic_og())
    with pytest.raises(RuntimeError, match="no populated"):
        ea.input_scaling_modifier()(_FakeMicrosim({}))


def test_null_macro_result_yields_no_modifier():
    """Double-counting invariant: factor 1.0 -> None, so the reform
    simulation carries no Dynamic and is identical to the static one."""
    ea = EconomicAssumptions.from_og_result(
        _synthetic_og(w_reform=1.00, l_reform=1.00)
    )
    assert ea.earnings_factor == 1.0
    assert ea.input_scaling_modifier() is None


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_guard_rejects_dead_economic_assumption_reforms():
    """Dynamic scoring refuses gov.economic_assumptions.* user overrides:
    they double-drive the overlay's channel, and the input-uprating index
    paths are additionally inert on pre-built datasets (some paths in the
    namespace ARE live at sim time — the guard's error says to apply those
    via a static run). Refused before any heavy import."""
    with pytest.raises(ValueError, match="double-drive"):
        core.dynamic_population_reform_impact(
            reform={
                "gov.economic_assumptions.indices.obr.average_earnings": 1.0
            }
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
        return _synthetic_og(
            w_reform=calls.get("w_reform", 0.99), start_year=start_year
        )

    def fake_micro(country, reform, year, dataset=None, reform_modifier=None):
        calls["micro"] = {"reform": reform, "year": year,
                          "reform_modifier": reform_modifier}
        return {
            "currency": "GBP", "budgetary_impact_bn": 5.0,
            "budgetary_impact_basis": "change in gov_balance",
            "headline": "The reform raises £5.0bn/year in 2026.",
            "decile_impacts": [], "winners": 0, "losers": 0,
            # Nested static score, as the real pe_population_impact
            # returns: the dynamic wrapper must STRIP it (one
            # authoritative score) — without this key the single-score
            # test would be vacuous.
            "score": {"model": "pe-microsim",
                      "analysis_type": "static microsimulation"},
        }

    monkeypatch.setattr(core, "og_score_reform", fake_og)
    monkeypatch.setattr(core, "pe_population_impact", fake_micro)
    return calls


def test_dynamic_applies_modifier_to_reform_side_only(fake_dynamic):
    reform = {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}
    res = core.dynamic_population_reform_impact(reform=reform, year=2026)
    micro = fake_dynamic["micro"]
    # User reform passes through UNCHANGED (applied once, as policy);
    # the overlay travels separately as the reform-side-only modifier.
    assert micro["reform"] == reform
    assert callable(micro["reform_modifier"])
    assert res["application"]["method"] == "input-scaling"
    assert res["application"]["applied"] is True
    assert res["application"]["variables_tried"] == list(SCALED_INPUT_VARIABLES)
    assert res["score"]["model"] == "og+microsim"
    assert res["economic_assumptions"]["earnings_factor"] == pytest.approx(0.99)
    assert any("effective-labour change" in c for c in res["caveats"])
    assert any("input" in a for a in res["assumptions"])
    json.dumps(res)


def test_dynamic_null_macro_reduces_to_static(fake_dynamic):
    """Invariant end to end: w unchanged -> no modifier attached, so the
    reform simulation is exactly the static one."""
    fake_dynamic["w_reform"] = 1.00
    res = core.dynamic_population_reform_impact(
        reform={"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}
    )
    assert fake_dynamic["micro"]["reform_modifier"] is None
    assert res["application"]["applied"] is False


def test_score_reform_routes_og_microsim(fake_dynamic):
    res = core.score_reform(
        "uk", {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21},
        model="og+microsim",
    )
    assert res["score"]["model"] == "og+microsim"
    assert fake_dynamic["og"]["start_year"] == 2026


# ---------------------------------------------------------------------------
# Engine-gated: the CRITICAL check that the NEW mechanism bites
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
def test_input_scaling_actually_bites():
    """Empirical proof the input-scaling overlay does something: attaching
    the x0.99 modifier via Dynamic(simulation_modifier=...) must lower
    aggregate employment income by ~1% vs stock.

    Context: the previous mechanism (overriding the derived uprating index
    gov.economic_assumptions.indices.obr.average_earnings) was proven DEAD
    against the production engine on 2026-07-20 — two population calls
    overriding the 2026 index by x0.99 (1.66561) and then 0.84 both
    returned exactly zero everywhere (£0.0bn, 0 winners, 0 losers, all
    deciles 0.0), because the per-year datasets are pre-uprated at build
    time. If THIS test fails, the replacement mechanism is dead too — do
    NOT ship the overlay.
    """
    import policyengine as pe
    from policyengine.core import Dynamic, Simulation
    from policyengine.outputs.aggregate import Aggregate, AggregateType

    year = 2026
    ea = EconomicAssumptions.from_og_result(_synthetic_og(w_reform=0.99))
    ds, base_sim = core._pe_pop_baseline("uk", year, None)
    ref_sim = Simulation(
        dataset=ds,
        tax_benefit_model_version=pe.uk.model,
        dynamic=Dynamic(
            name="test x0.99 earnings overlay",
            simulation_modifier=ea.input_scaling_modifier(),
            affects_labor_supply_response=False,
        ),
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
        f"input-scaling overlay did not bite: employment income ratio "
        f"{ratio:.4f} (expected ~0.99). Do NOT ship the overlay."
    )


# ---------------------------------------------------------------------------
# The PRODUCTION attachment seam (review #72.6): pe_population_impact must
# attach the modifier as Dynamic(simulation_modifier=..., no-LSR) on the
# reform Simulation, and attach nothing when the modifier is None — pinned
# with fake engine modules, no policyengine install needed.
# ---------------------------------------------------------------------------

class _FakeDS:
    name = "fake-ds"

    class data:
        household = [0] * 3


class _FakeAggregate:
    def __init__(self, simulation=None, variable=None, aggregate_type=None,
                 entity=None):
        self.result = 0.0

    def run(self):
        pass


class _FakeDynamic:
    def __init__(self, name=None, simulation_modifier=None,
                 affects_labor_supply_response=None):
        self.name = name
        self.simulation_modifier = simulation_modifier
        self.affects_labor_supply_response = affects_labor_supply_response


class _CaptureSimulation:
    captured: list = []

    def __init__(self, **kwargs):
        _CaptureSimulation.captured.append(kwargs)

    def run(self):
        pass


@pytest.fixture
def fake_engine(monkeypatch):
    import sys
    import types

    _CaptureSimulation.captured = []
    pe_mod = types.ModuleType("policyengine")
    pe_mod.uk = types.SimpleNamespace(model="fake-uk-model")
    core_mod = types.ModuleType("policyengine.core")
    core_mod.Simulation = _CaptureSimulation
    core_mod.Dynamic = _FakeDynamic
    outputs_mod = types.ModuleType("policyengine.outputs")
    agg_mod = types.ModuleType("policyengine.outputs.aggregate")
    agg_mod.Aggregate = _FakeAggregate
    agg_mod.AggregateType = types.SimpleNamespace(SUM="sum")
    dec_mod = types.ModuleType("policyengine.outputs.decile_impact")
    dec_mod.calculate_decile_impacts = (
        lambda **kw: types.SimpleNamespace(outputs=[])
    )
    for name, mod in {
        "policyengine": pe_mod,
        "policyengine.core": core_mod,
        "policyengine.outputs": outputs_mod,
        "policyengine.outputs.aggregate": agg_mod,
        "policyengine.outputs.decile_impact": dec_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(core, "_import_pe", lambda: pe_mod)
    monkeypatch.setattr(
        core, "_pe_pop_baseline", lambda c, y, d: (_FakeDS(), object())
    )
    return _CaptureSimulation


def test_seam_attaches_dynamic_on_reform_side(fake_engine):
    sentinel = object()
    core.pe_population_impact(
        country="uk", reform={"x": 1.0}, year=2026, reform_modifier=sentinel
    )
    (sim_kwargs,) = fake_engine.captured
    dyn = sim_kwargs["dynamic"]
    assert dyn.simulation_modifier is sentinel
    assert dyn.affects_labor_supply_response is False


def test_seam_attaches_nothing_when_modifier_none(fake_engine):
    core.pe_population_impact(
        country="uk", reform={"x": 1.0}, year=2026, reform_modifier=None
    )
    (sim_kwargs,) = fake_engine.captured
    assert "dynamic" not in sim_kwargs


# ---------------------------------------------------------------------------
# og_payload: the two-environment pipeline (review #72.1)
# ---------------------------------------------------------------------------

def _payload_for(reform, year=2026, w_reform=0.99):
    p = _synthetic_og(w_reform=w_reform, start_year=year)
    p["reform"] = dict(reform)
    return p


def test_og_payload_skips_og_solve(fake_dynamic, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("og_score_reform must not run when a payload is given")

    monkeypatch.setattr(core, "og_score_reform", boom)
    reform = {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}
    res = core.dynamic_population_reform_impact(
        reform=reform, year=2026, og_payload=_payload_for(reform)
    )
    assert res["economic_assumptions"]["earnings_factor"] == pytest.approx(0.99)
    assert res["application"]["applied"] is True


def test_og_payload_reform_and_year_mismatch_refused(fake_dynamic):
    reform = {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}
    with pytest.raises(ValueError, match="different reform"):
        core.dynamic_population_reform_impact(
            reform=reform, year=2026,
            og_payload=_payload_for({"other.param": 1.0}),
        )
    with pytest.raises(ValueError, match="start_year"):
        core.dynamic_population_reform_impact(
            reform=reform, year=2027, og_payload=_payload_for(reform, year=2026)
        )
    with pytest.raises(ValueError, match="missing required keys"):
        core.dynamic_population_reform_impact(
            reform=reform, year=2026, og_payload={"reform": reform}
        )


def test_dynamic_result_has_single_authoritative_score(fake_dynamic):
    """The embedded static microsim score is dropped (review #72.4): one
    dynamic result, one score block."""
    reform = {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}
    res = core.dynamic_population_reform_impact(
        reform=reform, year=2026, og_payload=_payload_for(reform)
    )
    assert "score" not in res["microsim"]
    assert res["score"]["model"] == "og+microsim"


def test_implausible_og_ratio_refused():
    with pytest.raises(ValueError, match="implausible"):
        EconomicAssumptions.from_og_result(_synthetic_og(w_reform=0.30))
    with pytest.raises(ValueError, match="degenerate"):
        EconomicAssumptions.from_og_result(_synthetic_og(w_reform=float("nan")))


def test_small_factor_survives_unrounded():
    """Review #72.2 / verify-round: a wage ratio of 1.00004 must NOT
    quantize to 1.0 — traversing core._og_ss_dict, the layer where the
    original 4dp rounding lived, not just the assumptions math."""

    class _SS:
        def __init__(self, w):
            self._d = {"r": 0.05, "w": w, "Y": 2.0, "K": 6.0, "L": 1.0,
                       "C": 1.4, "I": 0.4, "G": 0.2, "tax_revenue": 0.6,
                       "debt": 1.8}

        def model_dump(self):
            return dict(self._d)

    payload = {
        "start_year": 2026,
        "baseline_steady_state_model_units": core._og_ss_dict(_SS(1.0)),
        "reform_steady_state_model_units": core._og_ss_dict(_SS(1.00004)),
    }
    ea = EconomicAssumptions.from_og_result(payload)
    assert ea.earnings_factor == pytest.approx(1.00004, abs=1e-9)
    assert ea.earnings_factor != 1.0
    assert ea.input_scaling_modifier() is not None


@pytest.mark.slow
@pytest.mark.skipif(_PE_SKIP is not None, reason=_PE_SKIP or "")
def test_dynamic_og_payload_end_to_end_bites():
    """Engine-level proof of the PRODUCTION path (review #72.6): the full
    dynamic_population_reform_impact pipeline with a pre-computed OG payload
    (no oguk needed) must move aggregates vs the static score."""
    reform = {"gov.hmrc.income_tax.rates.uk[0].rate": 0.20}
    res = core.dynamic_population_reform_impact(
        reform=reform, year=2026, og_payload=_payload_for(reform, w_reform=0.99)
    )
    assert res["application"]["applied"] is True
    assert abs(res["microsim"]["household_net_income_change_bn"]) > 1.0


def test_mixed_mode_valueerror_gets_two_step_guidance(fake_dynamic, monkeypatch):
    """A mixed-computation-mode import clash is rewritten into the two-env
    guidance; an unrelated ValueError mentioning 'computation' is NOT."""
    reform = {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}

    def clash(**kw):
        raise ValueError("mixed computation mode: policyengine-uk 2.88 ...")

    monkeypatch.setattr(core, "og_score_reform", clash)
    with pytest.raises(RuntimeError, match="og-payload"):
        core.dynamic_population_reform_impact(reform=reform, year=2026)

    def unrelated(**kw):
        raise ValueError("steady-state computation did not converge")

    monkeypatch.setattr(core, "og_score_reform", unrelated)
    with pytest.raises(ValueError, match="did not converge"):
        core.dynamic_population_reform_impact(reform=reform, year=2026)


def test_non_numeric_payload_field_is_actionable(fake_dynamic):
    reform = {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}
    bad = _payload_for(reform)
    bad["reform_steady_state_model_units"]["w"] = "not-a-number"
    with pytest.raises(ValueError, match="og-score"):
        core.dynamic_population_reform_impact(
            reform=reform, year=2026, og_payload=bad
        )
