"""Tests for the common ScoreResult schema (#10) and the OBR bridge (#9).

Fast tests cover the pure costing->shock translation, the schema converters,
and the bridge wiring with pe_population_impact and run_reform stubbed out (no
heavy deps). The slow test runs the real pipeline: it needs obr_macro,
policyengine, and HUGGING_FACE_TOKEN for the UK microdata.
"""

from __future__ import annotations

import json
import os

import pytest

from macromod import core
from macromod.cli import main


# ---------------------------------------------------------------------------
# Pure translation: annual costing -> quarterly OBR shock path
# ---------------------------------------------------------------------------

def test_costing_to_shock_sign_units_and_shape():
    # +£10bn/year raised -> HHDI falls £2,500m per quarter; -£2bn (a giveaway)
    # -> HHDI rises £500m per quarter. Flat within each year, 4 quarters/year.
    path = core.obr_costing_to_shock([10.0, -2.0])
    assert path == [-2500.0] * 4 + [500.0] * 4


def test_costing_to_shock_empty():
    assert core.obr_costing_to_shock([]) == []


def test_costing_to_shock_zero_year_is_neutral():
    assert core.obr_costing_to_shock([0.0]) == [0.0] * 4


# ---------------------------------------------------------------------------
# ScoreResult converters (pure; canned adapter payloads)
# ---------------------------------------------------------------------------

def _canned_og_result():
    imp = {}
    for k in ("gdp", "consumption", "investment", "government",
              "tax_revenue", "debt"):
        imp[k] = 100.0
        imp[f"{k}_change"] = 1.5
        imp[f"{k}_pct"] = 0.05
    return {
        "reform": {"gov.x": 0.21},
        "assumptions": "pooled ages, single sector, steady state",
        "impact": {
            "levels_bn": {k: imp[k] for k in
                          ("gdp", "consumption", "investment", "government",
                           "tax_revenue", "debt")},
            "changes_bn": {k: v for k, v in imp.items() if k.endswith("_change")},
            "changes_pct": {k: v for k, v in imp.items() if k.endswith("_pct")},
        },
    }


def test_og_score_block_schema():
    score = core._og_score_block(_canned_og_result())
    assert score["model"] == "og-uk"
    assert score["model_class"] == "olg-ge"
    assert score["horizon"] == "steady-state"
    assert set(score["quantities"]) == set(core.SCORE_QUANTITIES)
    q = score["quantities"]["revenue"]  # tax_revenue is renamed to revenue
    assert q["level_bn"] == 100.0 and q["delta_bn"] == 1.5
    assert q["units"] and q["basis"]
    assert score["distributional"] is None
    json.dumps(score)


def test_pop_score_block_schema():
    res = {
        "country": "uk", "year": 2026, "currency": "GBP",
        "dataset": "enhanced_frs_2023_24", "n_households": 500,
        "reform": {"gov.x": 0.21},
        "budgetary_impact_bn": 6.5,
        "budgetary_impact_basis": "change in gov_balance",
        "decile_impacts": [{"decile": 1, "avg_income_change": -10.0}],
        "winners": 3, "losers": 7,
    }
    score = core._pop_score_block(res)
    assert score["model"] == "pe-microsim"
    assert score["model_class"] == "microsim"
    assert score["horizon"] == "annual 2026"
    # A static microsim fills only the revenue quantity.
    assert set(score["quantities"]) == {"revenue"}
    assert score["quantities"]["revenue"]["delta_bn"] == 6.5
    dist = score["distributional"]
    assert dist["winners"] == 3 and dist["losers"] == 7
    assert dist["decile_impacts"][0]["decile"] == 1
    json.dumps(score)


def test_score_result_rejects_missing_units():
    with pytest.raises(Exception):
        core.ScoreQuantity(delta_bn=1.0)  # units/basis are required


# ---------------------------------------------------------------------------
# OBR bridge wiring (pe_population_impact + run_reform stubbed; fast)
# ---------------------------------------------------------------------------

def _fake_run_reform_df(n_quarters, start_year):
    import pandas as pd

    periods = [
        f"{start_year + q // 4}Q{q % 4 + 1}" for q in range(n_quarters)
    ]
    return pd.DataFrame({
        "period": periods,
        "delta_gdp_bn": [-0.5] * n_quarters,
        "pct_gdp": [-0.06] * n_quarters,
        "delta_cons_m": [-400.0] * n_quarters,
        "delta_if_m": [-25.0] * n_quarters,
    })


@pytest.fixture
def fake_bridge(monkeypatch):
    calls = {"costed_years": [], "run_reform": []}

    def fake_pop(country, reform, year, dataset=None):
        calls["costed_years"].append(year)
        return {"budgetary_impact_bn": 10.0}  # £10bn/year raised, every year

    def fake_run_reform(**kwargs):
        calls["run_reform"].append(kwargs)
        return _fake_run_reform_df(len(kwargs["shock"]), 2026)

    monkeypatch.setattr(core, "pe_population_impact", fake_pop)
    monkeypatch.setattr(core, "_import_obr", lambda: fake_run_reform)
    return calls


def test_obr_bridge_costs_each_year_and_injects_hhdi_path(fake_bridge):
    res = core.obr_score_reform(
        {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21},
        start_year=2026, years=2,
    )
    assert fake_bridge["costed_years"] == [2026, 2027]
    (rr,) = fake_bridge["run_reform"]
    assert rr["var"] == "HHDI"
    # Sign-corrected and unit-converted: +£10bn/yr -> -£2,500m/quarter.
    assert rr["shock"] == [-2500.0] * 8
    assert rr["start"] == "2026Q1" and rr["end"] == "2027Q4"
    assert rr["investment_closure"] is False

    assert res["bridge_variable"] == "HHDI"
    assert res["annual_costings_bn"] == [
        {"year": 2026, "budgetary_impact_bn": 10.0},
        {"year": 2027, "budgetary_impact_bn": 10.0},
    ]
    assert len(res["results"]) == 8
    assert res["cumulative_delta_gdp_bn_over_shock_periods"] == pytest.approx(-4.0)
    assert res["caveats"]
    json.dumps(res)


def test_obr_bridge_score_block(fake_bridge):
    res = core.obr_score_reform({"gov.x": 1}, years=1)
    score = res["score"]
    assert score["model"] == "obr-emulator"
    assert score["model_class"] == "semi-structural"
    assert score["horizon"] == "quarterly window 2026Q1..2026Q4"
    q = score["quantities"]
    assert q["gdp"]["delta_bn"] == pytest.approx(-2.0)
    assert q["consumption"]["delta_bn"] == pytest.approx(-1.6)
    assert q["investment"]["delta_bn"] == pytest.approx(-0.1)
    # Revenue echoes the static costing INPUT, and says so.
    assert q["revenue"]["delta_bn"] == pytest.approx(10.0)
    assert "costing" in q["revenue"]["basis"]
    json.dumps(score)


def test_score_reform_routes_obr(fake_bridge):
    res = core.score_reform("uk", {"gov.x": 1}, model="obr", years=1)
    assert res["model"].startswith("OBR emulator")


def test_score_reform_routes_microsim(monkeypatch):
    seen = {}

    def fake_pop(country, reform, year, dataset=None):
        seen.update(country=country, reform=reform, year=year)
        return {"model": "PolicyEngine population microsimulation"}

    monkeypatch.setattr(core, "pe_population_impact", fake_pop)
    res = core.score_reform("uk", {"gov.x": 1}, model="microsim",
                            start_year=2027)
    assert seen == {"country": "uk", "reform": {"gov.x": 1}, "year": 2027}
    assert res["model"].startswith("PolicyEngine")


def test_obr_bridge_validates_before_importing_anything(fake_bridge):
    with pytest.raises(ValueError, match="non-empty"):
        core.obr_score_reform({})
    with pytest.raises(ValueError, match="years"):
        core.obr_score_reform({"gov.x": 1}, years=0)
    with pytest.raises(ValueError, match="TCPRO"):
        core.obr_score_reform({"gov.hmrc.corporation_tax.main_rate": 0.2})
    assert fake_bridge["run_reform"] == []


# ---------------------------------------------------------------------------
# CLI compare renders one table from N ScoreResults
# ---------------------------------------------------------------------------

def test_cli_compare_renders_common_table(monkeypatch):
    from click.testing import CliRunner

    def fake_score_reform(country, reform, model, start_year=2026, **kw):
        blocks = {
            "microsim": core._pop_score_block({
                "country": "uk", "year": start_year, "currency": "GBP",
                "dataset": "d", "n_households": 1, "reform": reform,
                "budgetary_impact_bn": 6.5, "budgetary_impact_basis": "b",
                "decile_impacts": [], "winners": 0, "losers": 0,
            }),
            "og": core._og_score_block({**_canned_og_result(),
                                        "reform": reform}),
        }
        return {"score": blocks[model]}

    monkeypatch.setattr(core, "score_reform", fake_score_reform)
    runner = CliRunner()
    res = runner.invoke(main, [
        "compare", "--reform", '{"gov.x": 0.21}', "--models", "microsim,og",
    ])
    assert res.exit_code == 0, res.output
    assert "pe-microsim" in res.output and "og-uk" in res.output
    assert "steady-state" in res.output
    # And --json emits a machine-readable list of ScoreResults.
    res = runner.invoke(main, [
        "compare", "--reform", '{"gov.x": 0.21}', "--models", "microsim,og",
        "--json",
    ])
    data = json.loads(res.output)
    assert [s["model"] for s in data] == ["pe-microsim", "og-uk"]


# ---------------------------------------------------------------------------
# Slow: the real pipeline (OBR emulator + PolicyEngine UK microdata)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not (os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")),
    reason="needs HUGGING_FACE_TOKEN for the UK population microdata",
)
def test_obr_bridge_end_to_end_basic_rate_rise():
    pytest.importorskip("obr_macro")
    pytest.importorskip("policyengine")
    # 1p on the basic rate: raises ~£6-7bn/yr statically; the second-round
    # demand effect on GDP must be negative (disposable income falls).
    res = core.obr_score_reform(
        {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21},
        start_year=2026, years=2,
    )
    assert all(c["budgetary_impact_bn"] > 0 for c in res["annual_costings_bn"])
    assert all(s < 0 for s in res["quarterly_shock_path_m"])
    assert res["cumulative_delta_gdp_bn_over_shock_periods"] < 0
    assert res["score"]["quantities"]["gdp"]["delta_bn"] < 0
    json.dumps(res)
