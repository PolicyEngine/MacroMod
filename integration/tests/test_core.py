"""Unit tests for the macromod adapters (small draws to keep runtime low)."""

import json

import pytest

from macromod import core


def test_list_variables():
    vars_ = core.obr_list_variables()
    codes = {v["var"] for v in vars_}
    assert {"CGG", "TCPRO"} <= codes
    for v in vars_:
        assert {"var", "description", "units", "investment_closure"} <= set(v)
    json.dumps(vars_)


def test_summary_parses():
    s = core.svar_summary()
    assert "replication" in s and "forecast_revision" in s
    fevd = s["replication"]["fevd_1yr_headline"]
    assert any("UK GDP" in row.get("Variable", "") for row in fevd)
    json.dumps(s)


@pytest.mark.slow
def test_score_reform_cgg():
    res = core.obr_score_reform(var="CGG", shock=1250, periods=4)
    assert res["periods"] == 4
    assert len(res["results"]) >= 4
    # A spending increase should raise GDP in the first shocked quarter.
    assert res["results"][0]["delta_gdp_bn"] > 0
    json.dumps(res)


@pytest.mark.slow
def test_svar_forecast_and_cache():
    res = core.svar_forecast(horizons=4, draws=100)
    assert len(res["gdp_growth_yoy"]) == 4
    row = res["gdp_growth_yoy"][0]
    assert row["lo90"] <= row["lo68"] <= row["median"] <= row["hi68"] <= row["hi90"]
    json.dumps(res)
    # Second call must hit the in-process cache (same object).
    assert core.svar_forecast(horizons=4, draws=100) is res


# ---------------------------------------------------------------------------
# PolicyEngine adapters
# ---------------------------------------------------------------------------

def test_pe_list_common_parameters():
    params = core.pe_list_common_parameters()
    assert len(params) >= 8
    for p in params:
        assert {"country", "path", "description", "unit"} <= set(p)
        assert p["country"] in ("uk", "us")
    paths = {p["path"] for p in params}
    assert "gov.hmrc.income_tax.rates.uk[0].rate" in paths
    json.dumps(params)


@pytest.mark.slow
def test_pe_household_uk_50k():
    res = core.pe_household("uk", [{"age": 35, "employment_income": 50_000}])
    s = res["summary"]
    it = s["income_tax_by_person"][0]
    assert 6_500 < it < 8_500  # ~£7,486 in 2026
    assert s["national_insurance_by_person"][0] > 0
    assert 30_000 < s["household_net_income"] < 45_000
    assert res["currency"] == "GBP"
    json.dumps(res)


@pytest.mark.slow
def test_pe_household_impact_uk_basic_rate_rise():
    res = core.pe_household_impact(
        "uk",
        [{"age": 35, "employment_income": 50_000}],
        reform={"gov.hmrc.income_tax.rates.uk[0].rate": 0.25},
    )
    d_it = res["change"]["income_tax_by_person"][0]
    assert d_it > 500  # 5pp on ~£37.4k of basic-rate income ≈ +£1,872
    assert res["net_income_change"] < 0
    json.dumps(res)


@pytest.mark.slow
def test_pe_household_us():
    res = core.pe_household(
        "us",
        [{"age": 35, "employment_income": 60_000}],
        tax_unit={"filing_status": "SINGLE"},
        household={"state_code_str": "CA"},
    )
    s = res["summary"]
    assert s["federal_income_tax"] > 0
    assert s["employee_payroll_tax"] > 0
    assert 35_000 < s["household_net_income"] < 60_000
    assert res["currency"] == "USD"
    json.dumps(res)


def test_pe_household_bad_country():
    with pytest.raises(ValueError):
        core.pe_household("fr", [{"age": 30}])
    with pytest.raises(ValueError):
        core.pe_household_impact("uk", [{"age": 30}], reform={})


@pytest.mark.slow
def test_svar_latest_shocks():
    res = core.svar_latest_shocks(draws=100)
    assert len(res["shocks"]) == 6
    for s in res["shocks"]:
        assert abs(s["p_positive"] + s["p_negative"] - 1.0) < 1e-6
    json.dumps(res)
