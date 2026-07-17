"""Wiring / integration tests.

Verify that everything is bolted together: every MCP tool is registered with a
description and schema and dispatches to core; every CLI subcommand runs and
emits valid JSON with --json; and the adapters reject bad input with clear
errors instead of crashing.

Kept fast: the introspection and pure-Python paths need no heavy model import,
and the couple of checks that touch a model (SVAR summary via boe_var) skip
cleanly when it is unavailable. Full solves live behind the `slow` marker, so
this whole module runs in the default (non-slow) suite in seconds.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from click.testing import CliRunner

from macromod import core, mcp_server
from macromod.cli import main

# The full tool surface the server must expose (README + mcp_server.py).
EXPECTED_TOOLS = {
    "score_reform",
    "list_reform_variables",
    "forecast_uk",
    "latest_shocks",
    "model_summary",
    "calculate_household",
    "household_reform_impact",
    "list_reform_parameters",
    "population_reform_impact",
    "obr_shock",
}


# ---------------------------------------------------------------------------
# MCP tool registration + schemas
# ---------------------------------------------------------------------------

def _registered_tools() -> dict:
    return {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}


def test_all_mcp_tools_registered():
    assert EXPECTED_TOOLS <= set(_registered_tools())


def test_mcp_tools_have_descriptions_and_object_schemas():
    for name, tool in _registered_tools().items():
        assert tool.description and tool.description.strip(), f"{name}: no description"
        schema = tool.inputSchema
        assert isinstance(schema, dict), f"{name}: no input schema"
        assert schema.get("type") == "object", f"{name}: schema not an object"


@pytest.mark.parametrize(
    "tool, params",
    [
        ("score_reform", {"country", "reform", "model"}),
        ("obr_shock", {"var", "shock", "periods"}),
        ("forecast_uk", {"horizons", "draws"}),
        ("latest_shocks", {"draws"}),
        ("calculate_household", {"country", "people"}),
        ("household_reform_impact", {"country", "people", "reform"}),
        ("population_reform_impact", {"country", "reform"}),
    ],
)
def test_mcp_tool_schema_exposes_expected_params(tool, params):
    props = set(_registered_tools()[tool].inputSchema.get("properties", {}))
    assert params <= props, f"{tool} missing {params - props}"


# ---------------------------------------------------------------------------
# MCP thin wrappers dispatch to core (instant tools, no heavy solve)
# ---------------------------------------------------------------------------

def test_mcp_list_reform_variables_wired():
    out = mcp_server.list_reform_variables()
    assert isinstance(out, list) and out
    assert {"CGG", "TCPRO"} <= {v["var"] for v in out}
    json.dumps(out)


def _no_pe():
    raise ImportError("fast suite: static catalogue only")


def test_mcp_list_reform_parameters_wired(monkeypatch):
    monkeypatch.setattr(core, "_import_pe", _no_pe)
    out = mcp_server.list_reform_parameters()
    assert isinstance(out, list) and len(out) >= 8
    assert all({"country", "path", "description", "unit"} <= set(p) for p in out)
    json.dumps(out)


def test_mcp_model_summary_wired():
    pytest.importorskip("boe_var")
    out = mcp_server.model_summary()
    assert {"replication", "forecast_revision"} <= set(out)
    json.dumps(out)


# ---------------------------------------------------------------------------
# CLI subcommands run and emit valid JSON with --json
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    return CliRunner()


def _json_ok(result):
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_cli_help_lists_all_commands(runner):
    out = runner.invoke(main, ["--help"]).output
    for cmd in [
        "score", "compare", "obr-shock", "variables", "forecast", "shocks", "summary",
        "parameters", "household", "household-impact", "population-impact",
        "og-score", "og-baseline",
    ]:
        assert cmd in out, f"CLI help missing subcommand {cmd!r}"


def test_cli_variables_json(runner):
    data = _json_ok(runner.invoke(main, ["variables", "--json"]))
    assert isinstance(data, list)
    assert {"CGG", "TCPRO"} <= {v["var"] for v in data}
    for v in data:
        assert {"var", "description", "units", "investment_closure"} <= set(v)


def test_cli_parameters_json(runner, monkeypatch):
    monkeypatch.setattr(core, "_import_pe", _no_pe)
    data = _json_ok(runner.invoke(main, ["parameters", "--json"]))
    assert isinstance(data, list) and len(data) >= 8
    assert all({"country", "path", "description", "unit"} <= set(p) for p in data)


def test_cli_summary_json(runner):
    pytest.importorskip("boe_var")
    data = _json_ok(runner.invoke(main, ["summary", "--json"]))
    assert "replication" in data and "forecast_revision" in data


# ---------------------------------------------------------------------------
# Bad input is reported with a clear error, not a crash
# ---------------------------------------------------------------------------

def test_cli_malformed_json_people(runner):
    res = runner.invoke(main, ["household", "--country", "uk", "--people", "not-json"])
    assert res.exit_code != 0
    assert "valid JSON" in res.output


def test_cli_household_requires_people(runner):
    res = runner.invoke(main, ["household", "--country", "uk"])
    assert res.exit_code != 0
    assert "people" in res.output.lower()


def test_cli_score_requires_reform_and_model(runner):
    res = runner.invoke(main, ["score", "--model", "og"])
    assert res.exit_code != 0
    assert "reform" in res.output.lower()
    res = runner.invoke(main, ["score", "--reform", '{"x": 1}'])
    assert res.exit_code != 0
    assert "model" in res.output.lower()


def test_cli_score_obr_corp_tax_is_clear_error(runner):
    """A corporation-tax reform must be refused with a pointer to the direct
    TCPRO lever, before any heavy model import."""
    res = runner.invoke(main, [
        "score", "--reform", '{"gov.hmrc.corporation_tax.main_rate": 0.2}',
        "--model", "obr",
    ])
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "TCPRO" in res.output


def test_cli_compare_bad_model_is_clean_error(runner):
    res = runner.invoke(main, [
        "compare", "--reform", '{"x": 1}', "--models", "svar",
    ])
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "model must be one of" in res.output


def test_cli_obr_shock_requires_var(runner):
    res = runner.invoke(main, ["obr-shock", "--shock", "1000"])
    assert res.exit_code != 0
    assert "var" in res.output.lower()


def test_cli_rejects_invalid_country_choice(runner):
    # click.Choice(["uk","us"]) rejects before any model import.
    res = runner.invoke(main, ["household", "--country", "fr", "--people", "[]"])
    assert res.exit_code != 0
    assert "fr" in res.output


# ---------------------------------------------------------------------------
# core adapters reject bad input with a clear error (no heavy import needed:
# these validate before importing the underlying model)
# ---------------------------------------------------------------------------

def test_core_household_bad_country_raises():
    with pytest.raises(ValueError):
        core.pe_household("fr", [{"age": 30}])


def test_core_household_impact_requires_reform():
    with pytest.raises(ValueError):
        core.pe_household_impact("uk", [{"age": 30}], reform={})


def test_core_population_impact_validation():
    with pytest.raises(ValueError):
        core.pe_population_impact("uk", reform={})
    with pytest.raises(ValueError):
        core.pe_population_impact("fr", reform={"x": 1})


# ---------------------------------------------------------------------------
# End-to-end CLI wiring through the real models (slow: OBR solve / PE import)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cli_obr_shock_json_end_to_end(runner):
    data = _json_ok(runner.invoke(
        main, ["obr-shock", "--var", "CGG", "--shock", "1250", "--periods", "4", "--json"]
    ))
    assert data["var"] == "CGG"
    assert len(data["results"]) >= 4
    assert data["results"][0]["delta_gdp_bn"] > 0


@pytest.mark.slow
def test_cli_forecast_json_end_to_end(runner):
    data = _json_ok(runner.invoke(
        main, ["forecast", "--horizons", "4", "--draws", "100", "--json"]
    ))
    assert data["horizons"] == 4
    assert len(data["gdp_growth_yoy"]) == 4


@pytest.mark.slow
def test_cli_household_json_end_to_end(runner):
    data = _json_ok(runner.invoke(main, [
        "household", "--country", "uk",
        "--people", '[{"age":35,"employment_income":50000}]', "--json",
    ]))
    assert data["currency"] == "GBP"
    assert data["summary"]["income_tax_by_person"][0] > 0


@pytest.mark.slow
def test_core_obr_extreme_shock_is_wellformed():
    # An out-of-range shock should still solve to well-formed output, not crash.
    res = core.obr_shock(var="CGG", shock=1_000_000, periods=2)
    assert res["periods"] == 2 and len(res["results"]) >= 2
    json.dumps(res)


def test_cli_obr_shock_closure_tristate(runner, monkeypatch):
    """Omitted --investment-closure must reach core as None (per-variable
    default), not False — the TCPRO zero-effects footgun (review #15.1)."""
    seen = []

    def fake_obr_shock(**kwargs):
        seen.append(kwargs["investment_closure"])
        return {"name": "x", "var": "TCPRO", "shock": -0.05, "periods": 1,
                "investment_closure": True, "results": [],
                "cumulative_delta_gdp_bn_over_shock_periods": 0.0,
                "peak_pct_gdp": 0.0, "peak_period": "2025Q1"}

    monkeypatch.setattr(core, "obr_shock", fake_obr_shock)
    for args, expected in [
        (["obr-shock", "--var", "TCPRO", "--shock", "-0.05", "--json"], None),
        (["obr-shock", "--var", "TCPRO", "--shock", "-0.05",
          "--investment-closure", "--json"], True),
        (["obr-shock", "--var", "TCPRO", "--shock", "-0.05",
          "--no-investment-closure", "--json"], False),
    ]:
        res = runner.invoke(main, args)
        assert res.exit_code == 0, res.output
        assert seen[-1] is expected


def test_cli_wrong_shaped_reform_is_clean_error(runner):
    """Valid JSON of the wrong shape ('[]', '{}') must be a Click error,
    never a traceback (review #15.4)."""
    for cmd in (["score", "--model", "og"], ["og-score"]):
        for bad in ("[]", "{}"):
            res = runner.invoke(main, cmd + ["--reform", bad])
            assert res.exit_code != 0
            assert "Traceback" not in res.output
            assert "non-empty" in res.output
