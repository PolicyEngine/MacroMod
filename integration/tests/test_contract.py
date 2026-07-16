"""Upstream contract tests: every name this package hardcodes must still
exist upstream.

The adapters compose upstream packages but necessarily hardcode a few names —
curated parameter paths, headline output-variable names, SVAR column names.
These tests fail loudly when an upstream rename would otherwise surface as a
silent drift (stale baseline, mislabeled series) or a runtime AttributeError.

The boe_var checks are fast and run by default. The PolicyEngine checks need
the full country models (~20s import) and are marked `slow`
(`pytest --runslow`); they also run implicitly wherever PE is installed,
e.g. the Modal image.
"""

import importlib
import os

import pytest

from macromod import core


def _import_or_require(modname: str):
    """Import an upstream module; in the dedicated contract job
    (MACROMOD_REQUIRE_PE=1) a missing/broken install FAILS instead of
    skipping — a contract run that silently skips is vacuous."""
    require = os.environ.get("MACROMOD_REQUIRE_PE") == "1"
    try:
        return importlib.import_module(modname)
    except Exception as e:
        msg = f"{modname} not importable: {type(e).__name__}: {e}"
        pytest.fail(msg) if require else pytest.skip(msg)


# ---------------------------------------------------------------------------
# boe_var (fast)
# ---------------------------------------------------------------------------

def test_svar_headline_columns_exist_upstream():
    data = _import_or_require("boe_var.data")
    for col in (core._COL_CPI, core._COL_GDP):
        assert col in data.COLUMNS, (
            f"boe_var.data.COLUMNS no longer contains {col!r}; "
            "the SVAR headline series lookup would fail"
        )


def test_svar_identified_schema_aligned():
    """WORLD_SHOCKS + UK_SHOCKS is the canonical schema the adapter uses;
    SHOCK_NAMES must stay positionally aligned with it (an upstream reorder
    must fail here, not silently shift UK shock probabilities)."""
    analysis = _import_or_require("boe_var.analysis")
    identified = sorted(analysis.WORLD_SHOCKS + analysis.UK_SHOCKS)
    assert len(identified) == 6 and len(set(identified)) == 6
    assert all(0 <= j < len(analysis.SHOCK_NAMES) for j in identified)
    assert all(
        not analysis.SHOCK_NAMES[j].startswith("Unident") for j in identified
    ), [analysis.SHOCK_NAMES[j] for j in identified]
    unidentified = [
        analysis.SHOCK_NAMES[j]
        for j in range(len(analysis.SHOCK_NAMES))
        if j not in identified
    ]
    assert all(n.startswith("Unident") for n in unidentified), unidentified
    # Semantic alignment, not just structure: a swap of e.g. UK demand/supply
    # labels would silently relabel probabilities — pin the exact names.
    assert tuple(
        analysis.SHOCK_NAMES[j] for j in analysis.WORLD_SHOCKS
    ) == ("World demand", "World energy", "World supply")
    assert tuple(
        analysis.SHOCK_NAMES[j] for j in analysis.UK_SHOCKS
    ) == ("UK demand", "UK supply", "UK mon. pol.")


# ---------------------------------------------------------------------------
# PolicyEngine (slow: full country-model import)
# ---------------------------------------------------------------------------

# Every variable name _pe_summary and pe_population_impact read off results.
PE_VARIABLE_CONTRACT = {
    "uk": [
        "income_tax", "national_insurance", "hbai_household_net_income",
        "household_tax", "household_benefits", "universal_credit",
        "child_benefit",
        # population scoring
        "gov_balance", "gov_tax", "household_net_income",
        "household_income_decile",
    ],
    "us": [
        "income_tax", "employee_payroll_tax", "state_income_tax", "ctc",
        "eitc", "household_net_income", "household_tax",
        "household_benefits",
    ],
}


def _pe_model(country):
    # importorskip only catches ImportError; a broken policyengine install
    # (e.g. a pydantic version mismatch) raises other exceptions at import
    # time and should also skip, not error. In the dedicated contract job
    # (MACROMOD_REQUIRE_PE=1) a missing/broken install FAILS instead: a
    # contract run that silently skips every contract is vacuous.
    require = os.environ.get("MACROMOD_REQUIRE_PE") == "1"
    try:
        import policyengine as pe
    except Exception as e:
        msg = f"policyengine not importable: {type(e).__name__}: {e}"
        pytest.fail(msg) if require else pytest.skip(msg)
    country_mod = getattr(pe, country, None)
    if country_mod is None:
        msg = f"policyengine has no {country} country model (base-only install)"
        pytest.fail(msg) if require else pytest.skip(msg)
    return country_mod.model


@pytest.mark.slow
@pytest.mark.parametrize("entry", core.PE_PARAMETERS,
                         ids=lambda e: f"{e['country']}:{e['path']}")
def test_curated_parameter_paths_resolve(entry):
    model = _pe_model(entry["country"])
    model.get_parameter(entry["path"])  # raises ValueError if renamed away


@pytest.mark.slow
@pytest.mark.parametrize("country", sorted(PE_VARIABLE_CONTRACT))
def test_hardcoded_variable_names_exist(country):
    model = _pe_model(country)
    missing = []
    for name in PE_VARIABLE_CONTRACT[country]:
        try:
            model.get_variable(name)
        except ValueError:
            missing.append(name)
    assert not missing, (
        f"{country} model no longer defines {missing}; "
        "_pe_summary/pe_population_impact would break or silently change"
    )


@pytest.mark.slow
def test_parameters_resolve_live_with_baselines():
    _pe_model("uk")  # skip cleanly when policyengine is not importable
    out = core.pe_list_common_parameters()
    broken = [(p["path"], p.get("live_error")) for p in out if not p.get("live")]
    assert not broken, f"parameter paths failed live resolution: {broken}"
    no_baseline = [p["path"] for p in out if p.get("baseline_value") is None]
    assert not no_baseline, f"no current baseline value for: {no_baseline}"
