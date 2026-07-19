"""Tests for the FRB/US adapters.

These are the tests that would catch a BROKEN DEPLOYMENT rather than a broken
import: the hosted server can only produce right answers if the model files
survived the image's vendor/ pruning, the tracking invariant still holds, and a
known shock still produces the documented impulse responses.

Everything that needs a real solve is marked `slow` (run with `--runslow`) and
additionally skipped when `frbus` or its vendor data is unavailable, so a
checkout without the model repo still gets the fast contract coverage.

Magnitudes are asserted with deliberately loose tolerances against the figures
in us-frb-model's VALIDATION.md. They are NOT exact floats: LONGBASE is a
quarterly data vintage, and a re-vintaged baseline legitimately moves these
responses by a few percent. The tests are here to catch a sign flip, a
disconnected lever or a mis-specified baseline — not to freeze a data release.
"""

from __future__ import annotations

import pytest

from policyengine_macro import core


def _frbus_available() -> bool:
    try:
        core._frbus_repo()
    except Exception:
        return False
    return True


requires_frbus = pytest.mark.skipif(
    not _frbus_available(),
    reason="frbus (installed editably, with vendor/ present) is required",
)


# ---------------------------------------------------------------------------
# Fast tests: catalogue, metadata and argument validation. No solve.
# ---------------------------------------------------------------------------

def test_list_variables_documents_units_for_every_lever():
    """Units are the whole point of this catalogue: the same number means a
    100bp rate move on one lever and a solver divergence on another."""
    variables = core.frbus_list_variables()
    assert len(variables) >= 5
    for entry in variables:
        assert entry["var"]
        assert entry["description"]
        assert entry["units"], f"{entry['var']} has no documented units"
        assert isinstance(entry["typical_shock"], (int, float))
        assert entry["requires_policy_rule"] in (None, *core.FRBUS_POLICY_RULES)


def test_list_variables_returns_copies():
    """A caller mutating the result must not corrupt the module-level catalogue
    for every later request in a long-lived container."""
    core.frbus_list_variables()[0]["units"] = "corrupted"
    assert core.frbus_list_variables()[0]["units"] != "corrupted"


def test_every_policy_rule_is_self_consistent():
    for name, spec in core.FRBUS_POLICY_RULES.items():
        assert spec["description"]
        assert isinstance(spec["switches"], dict)
        assert isinstance(spec["exogenize"], list)
        lever = spec["shock_lever"]
        if lever is not None:
            entry = core._FRBUS_VAR_INDEX[lever]
            assert entry["requires_policy_rule"] == name, (
                f"{name}'s shock_lever {lever} does not declare it back"
            )


def test_summary_reports_validation_provenance():
    summary = core.frbus_summary()
    assert summary["equations"] == 284
    assert "2026" in summary["data_vintage"]
    val = summary["validation"]
    # The gates the model repo's own CI enforces; if these drift in the text
    # here they stop matching VALIDATION.md and the provenance becomes a lie.
    assert val["tracking_invariant"]["value"] < val["tracking_invariant"]["gate"]
    assert val["vs_vendor_pyfrbus"]["value"] < val["vs_vendor_pyfrbus"]["gate"]
    assert val["monetary_tightening_properties"]["xgdp_trough_pct"] < 0
    assert val["monetary_tightening_properties"]["lur_peak_pp"] > 0
    assert "MCE" in summary["expectations"]


def test_summary_states_there_is_no_reform_bridge():
    assert "NONE" in core.frbus_summary()["reform_bridge"]


@pytest.mark.parametrize("rule", ["taylor", "fixed_funds_rate"])
def test_rffintay_shock_is_refused_under_a_rule_that_ignores_it(rule):
    """The silent-zero trap. Under a non-inertial rule this shock is a valid
    column, the solve converges, and every response is exactly 0.0 — which
    reads as 'monetary policy does nothing'. It must raise instead."""
    with pytest.raises(ValueError, match="disconnected"):
        core.frbus_shock(var="rffintay_aerr", shock=1.0, policy_rule=rule)


def test_unknown_policy_rule_is_rejected():
    with pytest.raises(ValueError, match="policy_rule must be one of"):
        core.frbus_shock(var="rffintay_aerr", shock=1.0, policy_rule="nope")


@pytest.mark.parametrize("kwargs", [
    {"periods": 0},
    {"periods": 30, "horizon": 20},
])
def test_invalid_windows_are_rejected(kwargs):
    with pytest.raises(ValueError):
        core.frbus_shock(var="rffintay_aerr", shock=1.0, **kwargs)


def test_score_reform_refuses_frbus_and_names_the_alternative():
    """SCOPE GUARD: there is no PolicyEngine->FRB/US reform mapping, so
    score_reform must never quietly return a number for model='frbus'."""
    with pytest.raises(ValueError) as excinfo:
        core.score_reform(
            country="us",
            reform={"gov.irs.credits.ctc.amount.base[0].amount": 3000},
            model="frbus",
        )
    message = str(excinfo.value)
    assert "frbus_shock" in message
    assert "no mapping" in message.lower()


def test_score_reform_rejects_frbus_even_with_junk_other_arguments():
    """The bridge refusal must not depend on the rest of the call being
    well-formed, or a caller fixing their reform dict would eventually get a
    fabricated FRB/US answer."""
    with pytest.raises(ValueError, match="frbus_shock"):
        core.score_reform(country=None, reform=None, model="frbus")


def test_frbus_is_not_in_the_supported_score_models():
    assert "frbus" not in core.SCORE_MODELS
    assert "frbus" in core.SCORE_MODELS_WITHOUT_REFORM_BRIDGE


# ---------------------------------------------------------------------------
# Slow tests: real solves. These are the deployment gates.
# ---------------------------------------------------------------------------

@pytest.mark.slow
@requires_frbus
def test_model_and_data_files_resolve_after_image_pruning():
    """The Modal image drops ~98% of us-frb-model's bytes. If the pruning ever
    takes one of these two files, every FRB/US tool 500s in production."""
    repo = core._frbus_repo()
    assert (repo / "vendor" / "data_only_package" / "LONGBASE.TXT").exists()
    assert (repo / "vendor" / "pyfrbus_package" / "models" / "model.xml").exists()


@pytest.mark.slow
@requires_frbus
def test_tracking_invariant_holds_after_init_trac():
    """VALIDATION.md Test 1, re-run through OUR adapter's baseline setup.

    After init_trac, solving the baseline must reproduce LONGBASE for all 284
    endogenous variables. If this fails, every deviation the shock tool reports
    is measured against a baseline that is already wrong, so nothing else in
    this file means anything. The model repo gates at 1e-8; we assert the same
    bound rather than the achieved 5.6e-17, so a solver change that stays
    within the published gate does not fail us.
    """
    import pandas as pd

    start = pd.Period(core.FRBUS_DEFAULT_START, freq="Q")
    end = start + core.FRBUS_DEFAULT_HORIZON - 1
    model, with_adds = core._frbus_baseline("inertial_taylor", start, end)

    solved = model.solve(start, end, with_adds)
    worst = max(
        float((solved.loc[start:end, v] - with_adds.loc[start:end, v]).abs().max())
        for v in model.endo_names
    )
    assert len(model.endo_names) == 284
    assert worst < 1e-8, f"tracking invariant broken: max abs error {worst:.2e}"


@pytest.mark.slow
@requires_frbus
def test_monetary_tightening_has_the_documented_signs_and_magnitudes():
    """VALIDATION.md Test 3: 100bp rffintay_aerr shock in 2026Q1.

    Documented: rff +1.00pp on impact, xgdp trough -0.55%, lur peak +0.26pp,
    picxfe trough -0.034pp. Asserted with wide bands (roughly +/-40% on the
    real-side magnitudes) so a LONGBASE re-vintage does not break the build,
    while a sign flip, a decoupled lever or an order-of-magnitude error does.
    """
    res = core.frbus_shock(var="rffintay_aerr", shock=1.0)

    assert res["policy_rule"] == "inertial_taylor"
    assert "warning" not in res, res.get("warning")
    assert len(res["results"]) == core.FRBUS_DEFAULT_HORIZON

    # Impact: the shock is a 100bp tightening and must show up as one.
    impact = res["results"][0]
    assert impact["period"] == "2026Q1"
    assert 0.9 < impact["rff"] < 1.1

    # Monetary tightening contracts output and raises unemployment.
    gdp = [r["xgdp"] for r in res["results"]]
    lur = [r["lur"] for r in res["results"]]
    picxfe = [r["picxfe"] for r in res["results"]]

    assert -0.85 < min(gdp) < -0.30, f"xgdp trough {min(gdp)} (expected ~-0.55%)"
    assert 0.15 < max(lur) < 0.40, f"lur peak {max(lur)} (expected ~+0.26pp)"
    assert min(picxfe) < 0, "core inflation must fall after a tightening"
    assert abs(min(picxfe)) < 0.5, "inflation response is implausibly large"

    # The trough is a lagged response, not an impact effect: FRB/US puts it
    # around 2027Q4. Anything in the first year would mean the propagation
    # dynamics are wrong even if the peak magnitude happened to look right.
    trough_period = res["peaks"]["xgdp"]["period"]
    assert trough_period.startswith(("2027", "2028")), trough_period
    assert res["peaks"]["xgdp"]["value"] == pytest.approx(min(gdp))


@pytest.mark.slow
@requires_frbus
def test_fixed_funds_rate_amplifies_a_demand_shock():
    """The economically load-bearing reason policy_rule is exposed.

    With the funds rate held on its baseline path there is no endogenous
    monetary offset, so the SAME demand shock must move GDP by more than under
    the inertial Taylor rule. If the rule switch were silently ignored the two
    runs would be identical — which is exactly the bug this catches.
    """
    taylor = core.frbus_shock(
        var="egfe_aerr", shock=0.01, periods=4, policy_rule="inertial_taylor")
    fixed = core.frbus_shock(
        var="egfe_aerr", shock=0.01, periods=4, policy_rule="fixed_funds_rate")

    taylor_peak = taylor["peaks"]["xgdp"]["value"]
    fixed_peak = fixed["peaks"]["xgdp"]["value"]

    assert taylor_peak > 0, "a government purchases increase must raise GDP"
    assert fixed_peak > taylor_peak, (
        f"fixed funds rate ({fixed_peak}) should amplify the multiplier "
        f"relative to the inertial Taylor rule ({taylor_peak})"
    )
    # Under the Taylor rule the funds rate responds; under the fixed rule it
    # must not move at all.
    assert abs(fixed["peaks"]["rff"]["value"]) < 1e-6
    assert abs(taylor["peaks"]["rff"]["value"]) > 1e-6


@pytest.mark.slow
@requires_frbus
def test_shock_sign_reverses_with_the_sign_of_the_shock():
    """Cheap symmetry check that the reported deviations really are driven by
    the shock argument, not by a baseline artefact."""
    up = core.frbus_shock(var="trp_aerr", shock=0.01, periods=4)
    down = core.frbus_shock(var="trp_aerr", shock=-0.01, periods=4)
    # A personal tax RISE contracts output; a cut expands it.
    assert min(r["xgdp"] for r in up["results"]) < 0
    assert max(r["xgdp"] for r in down["results"]) > 0


@pytest.mark.slow
@requires_frbus
def test_requested_extra_variables_are_returned_alongside_the_headline():
    res = core.frbus_shock(
        var="rffintay_aerr", shock=1.0, horizon=8, variables=["ecnia", "ebfi"])
    row = res["results"][0]
    for series in ("xgdp", "lur", "picxfe", "pcpi", "rff", "ecnia", "ebfi"):
        assert series in row
        assert series in res["series_meaning"]
    assert len(res["results"]) == 8


@pytest.mark.slow
@requires_frbus
def test_unknown_variables_are_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="frbus_list_variables"):
        core.frbus_shock(var="not_a_variable", shock=1.0)
    with pytest.raises(ValueError, match="not in the model"):
        core.frbus_shock(var="rffintay_aerr", shock=1.0, variables=["nope"])


@pytest.mark.slow
@requires_frbus
def test_an_absurdly_sized_shock_errors_instead_of_returning_nonsense():
    """egfe_aerr is in log points; a caller treating it as billions of dollars
    passes something like 10. The solver diverges, and the adapter must
    surface that as a units error rather than a raw solver traceback."""
    with pytest.raises(ValueError, match="units"):
        core.frbus_shock(var="egfe_aerr", shock=50.0)


@pytest.mark.slow
@requires_frbus
def test_baseline_is_cached_per_policy_rule():
    core._FRBUS_BASELINE_CACHE.clear()
    core.frbus_shock(var="rffintay_aerr", shock=1.0, horizon=8)
    assert len(core._FRBUS_BASELINE_CACHE) == 1
    core.frbus_shock(var="rffintay_aerr", shock=0.5, horizon=8)
    assert len(core._FRBUS_BASELINE_CACHE) == 1, "identical window re-solved"
    core.frbus_shock(var="egfe_aerr", shock=0.01, horizon=8,
                     policy_rule="fixed_funds_rate")
    assert len(core._FRBUS_BASELINE_CACHE) == 2
