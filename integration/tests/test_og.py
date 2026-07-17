"""Tests for the OG-UK steady-state adapters.

Fast tests mock the oguk solver and policy construction (a real solve takes
minutes). One slow end-to-end test runs a real reform score.

OG-UK (oguk) is a local-only tool: it is deliberately excluded from the Modal
deployment image (see modal_app.py). oguk 0.3.0 calibrates its tax functions
from the PolicyEngine enhanced-FRS microdata and pins policyengine-uk==2.88.0;
policyengine-uk >= 2.89 renamed the dataset keys (enhanced_frs_2023_24_* ->
populace_uk_*), so calibration KeyError-fails under a newer PE. The macromod
package itself requires policyengine[models]>=4 (which brings pe-uk >= 2.89 for
the household/population tools), so a single env cannot satisfy both. The real
end-to-end solve is therefore skipped when the installed PE is incompatible
with oguk's calibration; the adapter's translation of that failure into an
actionable error is covered by test_og_dataset_keyerror_translated.
"""

import json
import os

import pytest

from macromod import core


def _oguk_calibration_skip_reason():
    """Return a skip reason if a real oguk solve can't calibrate here, else None."""
    try:
        import oguk  # noqa: F401
    except ImportError:
        return "oguk not installed (local-only tool; excluded from Modal image)"
    try:
        from importlib.metadata import version

        peuk = version("policyengine-uk")
        major, minor = (int(x) for x in peuk.split(".")[:2])
    except Exception:
        return "policyengine-uk not importable for oguk calibration"
    if (major, minor) >= (2, 89):
        return (
            f"oguk 0.3.0 needs policyengine-uk==2.88.x; installed {peuk} renamed "
            "the enhanced-FRS dataset keys (calibration KeyError). Error path is "
            "covered by test_og_dataset_keyerror_translated."
        )
    if not (os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")):
        return "set HUGGING_FACE_TOKEN to download the enhanced-FRS microdata"
    return None


_OGUK_SKIP = _oguk_calibration_skip_reason()


class _FakeSS:
    """Stands in for oguk.SteadyStateResult."""

    def __init__(self, scale=1.0):
        self._d = {
            "r": 0.05, "w": 1.2, "Y": 2.0 * scale, "K": 6.0 * scale,
            "L": 1.0 * scale, "C": 1.4 * scale, "I": 0.4 * scale,
            "G": 0.2 * scale, "tax_revenue": 0.6 * scale, "debt": 1.8 * scale,
        }

    def model_dump(self):
        return dict(self._d)


class _FakeImpact:
    def model_dump(self):
        d = {}
        for k in ("gdp", "consumption", "investment", "government",
                  "tax_revenue", "debt"):
            d[k] = 100.0
            d[f"{k}_change"] = 1.5
            d[f"{k}_pct"] = 0.05
        d["r_baseline"] = 0.05
        d["r_reform"] = 0.051
        return d


@pytest.fixture
def fake_oguk(monkeypatch):
    calls = {"solve": []}

    def fake_solve(start_year=2026, policy=None, max_iter=250, **kw):
        calls["solve"].append({"start_year": start_year, "policy": policy,
                               "max_iter": max_iter})
        return _FakeSS(scale=1.0 if policy is None else 1.01)

    def fake_map(baseline, reform):
        return _FakeImpact()

    monkeypatch.setattr(core, "_import_oguk", lambda: (fake_solve, fake_map))
    def fake_build_policy(reform, start_year):
        calls.setdefault("policy_reforms", []).append(dict(reform))
        return object()

    monkeypatch.setattr(core, "_og_build_policy", fake_build_policy)
    monkeypatch.setattr(core, "_OG_BASELINE_CACHE", {})
    return calls


def test_og_baseline_shape(fake_oguk):
    res = core.og_baseline(start_year=2026)
    assert res["start_year"] == 2026
    ss = res["steady_state_model_units"]
    assert {"r", "w", "Y", "K", "L", "C", "I", "G", "tax_revenue",
            "debt"} == set(ss)
    assert "pooled ages" in res["assumptions"]
    json.dumps(res)


def test_og_score_reform_shape_and_mapping(fake_oguk):
    res = core.og_score_reform({"gov.hmrc.income_tax.rates.uk[0].rate": 0.21})
    imp = res["impact"]
    assert imp["levels_bn"]["gdp"] == 100.0
    assert imp["changes_bn"]["gdp_change"] == 1.5
    assert imp["changes_pct"]["tax_revenue_pct"] == 0.05
    assert imp["interest_rate"] == {"baseline": 0.05, "reform": 0.051}
    assert res["reform"] == {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}
    assert res["start_year"] == 2026
    # Baseline solved without a policy, reform with one.
    solves = fake_oguk["solve"]
    assert len(solves) == 2
    assert solves[0]["policy"] is None
    assert solves[1]["policy"] is not None
    json.dumps(res)


def test_og_baseline_cache_reused(fake_oguk):
    core.og_score_reform({"gov.hmrc.income_tax.rates.uk[0].rate": 0.21})
    core.og_score_reform({"gov.hmrc.income_tax.rates.uk[0].rate": 0.22})
    # 2 reform solves + only 1 baseline solve (cached).
    baselines = [c for c in fake_oguk["solve"] if c["policy"] is None]
    assert len(baselines) == 1
    assert len(fake_oguk["solve"]) == 3


def test_og_baseline_cache_bypass(fake_oguk):
    core.og_score_reform({"x": 0.21}, baseline_cache=False)
    core.og_score_reform({"x": 0.21}, baseline_cache=False)
    baselines = [c for c in fake_oguk["solve"] if c["policy"] is None]
    assert len(baselines) == 2


def test_og_dataset_keyerror_translated(monkeypatch):
    """A calibration KeyError becomes an actionable RuntimeError."""

    def broken_solve(**kw):
        raise KeyError("enhanced_frs_2023_24_2026")

    monkeypatch.setattr(core, "_import_oguk",
                        lambda: (broken_solve, lambda b, r: None))
    monkeypatch.setattr(core, "_OG_BASELINE_CACHE", {})
    with pytest.raises(RuntimeError) as exc:
        core.og_baseline()
    msg = str(exc.value)
    assert "HUGGING_FACE_TOKEN" in msg
    assert "ensure_datasets" in msg
    assert "policyengine-uk==2.88.0" in msg


@pytest.mark.slow
def test_og_build_policy_real():
    """Real PolicyEngine Policy construction (imports policyengine, ~20s)."""
    policy = core._og_build_policy(
        {
            "gov.hmrc.income_tax.rates.uk[0].rate": 0.21,
            "gov.hmrc.income_tax.allowances.personal_allowance.amount": 15000,
        },
        2026,
    )
    assert len(policy.parameter_values) == 2
    pv = policy.parameter_values[0]
    assert pv.value == 0.21
    assert pv.start_date.year == 2026
    assert pv.parameter.name == "gov.hmrc.income_tax.rates.uk[0].rate"


@pytest.mark.slow
@pytest.mark.skipif(_OGUK_SKIP is not None, reason=_OGUK_SKIP or "")
def test_og_score_reform_end_to_end():
    """Full baseline + reform steady-state solves (~10+ minutes)."""
    res = core.og_score_reform({"gov.hmrc.income_tax.rates.uk[0].rate": 0.21})
    imp = res["impact"]
    # A basic-rate rise should raise long-run tax revenue.
    assert imp["changes_bn"]["tax_revenue_change"] > 0
    assert res["baseline_steady_state_model_units"]["Y"] > 0
    json.dumps(res)


# ---------------------------------------------------------------------------
# Unified score_reform dispatcher (one reform vocabulary across the suite)
# ---------------------------------------------------------------------------

def test_og_multi_parameter_reform_passes_through(fake_oguk):
    """The full multi-parameter dict reaches policy construction (fast guard
    for the PR's central behavior; real Policy construction is slow-marked)."""
    reform = {
        "gov.hmrc.income_tax.rates.uk[0].rate": 0.21,
        "gov.hmrc.income_tax.allowances.personal_allowance.amount": 15000,
    }
    core.og_score_reform(reform)
    assert fake_oguk["policy_reforms"] == [reform]


def test_score_reform_routes_og(fake_oguk):
    res = core.score_reform(
        "uk", {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}, model="og"
    )
    assert res["model"].startswith("OG-UK")
    assert res["reform"] == {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21}
    assert len(fake_oguk["solve"]) == 2


def test_score_reform_og_is_uk_only(fake_oguk):
    with pytest.raises(ValueError, match="UK-only"):
        core.score_reform("us", {"x": 1}, model="og")


def test_score_reform_obr_is_uk_only():
    with pytest.raises(ValueError, match="UK-only"):
        core.score_reform("us", {"x": 1}, model="obr")


def test_score_reform_obr_corp_tax_points_to_escape_hatch():
    """Corporation tax is not household-borne in the microsim: the bridge must
    refuse it (before any heavy import) and point at the TCPRO lever."""
    with pytest.raises(ValueError) as exc:
        core.score_reform(
            "uk", {"gov.hmrc.corporation_tax.main_rate": 0.20}, model="obr"
        )
    msg = str(exc.value)
    assert "obr_shock" in msg
    assert "TCPRO" in msg


def test_score_reform_validates_reform_and_model():
    with pytest.raises(ValueError, match="non-empty"):
        core.score_reform("uk", {}, model="og")
    with pytest.raises(ValueError, match="non-empty"):
        core.score_reform("uk", "not-a-dict", model="og")
    with pytest.raises(ValueError, match="model must be one of"):
        core.score_reform("uk", {"x": 1}, model="svar")
