"""End-to-end test of the deployed PolicyEngine Macro MCP server on Modal.

Talks to the live deployment over streamable HTTP, so it needs network and a
deployed app. Skipped unless MACROMOD_REMOTE_TESTS=1 (CI without Modal auth
stays green). Run:

    MACROMOD_REMOTE_TESTS=1 python -m pytest tests/test_remote_mcp.py -v
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MACROMOD_REMOTE_TESTS") != "1",
    reason="set MACROMOD_REMOTE_TESTS=1 to hit the live Modal deployment",
)

URL = os.environ.get(
    "MACROMOD_REMOTE_URL",
    "https://policyengine--macromod-mcp-serve.modal.run/mcp",
)

EXPECTED_TOOLS = {
    "score_reform", "obr_shock", "list_reform_variables", "forecast_uk",
    "latest_shocks", "model_summary", "calculate_household",
    "household_reform_impact", "list_reform_parameters",
    "population_reform_impact",
}


async def _call(tool: str, args: dict | None = None):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(URL, timeout=300) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args or {})
            assert not result.isError, result.content
            return json.loads(result.content[0].text)


@pytest.mark.anyio
async def test_liveness_handshake():
    """Canonical uptime probe for the deployed server.

    A full MCP handshake over streamable HTTP: initialize (server responds and
    identifies itself), tools/list (the expected tools are advertised), and one
    instant tool call (model_summary returns real content). If this passes, the
    live Modal endpoint is up and answering correctly.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(URL, timeout=60) as (read, write, _):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            assert init.serverInfo is not None, "no serverInfo in initialize result"

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert EXPECTED_TOOLS <= names, names

            res = await session.call_tool("model_summary", {})
            assert not res.isError, res.content
            out = json.loads(res.content[0].text)
            assert "replication" in out, out


@pytest.mark.anyio
async def test_lists_all_five_tools():
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(URL, timeout=60) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
    assert EXPECTED_TOOLS <= names, names


@pytest.mark.anyio
async def test_model_summary_returns_fevd():
    out = await _call("model_summary")
    fevd = out["replication"]["fevd_1yr_headline"]
    assert fevd, "FEVD table missing — results/summary.md not baked in?"
    assert any("GDP" in json.dumps(row) for row in fevd)


@pytest.mark.anyio
async def test_obr_shock_cgg():
    out = await _call(
        "obr_shock", {"var": "CGG", "shock": 1250, "periods": 4}
    )
    assert len(out["results"]) >= 4
    # £1.25bn/quarter of extra spending should raise GDP by ~£1.25bn/quarter
    # in each shocked quarter (multiplier ~1 on impact in this model).
    first = out["results"][0]["delta_gdp_bn"]
    assert 0.8 < first < 2.0, first
    cum = out["cumulative_delta_gdp_bn_over_shock_periods"]
    assert 3.0 < cum < 8.0, cum


@pytest.mark.slow
@pytest.mark.anyio
async def test_score_reform_investment_closure_bounded_and_signed():
    """Regression guard for the corp-tax investment-closure instability.

    This is the exact path that once diverged to +/-£7tn in production while the
    liveness/tools smoke test stayed green. A corporation-tax (TCPRO) CUT scored
    WITH the investment closure must (a) stay bounded — |Δ business investment|
    well under £50bn/quarter — and (b) be correctly signed: a cut lowers the
    user cost of capital and RAISES investment. Runs a real closure solve on the
    server (the stabilisation adds a tracking pass), so it is marked `slow` and
    run in the scheduled full-validation workflow rather than on every deploy.
    """
    import asyncio

    try:
        out = await asyncio.wait_for(
            _call(
                "obr_shock",
                {"var": "TCPRO", "shock": -0.05, "periods": 4,
                 "investment_closure": True},
            ),
            timeout=420,
        )
    except asyncio.TimeoutError:
        pytest.fail(
            "investment-closure solve did not return within 420s on the server "
            "— too slow (or the container timed out). The closure invariant is "
            "hard-gated locally in the OBR repo CI."
        )
    ifs = [r["delta_if_m"] for r in out["results"] if r["delta_if_m"] is not None]
    assert ifs, "no investment deltas returned"
    peak = max(abs(x) for x in ifs)
    assert peak < 50_000, f"investment response £{peak:,.0f}m — closure diverging again"
    assert ifs[-1] > 0, f"corp-tax CUT should raise investment, got {ifs[-1]:+,.0f}"


@pytest.mark.anyio
async def test_score_reform_obr_refuses_corp_tax_actionably():
    """The OBR bridge (#9) must refuse corporation-tax reforms — they are not
    household-borne, so the static-costing bridge cannot carry them — with a
    pointer to the direct obr_shock TCPRO lever, not silently or cryptically."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(URL, timeout=60) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(
                "score_reform",
                {"country": "uk",
                 "reform": {"gov.hmrc.corporation_tax.main_rate": 0.28},
                 "model": "obr"},
            )
    assert res.isError
    assert "obr_shock" in json.dumps([c.text for c in res.content])


@pytest.mark.anyio
async def test_score_reform_microsim_scoreresult():
    """score_reform(model='microsim') on the hosted server: a basic-rate
    rise must raise revenue, and the common ScoreResult block (#10) must be
    present and coherent (units, basis, distributional)."""
    import asyncio

    out = await asyncio.wait_for(
        _call(
            "score_reform",
            {"country": "uk",
             "reform": {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21},
             "model": "microsim"},
        ),
        timeout=300,
    )
    assert out["budgetary_impact_bn"] > 0, out["budgetary_impact_bn"]
    score = out["score"]
    assert score["model_class"] == "microsim"
    rev = score["quantities"]["revenue"]
    assert rev["delta_bn"] == out["budgetary_impact_bn"]
    assert rev["units"] and rev["basis"]
    assert score["distributional"]["decile_impacts"]


@pytest.mark.slow
@pytest.mark.anyio
async def test_score_reform_obr_bridge_end_to_end():
    """The full #9 pipeline on the hosted server: microsim static costing →
    HHDI shock path → OBR second-round effects. A basic-rate RISE raises
    revenue, so disposable income and hence GDP must FALL, by a sane
    magnitude. One-year window to keep the runtime bounded; slow-marked, so
    it runs in the scheduled full validation, not on every deploy."""
    import asyncio

    out = await asyncio.wait_for(
        _call(
            "score_reform",
            {"country": "uk",
             "reform": {"gov.hmrc.income_tax.rates.uk[0].rate": 0.21},
             "model": "obr", "years": 1},
        ),
        timeout=540,
    )
    costing = out["annual_costings_bn"][0]["budgetary_impact_bn"]
    assert costing > 0, costing
    # Sign: revenue raised => HHDI falls => GDP falls.
    assert all(q < 0 for q in out["quarterly_shock_path_m"])
    cum = out["cumulative_delta_gdp_bn_over_shock_periods"]
    assert cum < 0, cum
    # Magnitude: |cumulative GDP effect| within ~2x the annual costing
    # (demand-side multiplier of order 1 over the shocked year).
    assert abs(cum) < 2 * costing + 1, (cum, costing)
    score = out["score"]
    assert score["model_class"] == "semi-structural"
    assert score["caveats"]


@pytest.mark.anyio
async def test_forecast_uk_small():
    out = await _call("forecast_uk", {"horizons": 4, "draws": 200})
    assert out["horizons"] == 4
    assert len(out["gdp_growth_yoy"]) == 4
    q0 = out["gdp_growth_yoy"][0]
    assert q0["lo90"] <= q0["median"] <= q0["hi90"]
    assert len(out["cpi_inflation_yoy"]) == 4


@pytest.mark.anyio
async def test_calculate_household_uk_50k():
    """Smoke test of the PolicyEngine engine on the hosted server: a single
    UK earner on £50k should net ~£39,519 in 2026. Catches engine/package
    breakage in the deployed image (issue #2 hardening)."""
    out = await _call(
        "calculate_household",
        {"country": "uk", "people": [{"age": 35, "employment_income": 50_000}]},
    )
    ni = out["summary"]["household_net_income"]
    assert 38_500 < ni < 40_500, ni
    assert out["summary"]["income_tax_by_person"][0] > 6_000


@pytest.fixture
def anyio_backend():
    return "asyncio"
