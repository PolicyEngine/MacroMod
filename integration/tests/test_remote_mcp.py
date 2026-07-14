"""End-to-end test of the deployed MacroMod MCP server on Modal.

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
    "score_reform", "list_reform_variables", "forecast_uk",
    "latest_shocks", "model_summary",
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
async def test_score_reform_cgg():
    out = await _call(
        "score_reform", {"var": "CGG", "shock": 1250, "periods": 4}
    )
    assert len(out["results"]) >= 4
    # £1.25bn/quarter of extra spending should raise GDP by ~£1.25bn/quarter
    # in each shocked quarter (multiplier ~1 on impact in this model).
    first = out["results"][0]["delta_gdp_bn"]
    assert 0.8 < first < 2.0, first
    cum = out["cumulative_delta_gdp_bn_over_shock_periods"]
    assert 3.0 < cum < 8.0, cum


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
