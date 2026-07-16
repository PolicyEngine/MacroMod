"""End-to-end MCP server test over stdio using the official mcp client."""

import json
import sys

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(command=sys.executable, args=["-m", "macromod.mcp_server"])

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


def _payload(result):
    assert not result.isError, result.content
    return json.loads(result.content[0].text)


@pytest.mark.slow
@pytest.mark.anyio
async def test_mcp_tools_and_calls():
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert EXPECTED_TOOLS <= names

            res = await session.call_tool("model_summary", {})
            summary = _payload(res)
            assert "replication" in summary

            res = await session.call_tool(
                "obr_shock", {"var": "CGG", "shock": 1250, "periods": 4}
            )
            score = _payload(res)
            assert score["var"] == "CGG"
            assert score["results"][0]["delta_gdp_bn"] > 0

            res = await session.call_tool("list_reform_parameters", {})
            assert not res.isError, res.content
            if res.structuredContent:  # mcp >= 1.9 wraps lists as {"result": [...]}
                params = res.structuredContent.get("result", res.structuredContent)
            else:
                params = [json.loads(c.text) for c in res.content]
            assert any(
                p["path"] == "gov.hmrc.income_tax.rates.uk[0].rate" for p in params
            )

            # First policyengine call in the server pays the lazy import
            # (~20s) plus the calculation.
            res = await session.call_tool(
                "calculate_household",
                {
                    "country": "uk",
                    "people": [{"age": 35, "employment_income": 50_000}],
                },
            )
            hh = _payload(res)
            assert hh["country"] == "uk"
            assert 6_500 < hh["summary"]["income_tax_by_person"][0] < 8_500
            json.dumps(hh)


@pytest.fixture
def anyio_backend():
    return "asyncio"
