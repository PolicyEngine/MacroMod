"""The golden MCP tool surface.

This is the published contract of the hosted PolicyEngine Macro MCP server:
the exact set of tool names, and nothing else. Clients (Claude, the docs site,
downstream agents) bind to these names, so adding, removing or renaming a tool
is a breaking change to the public API and must be a deliberate, reviewed edit
of this list — never an accident that ships silently.

Both the in-process test (tests/test_tool_surface.py) and the live-deployment
test (tests/test_remote_mcp.py) assert equality against this set, so a drift
between the code and the deployment is also caught.
"""

from __future__ import annotations

GOLDEN_TOOLS = frozenset(
    {
        "calculate_household",
        "dynamic_reform_impact",
        "forecast_uk",
        "format_score_report",
        "frbus_list_variables",
        "frbus_shock",
        "frbus_summary",
        "household_reform_impact",
        "latest_shocks",
        "list_model_capabilities",
        "list_reform_parameters",
        "list_reform_variables",
        "model_summary",
        "get_model_status",
        "obr_shock",
        "population_reform_impact",
        "score_reform",
        "recommend_model",
    }
)

GOLDEN_TOOL_COUNT = 18

assert len(GOLDEN_TOOLS) == GOLDEN_TOOL_COUNT


def assert_surface(names) -> None:
    """Assert an observed tool-name collection is exactly the golden surface."""
    observed = set(names)
    missing = GOLDEN_TOOLS - observed
    extra = observed - GOLDEN_TOOLS
    assert not missing and not extra, (
        "MCP tool surface changed — this is a breaking API change.\n"
        f"  removed/renamed away: {sorted(missing) or 'none'}\n"
        f"  added/renamed in:     {sorted(extra) or 'none'}\n"
        "If intentional, update GOLDEN_TOOLS in tests/tool_surface.py "
        "in the same PR."
    )
    assert len(observed) == GOLDEN_TOOL_COUNT, (
        f"expected exactly {GOLDEN_TOOL_COUNT} tools, got {len(observed)}: "
        f"{sorted(observed)}"
    )
