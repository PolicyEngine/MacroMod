"""Contract test: the MCP tool surface is exactly the golden list.

Fast and dependency-free (importing mcp_server does NOT import PolicyEngine —
that import is lazy), so this runs as a hard gate on every PR. It guards the
published API: adding, removing or renaming a tool fails CI loudly instead of
silently reshaping what hosted clients see.
"""

from __future__ import annotations

import asyncio

import pytest

from policyengine_macro import mcp_server

from tool_surface import GOLDEN_TOOL_COUNT, GOLDEN_TOOLS, assert_surface


def _registered():
    return asyncio.run(mcp_server.mcp.list_tools())


def test_tool_names_are_exactly_the_golden_set():
    assert_surface(t.name for t in _registered())


def test_tool_count_matches_the_golden_count():
    assert len(_registered()) == GOLDEN_TOOL_COUNT


def test_no_duplicate_tool_names():
    names = [t.name for t in _registered()]
    assert len(names) == len(set(names)), names


@pytest.mark.parametrize("name", sorted(GOLDEN_TOOLS))
def test_every_golden_tool_is_registered_and_described(name):
    """Each advertised tool must exist and carry a docstring-derived
    description plus an object input schema — that text is the only thing a
    model sees when choosing a tool."""
    tool = next((t for t in _registered() if t.name == name), None)
    assert tool is not None, f"{name} is missing from the server"
    assert tool.description and len(tool.description) > 40, tool.description
    assert tool.inputSchema.get("type") == "object", tool.inputSchema


def test_household_tools_do_not_require_arguments_at_schema_level():
    """Regression guard for #38: country/people are validated in core so the
    caller gets a sentence, not a pydantic 'Field required' dump. If they were
    marked required in the JSON schema again, the raw dump would come back."""
    tools = {t.name: t for t in _registered()}
    for name in ("calculate_household", "household_reform_impact"):
        required = tools[name].inputSchema.get("required", [])
        assert "country" not in required and "people" not in required, (
            f"{name} re-declares country/people as schema-required: {required}"
        )
