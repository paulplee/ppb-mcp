"""Tests for the FastMCP server registration."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_four_tools_registered() -> None:
    from ppb_mcp.server import app

    tools = await app.list_tools()
    names = {t.name for t in tools}
    assert {"list_tested_configs", "query_ppb_results", "recommend_quantization", "get_gpu_headroom"} <= names


@pytest.mark.asyncio
async def test_tool_schemas_have_descriptions() -> None:
    from ppb_mcp.server import app

    tools = await app.list_tools()
    by_name = {t.name: t for t in tools}
    for name in (
        "list_tested_configs",
        "query_ppb_results",
        "recommend_quantization",
        "get_gpu_headroom",
    ):
        assert by_name[name].description, f"{name} is missing a description"
