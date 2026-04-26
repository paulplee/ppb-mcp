"""Tests for the qualitative MCP tools."""

from __future__ import annotations

import pytest

from ppb_mcp.tools.compare_quants import compare_quants_qualitative
from ppb_mcp.tools.context_rot import get_context_rot_breakdown
from ppb_mcp.tools.qualitative_query import query_qualitative_results
from ppb_mcp.tools.qualitative_summary import get_qualitative_summary
from ppb_mcp.tools.tool_accuracy import get_tool_accuracy_breakdown


@pytest.mark.asyncio
async def test_get_qualitative_summary_found(qualitative_store):
    result = await get_qualitative_summary("Qwen3.5-0.8B", "Q4_K_M")
    assert result.context_rot_score == pytest.approx(0.45)
    assert result.overall_tool_accuracy == pytest.approx(0.22)
    assert result.mt_bench_score == pytest.approx(8.2)
    assert "context-rot" in result.phases_available
    assert "tool-accuracy" in result.phases_available
    assert "multiturn" in result.phases_available


@pytest.mark.asyncio
async def test_get_qualitative_summary_not_found(qualitative_store):
    result = await get_qualitative_summary("NonExistentModel", "Q4_K_M")
    assert result.context_rot_score is None
    assert result.phases_available == []


@pytest.mark.asyncio
async def test_context_rot_breakdown_parses_nested(qualitative_store):
    result = await get_context_rot_breakdown("Qwen3.5-0.8B", "Q4_K_M")
    assert result.by_length["4096"] == pytest.approx(0.9)
    assert result.by_length["131072"] == pytest.approx(0.0)
    assert result.by_needle["constellation"] == pytest.approx(0.0)
    assert result.cases_skipped == 5


@pytest.mark.asyncio
async def test_tool_accuracy_breakdown(qualitative_store):
    result = await get_tool_accuracy_breakdown("Qwen3.5-0.8B", "Q4_K_M")
    assert result.tool_selection_accuracy == pytest.approx(0.22)
    assert result.overall_tool_accuracy == pytest.approx(0.22)


@pytest.mark.asyncio
async def test_compare_quants_qualitative_insight(qualitative_store):
    result = await compare_quants_qualitative("Qwen3.5-0.8B")
    assert result.best_tool_accuracy == "Q4_0"
    assert "Q4_0" in result.insight
    assert len(result.rows) == 2  # Q4_K_M and Q4_0


@pytest.mark.asyncio
async def test_query_qualitative_results_filter_runner_type(qualitative_store):
    result = await query_qualitative_results(runner_type="tool-accuracy")
    assert result.filtered_count == 2
    assert all(r.runner_type == "tool-accuracy" for r in result.rows)


@pytest.mark.asyncio
async def test_query_qualitative_results_min_score_filter(qualitative_store):
    result = await query_qualitative_results(min_overall_tool_accuracy=0.30)
    assert result.filtered_count == 1
    assert result.rows[0].overall_tool_accuracy == pytest.approx(0.45)
