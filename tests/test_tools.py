"""Tests for the four MCP tools."""

from __future__ import annotations

import pytest

from ppb_mcp.tools.headroom import get_gpu_headroom
from ppb_mcp.tools.list_configs import list_tested_configs
from ppb_mcp.tools.query import query_ppb_results
from ppb_mcp.tools.recommend import recommend_quantization


class TestListConfigs:
    @pytest.mark.asyncio
    async def test_returns_aggregated_unique_values(self, store) -> None:
        result = await list_tested_configs()
        assert "TestGPU-8GB" in result.gpus
        assert "TestGPU-16GB" in result.gpus
        assert "TestGPU-24GB" in result.gpus
        assert "Llama-7B" in result.models
        assert "Mistral-7B" in result.models
        assert "Q4_K_M" in result.quantizations
        assert "Q5_K_M" in result.quantizations
        assert result.total_benchmark_rows > 0
        assert result.last_updated != "never"


class TestQuery:
    @pytest.mark.asyncio
    async def test_no_filters_returns_stratified_sample(self, store) -> None:
        result = await query_ppb_results()
        assert result.total_count > 0
        # Stratified sample: each (gpu, model, quant) appears at most once.
        seen = set()
        for r in result.rows:
            key = (r.gpu_name, r.model, r.quantization)
            assert key not in seen, f"duplicate stratum: {key}"
            seen.add(key)

    @pytest.mark.asyncio
    async def test_filter_by_gpu_partial_match(self, store) -> None:
        result = await query_ppb_results(gpu_name="24GB")
        assert all("24GB" in r.gpu_name for r in result.rows)
        assert result.filtered_count > 0

    @pytest.mark.asyncio
    async def test_filter_by_quantization_exact(self, store) -> None:
        result = await query_ppb_results(quantization="Q4_K_M")
        assert all(r.quantization == "Q4_K_M" for r in result.rows)

    @pytest.mark.asyncio
    async def test_filter_by_vram_range(self, store) -> None:
        result = await query_ppb_results(vram_gb_min=10.0, vram_gb_max=20.0)
        assert all(10.0 <= r.vram_gb <= 20.0 for r in result.rows)

    @pytest.mark.asyncio
    async def test_filter_by_concurrent_users(self, store) -> None:
        result = await query_ppb_results(concurrent_users=4)
        assert all(r.concurrent_users == 4 for r in result.rows)

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self, store) -> None:
        result = await query_ppb_results(gpu_name="NoSuchGPU")
        assert result.rows == []
        assert result.filtered_count == 0
        assert result.total_count > 0

    @pytest.mark.asyncio
    async def test_limit_caps_results(self, store) -> None:
        result = await query_ppb_results(limit=2)
        assert len(result.rows) <= 2


class TestRecommend:
    @pytest.mark.asyncio
    async def test_high_confidence_when_many_rows(self, store) -> None:
        rec = await recommend_quantization(gpu_vram_gb=24.0, concurrent_users=1, model="Llama-7B")
        assert rec.recommended_quantization in {"Q4_K_M", "Q5_K_M", "Q8_0"}
        assert rec.confidence in {"high", "medium"}
        assert rec.estimated_tokens_per_second > 0
        assert rec.headroom_gb >= 0

    @pytest.mark.asyncio
    async def test_priority_quality_prefers_higher_bpw(self, store) -> None:
        rec = await recommend_quantization(
            gpu_vram_gb=24.0, concurrent_users=1, model="Llama-7B", priority="quality"
        )
        assert rec.recommended_quantization == "Q8_0"

    @pytest.mark.asyncio
    async def test_priority_speed_returns_a_quant(self, store) -> None:
        rec = await recommend_quantization(
            gpu_vram_gb=24.0, concurrent_users=1, model="Llama-7B", priority="speed"
        )
        # Speed prefers highest measured throughput.
        assert rec.recommended_quantization in {"Q4_K_M", "Q5_K_M", "Q8_0"}
        assert rec.confidence in {"high", "medium"}

    @pytest.mark.asyncio
    async def test_no_match_returns_low_confidence_or_empty(self, store) -> None:
        rec = await recommend_quantization(gpu_vram_gb=4.0, concurrent_users=32, model="Llama-13B")
        # Either no fit (recommended_quantization == "(none)") or low confidence formula.
        assert rec.confidence == "low"

    @pytest.mark.asyncio
    async def test_returns_alternatives(self, store) -> None:
        rec = await recommend_quantization(gpu_vram_gb=24.0, concurrent_users=1, model="Llama-7B")
        assert isinstance(rec.alternatives, list)

    @pytest.mark.asyncio
    async def test_recommend_floors_concurrent_users(self, store) -> None:
        # Fixture has rows for users in {1,2,4,8} on TestGPU-24GB.
        # Requesting 3 should floor to 2 and stay in Tier 1.
        rec = await recommend_quantization(gpu_vram_gb=24.0, concurrent_users=3, model="Llama-7B")
        assert rec.recommended_quantization != "(none)"
        assert "nearest tested benchmark" in rec.reasoning

    @pytest.mark.asyncio
    async def test_recommend_no_floor_note_on_exact_match(self, store) -> None:
        rec = await recommend_quantization(gpu_vram_gb=24.0, concurrent_users=2, model="Llama-7B")
        assert "nearest tested benchmark" not in rec.reasoning

    @pytest.mark.asyncio
    async def test_recommend_efficiency_priority_with_power_data(self, ppb_store) -> None:
        result = await recommend_quantization(
            gpu_vram_gb=24, concurrent_users=1, priority="efficiency"
        )
        assert result.recommended_quantization != "(none)"
        assert result.estimated_tokens_per_watt is not None
        assert result.estimated_tokens_per_watt > 0

    @pytest.mark.asyncio
    async def test_recommend_efficiency_priority_no_power_data(self, ppb_store_no_power) -> None:
        result = await recommend_quantization(
            gpu_vram_gb=24, concurrent_users=1, priority="efficiency"
        )
        assert result.recommended_quantization != "(none)"
        assert result.estimated_tokens_per_watt is None

    @pytest.mark.asyncio
    async def test_recommend_tier3_no_tokens_per_watt(self, ppb_store_empty) -> None:
        result = await recommend_quantization(
            gpu_vram_gb=24,
            concurrent_users=1,
            model="NonExistentModel-99B",
            priority="efficiency",
        )
        assert result.estimated_tokens_per_watt is None

    @pytest.mark.asyncio
    async def test_recommend_efficiency_reasoning_mentions_power(self, ppb_store) -> None:
        result = await recommend_quantization(
            gpu_vram_gb=24, concurrent_users=1, priority="efficiency"
        )
        if result.estimated_tokens_per_watt is not None:
            assert "tokens/sec/watt" in result.reasoning


def test_benchmark_row_power_fields() -> None:
    from ppb_mcp.models import BenchmarkRow

    row = BenchmarkRow(
        gpu_name="RTX 4090",
        vram_gb=24.0,
        model="Qwen3.5-9B",
        quantization="Q4_K_M",
        concurrent_users=1,
        tokens_per_second=85.0,
        avg_power_w=245.5,
        max_power_w=280.0,
        avg_gpu_temp_c=72.0,
        max_gpu_temp_c=78.0,
    )
    assert row.avg_power_w == 245.5
    assert row.max_gpu_temp_c == 78.0


class TestHeadroom:
    @pytest.mark.asyncio
    async def test_empirical_match(self, store) -> None:
        h = await get_gpu_headroom(
            gpu_name="TestGPU-24GB",
            quantization="Q4_K_M",
            model="Llama-7B",
            concurrent_users=1,
        )
        assert h.is_viable is True
        assert h.vram_available_gb == pytest.approx(24.0)
        assert h.max_safe_concurrent_users >= 1

    @pytest.mark.asyncio
    async def test_unknown_gpu_not_viable(self, store) -> None:
        h = await get_gpu_headroom(
            gpu_name="NonexistentGPU",
            quantization="Q4_K_M",
            model="Llama-7B",
        )
        assert h.is_viable is False
        assert h.warning is not None

    @pytest.mark.asyncio
    async def test_formula_fallback(self, store) -> None:
        # Llama-13B at Q8_0 not in fixture for 24 GB → formula path.
        h = await get_gpu_headroom(
            gpu_name="TestGPU-24GB",
            quantization="Q8_0",
            model="Llama-13B",
            concurrent_users=1,
        )
        # Either viable (most likely with 13B Q8_0 ≈ 15 GB on 24 GB) or warns.
        assert h.vram_available_gb == pytest.approx(24.0)
        assert h.vram_required_gb > 0
