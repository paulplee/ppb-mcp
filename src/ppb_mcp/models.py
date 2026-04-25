"""Pydantic response models for ppb-mcp tools."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BenchmarkRow(BaseModel):
    """Single benchmark result row, mapped from the 61-column raw schema."""

    gpu_name: str
    vram_gb: float
    model: str  # from model_base
    model_org: str | None = None
    model_full_path: str | None = None  # from raw `model` column
    quantization: str
    concurrent_users: int
    tokens_per_second: float  # from throughput_tok_s
    avg_ttft_ms: float | None = None
    p50_itl_ms: float | None = None
    n_ctx: int | None = None
    backend: str | None = None  # from `backends`
    submitter: str | None = None
    timestamp: str | None = None


class QueryResult(BaseModel):
    rows: list[BenchmarkRow]
    total_count: int = Field(..., description="Total rows in the dataset")
    filtered_count: int = Field(..., description="Rows matching the filters before `limit`")


class QuantizationRecommendation(BaseModel):
    recommended_quantization: str
    model: str
    gpu_vram_gb: float
    concurrent_users: int
    estimated_vram_usage_gb: float
    estimated_vram_per_user_gb: float
    estimated_tokens_per_second: float
    headroom_gb: float
    confidence: Literal["high", "medium", "low"]
    reasoning: str
    alternatives: list[str] = Field(default_factory=list)


class GPUHeadroom(BaseModel):
    gpu_name: str
    quantization: str
    model: str
    vram_required_gb: float
    vram_available_gb: float
    headroom_gb: float
    max_safe_concurrent_users: int
    is_viable: bool
    warning: str | None = None


class TestedConfigs(BaseModel):
    gpus: list[str]
    models: list[str]
    quantizations: list[str]
    total_benchmark_rows: int
    last_updated: str
