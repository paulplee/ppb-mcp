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
    runner_type: str | None = None  # "llama-bench" | "llama-server" | "llama-server-loadtest"
    avg_power_w: float | None = None
    max_power_w: float | None = None
    avg_gpu_temp_c: float | None = None
    max_gpu_temp_c: float | None = None
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
    estimated_tokens_per_watt: float | None = Field(
        default=None,
        description=(
            "Tokens per second per watt — only populated when avg_power_w data "
            "is available in the dataset for this configuration. None means no "
            "power data exists for this recommendation."
        ),
    )
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
    runner_types: list[str] = Field(
        default_factory=list,
        description="Available runner_type values; filter by these in query_ppb_results",
    )
    total_benchmark_rows: int
    last_updated: str


# ─── Qualitative benchmark models ────────────────────────────────────────────


class QualitativeRow(BaseModel):
    """Single qualitative benchmark result row."""

    gpu_name: str
    model: str
    quantization: str
    run_type: Literal["qualitative"] = "qualitative"
    runner_type: str  # 'context-rot' | 'tool-accuracy' | 'answer-quality' | 'multiturn'
    suite_id: str | None = None
    benchmark_version: str | None = None
    # Context rot phase
    context_rot_score: float | None = None
    context_rot_accuracy_by_length: dict[str, float | None] | None = None
    context_rot_accuracy_by_depth: dict[str, float] | None = None
    context_rot_accuracy_by_needle: dict[str, float] | None = None
    cases_skipped_context: int | None = None
    # Tool accuracy phase
    tool_selection_accuracy: float | None = None
    parameter_accuracy: float | None = None
    parameter_hallucination_rate: float | None = None
    parse_success_rate: float | None = None
    no_call_accuracy: float | None = None
    overall_tool_accuracy: float | None = None
    # Answer quality phase
    knowledge_accuracy_mean: float | None = None
    knowledge_accuracy_std: float | None = None
    answer_relevancy_mean: float | None = None
    coherence_mean: float | None = None
    quality_composite_score: float | None = None
    # Multi-turn phase
    memory_accuracy: float | None = None
    mt_bench_score: float | None = None
    cases_evaluated: int | None = None


class QualitativeSummary(BaseModel):
    """All available qualitative scores for one (gpu, model, quant) combination."""

    gpu_name: str
    model: str
    quantization: str | None = None
    context_rot_score: float | None = None
    overall_tool_accuracy: float | None = None
    quality_composite_score: float | None = None
    mt_bench_score: float | None = None
    memory_accuracy: float | None = None
    phases_available: list[str] = Field(
        default_factory=list,
        description="Which qualitative phases have data for this config",
    )
    suite_id: str | None = None
    benchmark_version: str | None = None


class ContextRotBreakdown(BaseModel):
    """Context-rot scores broken down by context length, depth, and needle type."""

    model: str
    quantization: str
    gpu_name: str
    overall_score: float | None = None
    by_length: dict[str, float | None] = Field(
        default_factory=dict,
        description="Score at each context length in tokens, e.g. {'4096': 0.9, '131072': 0.0}",
    )
    by_depth: dict[str, float] = Field(
        default_factory=dict,
        description="Score at each depth position %, e.g. {'10': 0.8, '90': 0.5}",
    )
    by_needle: dict[str, float] = Field(
        default_factory=dict,
        description="Score per needle type, e.g. {'code': 1.0, 'constellation': 0.0}",
    )
    cases_skipped: int | None = None


class ToolAccuracyBreakdown(BaseModel):
    """Detailed tool-call accuracy metrics."""

    model: str
    quantization: str
    gpu_name: str
    tool_selection_accuracy: float | None = None
    parameter_accuracy: float | None = None
    parameter_hallucination_rate: float | None = None
    parse_success_rate: float | None = None
    no_call_accuracy: float | None = None
    overall_tool_accuracy: float | None = None


class QualitativeComparisonRow(BaseModel):
    quantization: str
    context_rot_score: float | None = None
    overall_tool_accuracy: float | None = None
    quality_composite_score: float | None = None
    mt_bench_score: float | None = None
    memory_accuracy: float | None = None


class QualitativeComparison(BaseModel):
    """Side-by-side qualitative comparison across quantizations for a single model+gpu."""

    model: str
    gpu_name: str
    rows: list[QualitativeComparisonRow]
    best_context_rot: str | None = None
    best_tool_accuracy: str | None = None
    best_quality_composite: str | None = None
    best_mt_bench: str | None = None
    insight: str = Field(description="1-2 sentence natural language summary of the comparison")


class QualitativeQueryResult(BaseModel):
    rows: list[QualitativeRow]
    total_qualitative_rows: int
    filtered_count: int


# ─── New tool models ──────────────────────────────────────────────────────────


class QuantitativeComparisonRow(BaseModel):
    quantization: str
    tokens_per_second: float | None = None
    avg_ttft_ms: float | None = None
    p50_itl_ms: float | None = None
    vram_gb: float | None = None
    concurrent_users: int | None = None
    runner_type: str | None = None
    n_rows: int = 1  # how many benchmark rows were averaged


class QuantitativeComparison(BaseModel):
    model: str
    gpu_name: str
    rows: list[QuantitativeComparisonRow]
    fastest_quant: str | None = None
    lowest_ttft_quant: str | None = None
    most_efficient_quant: str | None = None
    insight: str


class CombinedScores(BaseModel):
    """Quantitative + qualitative metrics for one (gpu, model, quant) configuration."""

    gpu_name: str
    model: str
    quantization: str
    # Quantitative
    tokens_per_second: float | None = None
    avg_ttft_ms: float | None = None
    p50_itl_ms: float | None = None
    vram_gb: float | None = None
    runner_type: str | None = None
    # Qualitative
    context_rot_score: float | None = None
    overall_tool_accuracy: float | None = None
    quality_composite_score: float | None = None
    mt_bench_score: float | None = None
    # Meta
    has_quantitative_data: bool = False
    has_qualitative_data: bool = False
    insight: str


class RankedConfig(BaseModel):
    quantization: str
    rank: int
    tokens_per_second: float | None = None
    context_rot_score: float | None = None
    mt_bench_score: float | None = None
    overall_tool_accuracy: float | None = None
    composite_score: float
    insight: str


class RankedQuantizations(BaseModel):
    model: str
    gpu_name: str
    priority: str
    rows: list[RankedConfig]
