"""get_qualitative_summary tool."""

from __future__ import annotations

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import QualitativeSummary
from ppb_mcp.tools._filters import is_blank
from ppb_mcp.tools._qualitative import filter_qualitative, first_non_null, opt_float, opt_str


async def get_qualitative_summary(
    model: str,
    quantization: str,
    gpu_name: str | None = None,
) -> QualitativeSummary:
    """Get all available qualitative benchmark scores for a model+quant combination.

    USE THIS TOOL when a user asks about model quality, context recall ability, tool-call
    accuracy, or MT-Bench scores for a specific model and quantization.

    Returns scores for whichever of the four qualitative phases have been run:
    context rot (long-context recall), tool accuracy (structured output),
    answer quality (knowledge accuracy + coherence), and multi-turn (memory).
    Check has_qualitative_data via phases_available — qualitative data is sparse.

    NOTE: Do NOT pass "null" for gpu_name — omit it entirely if unspecified. Qualitative
    data may be absent for many (model, quant, gpu) combos; check phases_available.

    Args:
        model: Partial match on model name, e.g. "Qwen3.5-0.8B".
        quantization: Exact quantization label, e.g. "Q4_K_M".
        gpu_name: Optional partial match on GPU name. If omitted, uses
                  the first matching GPU in the dataset.
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    sub = filter_qualitative(df, model=model, quantization=quantization, gpu_name=gpu_name)

    if sub.empty:
        return QualitativeSummary(
            gpu_name=gpu_name or "",
            model=model,
            quantization=quantization,
            phases_available=[],
        )

    # If no gpu_name supplied, scope to the first matching GPU in the data.
    if is_blank(gpu_name) and "gpu_name" in sub.columns:
        first_gpu = sub["gpu_name"].dropna().astype(str).iloc[0]
        sub = sub[sub["gpu_name"] == first_gpu]
        chosen_gpu = first_gpu
    else:
        chosen_gpu = (
            opt_str(sub["gpu_name"].iloc[0]) if "gpu_name" in sub.columns else gpu_name
        ) or (gpu_name or "")

    chosen_model = (
        opt_str(sub["model_base"].iloc[0]) if "model_base" in sub.columns else model
    ) or model

    phases: list[str] = []
    if "runner_type" in sub.columns:
        phases = sorted({str(v) for v in sub["runner_type"].dropna().tolist()})

    def pick(col: str) -> float | None:
        if col not in sub.columns:
            return None
        return opt_float(first_non_null(sub[col]))

    suite_id = opt_str(first_non_null(sub["suite_id"])) if "suite_id" in sub.columns else None
    bench_v = (
        opt_str(first_non_null(sub["benchmark_version"]))
        if "benchmark_version" in sub.columns
        else None
    )

    return QualitativeSummary(
        gpu_name=chosen_gpu,
        model=chosen_model,
        quantization=quantization,
        context_rot_score=pick("context_rot_score"),
        overall_tool_accuracy=pick("overall_tool_accuracy"),
        quality_composite_score=pick("quality_composite_score"),
        mt_bench_score=pick("mt_bench_score"),
        memory_accuracy=pick("memory_accuracy"),
        phases_available=phases,
        suite_id=suite_id,
        benchmark_version=bench_v,
    )
