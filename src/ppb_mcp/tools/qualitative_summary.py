"""get_qualitative_summary tool."""

from __future__ import annotations

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import QualitativeSummary
from ppb_mcp.tools._filters import is_blank
from ppb_mcp.tools._qualitative import filter_qualitative, first_non_null, opt_float, opt_str


async def get_qualitative_summary(
    model: str,
    quantization: str | None = None,
    gpu_name: str | None = None,
) -> list[QualitativeSummary]:
    """Get all available qualitative benchmark scores for a model, optionally filtered
    to a specific quantization and/or GPU.

    START HERE for qualitative questions about a specific model. Returns the best
    single-quant scorecard per GPU based on a composite quality score.  When
    gpu_name is omitted, one entry is returned for EACH GPU that has qualitative
    data for the model — so the result is always a list.  If you need a
    cross-quant comparison table for a single GPU, follow up with
    compare_quants_qualitative.

    USE THIS TOOL when a user asks about model quality, context recall ability,
    tool-call accuracy, or MT-Bench scores for a specific model.

    Returns scores for whichever of the four qualitative phases have been run:
    context rot (long-context recall), tool accuracy (structured output),
    answer quality (knowledge accuracy + coherence), and multi-turn (memory).
    Check phases_available — qualitative data is sparse.

    NOTE: Do NOT pass "null" for gpu_name or quantization — omit them entirely if
    unspecified. Qualitative data may be absent for many (model, quant, gpu) combos.

    Args:
        model: Partial match on model name, e.g. "Qwen3.5-0.8B".
        quantization: Optional exact quantization label, e.g. "Q4_K_M". When omitted,
                      the best-covered quantization per GPU is returned.
        gpu_name: Optional partial match on GPU name. If omitted, results for ALL
                  GPUs with matching qualitative data are returned.
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    sub = filter_qualitative(df, model=model, quantization=quantization, gpu_name=gpu_name)

    if sub.empty:
        return []

    # Determine which GPUs to produce a summary for.
    if not is_blank(gpu_name) or "gpu_name" not in sub.columns:
        gpu_list = [
            opt_str(sub["gpu_name"].iloc[0]) if "gpu_name" in sub.columns else gpu_name
        ]
    else:
        gpu_list = sub["gpu_name"].dropna().astype(str).unique().tolist()

    def _build_summary(gpu_sub, chosen_gpu: str) -> QualitativeSummary:
        chosen_model = (
            opt_str(gpu_sub["model_base"].iloc[0]) if "model_base" in gpu_sub.columns else model
        ) or model

        # When quantization is not specified, pick the quant with the best composite score
        # rather than the first quant in storage order (which may be BF16, etc.).
        working = gpu_sub
        if quantization is None and "quant" in gpu_sub.columns:

            def _quant_score(rows) -> float:
                def _v(col: str) -> float:
                    if col not in rows.columns:
                        return 0.0
                    val = rows[col].dropna()
                    return float(val.iloc[0]) if not val.empty else 0.0

                rot = _v("context_rot_score")
                ta = _v("overall_tool_accuracy")
                mt = _v("mt_bench_score")
                return rot * 0.3 + ta * 0.4 + (mt / 10.0) * 0.3

            quants = gpu_sub["quant"].dropna().unique().tolist()
            best_quant = None
            best_score = -1.0
            for q in quants:
                q_rows = gpu_sub[gpu_sub["quant"] == q]
                score = _quant_score(q_rows)
                if score > best_score:
                    best_score = score
                    best_quant = q
            if best_quant is not None and best_score > 0.0:
                working = gpu_sub[gpu_sub["quant"] == best_quant]

        phases: list[str] = []
        if "runner_type" in working.columns:
            phases = sorted({str(v) for v in working["runner_type"].dropna().tolist()})

        def pick(col: str) -> float | None:
            if col not in working.columns:
                return None
            return opt_float(first_non_null(working[col]))

        suite_id = opt_str(first_non_null(working["suite_id"])) if "suite_id" in working.columns else None
        bench_v = (
            opt_str(first_non_null(working["benchmark_version"]))
            if "benchmark_version" in working.columns
            else None
        )

        chosen_quant = quantization
        if chosen_quant is None and "quant" in working.columns:
            chosen_quant = opt_str(first_non_null(working["quant"]))

        return QualitativeSummary(
            gpu_name=chosen_gpu,
            model=chosen_model,
            quantization=chosen_quant,
            context_rot_score=pick("context_rot_score"),
            overall_tool_accuracy=pick("overall_tool_accuracy"),
            quality_composite_score=pick("quality_composite_score"),
            mt_bench_score=pick("mt_bench_score"),
            memory_accuracy=pick("memory_accuracy"),
            phases_available=phases,
            suite_id=suite_id,
            benchmark_version=bench_v,
        )

    results: list[QualitativeSummary] = []
    for g in gpu_list:
        gpu_sub = sub[sub["gpu_name"] == g] if "gpu_name" in sub.columns else sub
        if not gpu_sub.empty:
            results.append(_build_summary(gpu_sub, g))

    return results
