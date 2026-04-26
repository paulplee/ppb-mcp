"""query_qualitative_results tool."""

from __future__ import annotations

import pandas as pd

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import QualitativeQueryResult, QualitativeRow
from ppb_mcp.tools._qualitative import (
    filter_qualitative,
    opt_float,
    opt_int,
    opt_str,
    parse_json_dict,
)


def _row_to_model(r: pd.Series) -> QualitativeRow:
    return QualitativeRow(
        gpu_name=str(r.get("gpu_name") or ""),
        model=str(r.get("model_base") or ""),
        quantization=str(r.get("quant") or ""),
        runner_type=str(r.get("runner_type") or ""),
        suite_id=opt_str(r.get("suite_id")),
        benchmark_version=opt_str(r.get("benchmark_version")),
        context_rot_score=opt_float(r.get("context_rot_score")),
        context_rot_accuracy_by_length=parse_json_dict(r.get("context_rot_accuracy_by_length")),
        context_rot_accuracy_by_depth=parse_json_dict(r.get("context_rot_accuracy_by_depth")),
        context_rot_accuracy_by_needle=parse_json_dict(r.get("context_rot_accuracy_by_needle")),
        cases_skipped_context=opt_int(r.get("cases_skipped_context")),
        tool_selection_accuracy=opt_float(r.get("tool_selection_accuracy")),
        parameter_accuracy=opt_float(r.get("parameter_accuracy")),
        parameter_hallucination_rate=opt_float(r.get("parameter_hallucination_rate")),
        parse_success_rate=opt_float(r.get("parse_success_rate")),
        no_call_accuracy=opt_float(r.get("no_call_accuracy")),
        overall_tool_accuracy=opt_float(r.get("overall_tool_accuracy")),
        knowledge_accuracy_mean=opt_float(r.get("knowledge_accuracy_mean")),
        knowledge_accuracy_std=opt_float(r.get("knowledge_accuracy_std")),
        answer_relevancy_mean=opt_float(r.get("answer_relevancy_mean")),
        coherence_mean=opt_float(r.get("coherence_mean")),
        quality_composite_score=opt_float(r.get("quality_composite_score")),
        memory_accuracy=opt_float(r.get("memory_accuracy")),
        mt_bench_score=opt_float(r.get("mt_bench_score")),
        cases_evaluated=opt_int(r.get("cases_evaluated")),
    )


async def query_qualitative_results(
    model: str | None = None,
    quantization: str | None = None,
    gpu_name: str | None = None,
    runner_type: str | None = None,
    min_context_rot_score: float | None = None,
    min_overall_tool_accuracy: float | None = None,
    min_mt_bench_score: float | None = None,
    limit: int = 50,
) -> QualitativeQueryResult:
    """Query qualitative benchmark results with optional filters.

    Use runner_type to restrict to a specific phase:
      'context-rot', 'tool-accuracy', 'answer-quality', 'multiturn'

    Args:
        model: Partial match on model name.
        quantization: Exact quantization label.
        gpu_name: Partial match on GPU name.
        runner_type: Filter to one qualitative phase.
        min_context_rot_score: Only return rows where context_rot_score >= this value.
        min_overall_tool_accuracy: Only return rows where overall_tool_accuracy >= this.
        min_mt_bench_score: Only return rows where mt_bench_score >= this.
        limit: Max rows to return (1–200).
    """
    limit = max(1, min(int(limit), 200))
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    qualitative_total = (
        int((df["run_type"] == "qualitative").sum()) if "run_type" in df.columns else 0
    )

    sub = filter_qualitative(
        df,
        model=model,
        quantization=quantization,
        gpu_name=gpu_name,
        runner_type=runner_type,
    )

    if min_context_rot_score is not None and "context_rot_score" in sub.columns:
        sub = sub[sub["context_rot_score"].fillna(-1.0) >= min_context_rot_score]
    if min_overall_tool_accuracy is not None and "overall_tool_accuracy" in sub.columns:
        sub = sub[sub["overall_tool_accuracy"].fillna(-1.0) >= min_overall_tool_accuracy]
    if min_mt_bench_score is not None and "mt_bench_score" in sub.columns:
        sub = sub[sub["mt_bench_score"].fillna(-1.0) >= min_mt_bench_score]

    filtered_count = len(sub)
    page = sub.head(limit)
    rows = [_row_to_model(r) for _, r in page.iterrows()]
    return QualitativeQueryResult(
        rows=rows,
        total_qualitative_rows=qualitative_total,
        filtered_count=filtered_count,
    )
