"""get_context_rot_breakdown tool."""

from __future__ import annotations

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import ContextRotBreakdown
from ppb_mcp.tools._qualitative import (
    filter_qualitative,
    opt_float,
    opt_int,
    parse_json_dict,
)


async def get_context_rot_breakdown(
    model: str,
    quantization: str,
    gpu_name: str | None = None,
) -> ContextRotBreakdown:
    """Get detailed context-rot (long-context recall) scores broken down by
    context length, insertion depth, and needle type.

    Context rot measures whether a model can recall a specific fact buried
    in a long context. Scores drop at longer lengths and extreme depths.
    A score of 0.0 at 131072 tokens means the model completely fails at
    128K context; 1.0 at 4096 means perfect recall at 4K.

    Args:
        model: Partial match on model name.
        quantization: Exact quantization label.
        gpu_name: Optional GPU filter.
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    sub = filter_qualitative(
        df,
        model=model,
        quantization=quantization,
        gpu_name=gpu_name,
        runner_type="context-rot",
    )

    if sub.empty:
        return ContextRotBreakdown(
            model=model,
            quantization=quantization,
            gpu_name=gpu_name or "",
        )

    row = sub.iloc[0]
    by_length_raw = parse_json_dict(row.get("context_rot_accuracy_by_length")) or {}
    by_depth_raw = parse_json_dict(row.get("context_rot_accuracy_by_depth")) or {}
    by_needle_raw = parse_json_dict(row.get("context_rot_accuracy_by_needle")) or {}

    by_length: dict[str, float | None] = {
        str(k): (None if v is None else opt_float(v)) for k, v in by_length_raw.items()
    }
    by_depth: dict[str, float] = {
        str(k): float(v) for k, v in by_depth_raw.items() if opt_float(v) is not None
    }
    by_needle: dict[str, float] = {
        str(k): float(v) for k, v in by_needle_raw.items() if opt_float(v) is not None
    }

    return ContextRotBreakdown(
        model=str(row.get("model_base") or model),
        quantization=str(row.get("quant") or quantization),
        gpu_name=str(row.get("gpu_name") or gpu_name or ""),
        overall_score=opt_float(row.get("context_rot_score")),
        by_length=by_length,
        by_depth=by_depth,
        by_needle=by_needle,
        cases_skipped=opt_int(row.get("cases_skipped_context")),
    )
