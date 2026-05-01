"""get_tool_accuracy_breakdown tool."""

from __future__ import annotations

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import ToolAccuracyBreakdown
from ppb_mcp.tools._qualitative import filter_qualitative, opt_float


async def get_tool_accuracy_breakdown(
    model: str,
    quantization: str,
    gpu_name: str | None = None,
) -> ToolAccuracyBreakdown:
    """Get detailed tool-call accuracy metrics for a model+quant.

    USE THIS TOOL when a user asks whether a model can reliably call tools, produce
    valid JSON, or select the correct function with correct parameters.

    Measures whether the model produces valid JSON tool calls with correct
    tool selection and parameter values. Key metrics:
    - tool_selection_accuracy: correct tool name chosen
    - parameter_accuracy: all required params correct and typed correctly
    - parameter_hallucination_rate: fraction of responses with invented params
    - overall_tool_accuracy: geometric mean of selection × parameter accuracy
      (collapses to 0 if either is 0)

    NOTE: Do NOT pass "null" for gpu_name — omit it entirely if unspecified.
    Qualitative data is sparse; many (model, quant, gpu) combos have no data yet.

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
        runner_type="tool-accuracy",
    )

    # Fall back to qualitative summary rows (runner_type="qualitative") which embed
    # tool accuracy columns for most models.
    if sub.empty:
        sub = filter_qualitative(
            df,
            model=model,
            quantization=quantization,
            gpu_name=gpu_name,
            runner_type="qualitative",
        )

    if sub.empty:
        return ToolAccuracyBreakdown(
            model=model,
            quantization=quantization,
            gpu_name=gpu_name or "",
        )

    row = sub.iloc[0]
    return ToolAccuracyBreakdown(
        model=str(row.get("model_base") or model),
        quantization=str(row.get("quant") or quantization),
        gpu_name=str(row.get("gpu_name") or gpu_name or ""),
        tool_selection_accuracy=opt_float(row.get("tool_selection_accuracy")),
        parameter_accuracy=opt_float(row.get("parameter_accuracy")),
        parameter_hallucination_rate=opt_float(row.get("parameter_hallucination_rate")),
        parse_success_rate=opt_float(row.get("parse_success_rate")),
        no_call_accuracy=opt_float(row.get("no_call_accuracy")),
        overall_tool_accuracy=opt_float(row.get("overall_tool_accuracy")),
    )
