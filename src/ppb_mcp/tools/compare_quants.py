"""compare_quants_qualitative tool."""

from __future__ import annotations

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import QualitativeComparison, QualitativeComparisonRow
from ppb_mcp.tools._filters import is_blank
from ppb_mcp.tools._qualitative import filter_qualitative, first_non_null, opt_float

_K_QUANT_PREFIXES = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K")


def _is_k_quant(label: str) -> bool:
    return any(label.startswith(p) for p in _K_QUANT_PREFIXES)


def _build_insight(
    rows: list[QualitativeComparisonRow],
    bests: dict[str, str | None],
) -> str:
    if not rows:
        return "No qualitative data available for this model."
    if len(rows) == 1:
        return (
            f"Only {rows[0].quantization} has qualitative data for this model; "
            "no cross-quantization comparison is possible yet."
        )

    # Filter out None bests (metric not measured anywhere)
    measured = {k: v for k, v in bests.items() if v is not None}
    if not measured:
        return "No qualitative metrics are populated across the tested quantizations."

    unique_winners = set(measured.values())
    if len(unique_winners) == 1:
        winner = next(iter(unique_winners))
        return f"{winner} dominates across all qualitative metrics."

    parts: list[str] = []
    for metric_label, key in [
        ("context rot", "best_context_rot"),
        ("tool accuracy", "best_tool_accuracy"),
        ("answer quality", "best_quality_composite"),
        ("MT-Bench", "best_mt_bench"),
    ]:
        if measured.get(key):
            parts.append(f"{metric_label}: {measured[key]}")
    split = "; ".join(parts)

    # Look for the K-quant vs non-K-quant inversion on tool accuracy.
    tool_winner = measured.get("best_tool_accuracy")
    rot_winner = measured.get("best_context_rot")
    if (
        tool_winner
        and rot_winner
        and tool_winner != rot_winner
        and not _is_k_quant(tool_winner)
        and _is_k_quant(rot_winner)
    ):
        return (
            f"Best quant differs by metric ({split}). "
            f"{tool_winner}'s simpler quantization scheme may preserve structured "
            "output token distributions better than K-quants on this model."
        )

    return f"Best quant differs by metric ({split})."


async def compare_quants_qualitative(
    model: str,
    gpu_name: str | None = None,
    quantizations: list[str] | None = None,
) -> QualitativeComparison:
    """Compare qualitative benchmark scores across quantizations for a model.

    USE THIS TOOL when a user asks "which quantization has better quality?" or
    wants a side-by-side comparison of context recall, tool accuracy, and MT-Bench
    scores across quantizations for the same model.

    Returns a table of qualitative scores for each tested quantization,
    identifies the best quant per metric, and provides a plain-language
    insight summarizing the trade-offs.

    NOTE: Do NOT pass "null" for gpu_name — omit it entirely if unspecified.
    Qualitative data is sparse; many (model, quant, gpu) combos have no data yet.

    Args:
        model: Partial match on model name.
        gpu_name: Optional GPU filter.
        quantizations: Optional list of exact quant labels to compare.
                       If None, all tested quantizations are included.
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    sub = filter_qualitative(df, model=model, gpu_name=gpu_name)

    chosen_gpu = gpu_name if not is_blank(gpu_name) else ""
    if sub.empty:
        return QualitativeComparison(
            model=model,
            gpu_name=chosen_gpu,
            rows=[],
            insight="No qualitative data available for this model.",
        )

    if not gpu_name and "gpu_name" in sub.columns:
        first_gpu = str(sub["gpu_name"].dropna().iloc[0])
        sub = sub[sub["gpu_name"] == first_gpu]
        chosen_gpu = first_gpu

    if quantizations and "quant" in sub.columns:
        sub = sub[sub["quant"].isin(quantizations)]

    if sub.empty or "quant" not in sub.columns:
        return QualitativeComparison(
            model=model,
            gpu_name=chosen_gpu,
            rows=[],
            insight="No qualitative data available for the requested quantizations.",
        )

    rows: list[QualitativeComparisonRow] = []
    for quant_label, group in sub.groupby("quant"):
        def pick(col: str, g=group) -> float | None:
            if col not in g.columns:
                return None
            return opt_float(first_non_null(g[col]))

        rows.append(
            QualitativeComparisonRow(
                quantization=str(quant_label),
                context_rot_score=pick("context_rot_score"),
                overall_tool_accuracy=pick("overall_tool_accuracy"),
                quality_composite_score=pick("quality_composite_score"),
                mt_bench_score=pick("mt_bench_score"),
                memory_accuracy=pick("memory_accuracy"),
            )
        )

    rows.sort(key=lambda r: r.quantization)

    def _argmax(attr: str) -> str | None:
        scored = [(getattr(r, attr), r.quantization) for r in rows if getattr(r, attr) is not None]
        if not scored:
            return None
        return max(scored, key=lambda t: t[0])[1]

    bests = {
        "best_context_rot": _argmax("context_rot_score"),
        "best_tool_accuracy": _argmax("overall_tool_accuracy"),
        "best_quality_composite": _argmax("quality_composite_score"),
        "best_mt_bench": _argmax("mt_bench_score"),
    }

    insight = _build_insight(rows, bests)

    chosen_model = model
    if "model_base" in sub.columns:
        first_m = sub["model_base"].dropna()
        if not first_m.empty:
            chosen_model = str(first_m.iloc[0])

    return QualitativeComparison(
        model=chosen_model,
        gpu_name=chosen_gpu,
        rows=rows,
        best_context_rot=bests["best_context_rot"],
        best_tool_accuracy=bests["best_tool_accuracy"],
        best_quality_composite=bests["best_quality_composite"],
        best_mt_bench=bests["best_mt_bench"],
        insight=insight,
    )
