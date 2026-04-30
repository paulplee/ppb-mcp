"""compare_quants_quantitative tool."""

from __future__ import annotations

import math

import pandas as pd

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import QuantitativeComparison, QuantitativeComparisonRow
from ppb_mcp.tools._filters import is_blank


def _opt_float(v: object) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_int(v: object) -> int | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _build_insight(
    rows: list[QuantitativeComparisonRow],
    fastest: str | None,
    lowest_ttft: str | None,
    most_efficient: str | None,
) -> str:
    if not rows:
        return "No quantitative data available for this model."
    if len(rows) == 1:
        r = rows[0]
        return (
            f"Only {r.quantization} has quantitative data; no cross-quantization "
            "comparison is possible yet."
        )

    parts: list[str] = []
    if fastest:
        parts.append(f"fastest: {fastest}")
    if lowest_ttft and lowest_ttft != fastest:
        parts.append(f"lowest TTFT: {lowest_ttft}")
    if most_efficient and most_efficient not in (fastest, lowest_ttft):
        parts.append(f"most VRAM-efficient: {most_efficient}")

    unique = {q for q in (fastest, lowest_ttft, most_efficient) if q}
    if len(unique) == 1:
        winner = next(iter(unique))
        return f"{winner} leads across all measured quantitative metrics."

    # Show top-2 TPS comparison if available
    by_tps = sorted(
        [(r.tokens_per_second, r.quantization) for r in rows if r.tokens_per_second is not None],
        reverse=True,
    )
    if len(by_tps) >= 2:
        top_tps, top_q = by_tps[0]
        sec_tps, sec_q = by_tps[1]
        speedup = top_tps / sec_tps if sec_tps > 0 else 1.0
        summary = (
            f"{top_q} is {speedup:.1f}× faster than {sec_q} "
            f"({top_tps:.0f} vs {sec_tps:.0f} tok/s)."
        )
        if parts:
            summary += f" Best by metric — {'; '.join(parts)}."
        return summary

    if parts:
        return f"Best by metric — {'; '.join(parts)}."
    return "Multiple quantizations tested; see rows for details."


async def compare_quants_quantitative(
    model: str,
    gpu_name: str | None = None,
    runner_type: str | None = None,
    concurrent_users: int | None = None,
) -> QuantitativeComparison:
    """Compare quantitative benchmark scores across quantizations for a model.

    Returns one row per quantization (averaged over matching benchmark runs),
    identifies the fastest, lowest-TTFT, and most VRAM-efficient quantization,
    and provides a plain-language insight.

    USE THIS TOOL when an agent or user asks "which quantization is fastest for X
    on Y GPU?" or "what's the speed trade-off between Q4 and Q8 for this model?"

    NOTE: Set runner_type="llama-server-loadtest" for concurrent-user comparisons,
    or "llama-bench" for raw single-user throughput. Mixing runner types produces
    misleading comparisons. Do NOT pass "null" for any string parameter — omit it.

    Args:
        model: Partial match on model name (required).
        gpu_name: Optional partial match on GPU name.
        runner_type: Optional filter by runner type (recommended to avoid mixing).
        concurrent_users: Optional exact match on concurrent_users.
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    if df.empty or "quant" not in df.columns or "throughput_tok_s" not in df.columns:
        return QuantitativeComparison(
            model=model,
            gpu_name=gpu_name if not is_blank(gpu_name) else "",
            rows=[],
            insight="No benchmark data available.",
        )

    # Filter to quantitative rows only (exclude qualitative run_type).
    sub = df
    if "run_type" in sub.columns:
        sub = sub[sub["run_type"] != "qualitative"]

    # Apply filters.
    if not is_blank(model) and "model_base" in sub.columns:
        sub = sub[sub["model_base"].astype(str).str.contains(model, case=False, na=False)]
    if not is_blank(gpu_name) and "gpu_name" in sub.columns:
        sub = sub[sub["gpu_name"].astype(str).str.contains(gpu_name, case=False, na=False)]
    if not is_blank(runner_type) and "runner_type" in sub.columns:
        sub = sub[sub["runner_type"].astype(str).str.contains(runner_type, case=False, na=False)]
    if concurrent_users is not None and "concurrent_users" in sub.columns:
        sub = sub[sub["concurrent_users"] == concurrent_users]

    sub = sub.dropna(subset=["quant", "throughput_tok_s"])

    if sub.empty:
        return QuantitativeComparison(
            model=model,
            gpu_name=gpu_name if not is_blank(gpu_name) else "",
            rows=[],
            insight="No quantitative data found for the specified filters.",
        )

    # Determine chosen GPU and model labels from data.
    chosen_gpu = gpu_name if not is_blank(gpu_name) else ""
    if not chosen_gpu and "gpu_name" in sub.columns:
        first_gpu = sub["gpu_name"].dropna()
        if not first_gpu.empty:
            chosen_gpu = str(first_gpu.iloc[0])

    chosen_model = model
    if "model_base" in sub.columns:
        first_m = sub["model_base"].dropna()
        if not first_m.empty:
            chosen_model = str(first_m.iloc[0])

    # Detect mixed runner types.
    runner_types_present: list[str] = []
    if "runner_type" in sub.columns:
        runner_types_present = sub["runner_type"].dropna().unique().tolist()

    vram_col = "gpu_total_vram_gb" if "gpu_total_vram_gb" in sub.columns else "gpu_vram_gb"

    rows: list[QuantitativeComparisonRow] = []
    for quant_label, group in sub.groupby("quant"):
        tps_vals = group["throughput_tok_s"].dropna()
        avg_tps = float(tps_vals.mean()) if not tps_vals.empty else None

        ttft_vals = group["avg_ttft_ms"].dropna() if "avg_ttft_ms" in group.columns else pd.Series([], dtype=float)
        avg_ttft = float(ttft_vals.mean()) if not ttft_vals.empty else None

        itl_vals = group["p50_itl_ms"].dropna() if "p50_itl_ms" in group.columns else pd.Series([], dtype=float)
        avg_itl = float(itl_vals.mean()) if not itl_vals.empty else None

        vram_vals = group[vram_col].dropna() if vram_col in group.columns else pd.Series([], dtype=float)
        max_vram = float(vram_vals.max()) if not vram_vals.empty else None

        cu_vals = group["concurrent_users"].dropna() if "concurrent_users" in group.columns else pd.Series([], dtype=float)
        cu = _opt_int(cu_vals.iloc[0]) if not cu_vals.empty else None

        # Determine runner_type label for this quant group.
        rt_label: str | None = None
        if "runner_type" in group.columns:
            rts = group["runner_type"].dropna().unique().tolist()
            if len(rts) == 1:
                rt_label = str(rts[0])
            elif len(rts) > 1:
                rt_label = "mixed"

        rows.append(
            QuantitativeComparisonRow(
                quantization=str(quant_label),
                tokens_per_second=round(avg_tps, 2) if avg_tps is not None else None,
                avg_ttft_ms=round(avg_ttft, 2) if avg_ttft is not None else None,
                p50_itl_ms=round(avg_itl, 2) if avg_itl is not None else None,
                vram_gb=round(max_vram, 2) if max_vram is not None else None,
                concurrent_users=cu,
                runner_type=rt_label,
                n_rows=len(group),
            )
        )

    rows.sort(key=lambda r: r.quantization)

    # Identify top performers.
    fastest_quant: str | None = None
    tps_scored = [(r.tokens_per_second, r.quantization) for r in rows if r.tokens_per_second is not None]
    if tps_scored:
        fastest_quant = max(tps_scored, key=lambda t: t[0])[1]

    lowest_ttft_quant: str | None = None
    ttft_scored = [(r.avg_ttft_ms, r.quantization) for r in rows if r.avg_ttft_ms is not None]
    if ttft_scored:
        lowest_ttft_quant = min(ttft_scored, key=lambda t: t[0])[1]

    most_efficient_quant: str | None = None
    efficiency_scored = [
        (r.tokens_per_second / r.vram_gb, r.quantization)
        for r in rows
        if r.tokens_per_second is not None and r.vram_gb is not None and r.vram_gb > 0
    ]
    if efficiency_scored:
        most_efficient_quant = max(efficiency_scored, key=lambda t: t[0])[1]

    insight = _build_insight(rows, fastest_quant, lowest_ttft_quant, most_efficient_quant)

    # Warn about mixed runner types in insight.
    if len(runner_types_present) > 1 and is_blank(runner_type):
        insight += (
            f" WARNING: results mix runner types ({', '.join(str(r) for r in runner_types_present)})"
            " — filter by runner_type for a fair comparison."
        )

    return QuantitativeComparison(
        model=chosen_model,
        gpu_name=chosen_gpu,
        rows=rows,
        fastest_quant=fastest_quant,
        lowest_ttft_quant=lowest_ttft_quant,
        most_efficient_quant=most_efficient_quant,
        insight=insight,
    )
