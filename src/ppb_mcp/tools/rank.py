"""rank_by_priority tool."""

from __future__ import annotations

import math
from typing import Literal

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import RankedConfig, RankedQuantizations
from ppb_mcp.tools._filters import is_blank
from ppb_mcp.tools._qualitative import filter_qualitative


def _opt_float(v: object) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_series(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _row_insight(rank: int, quant: str, composite: float, tps: float | None) -> str:
    if rank == 1:
        speed_str = f" ({tps:.0f} tok/s)" if tps is not None else ""
        return f"Top-ranked{speed_str}; best composite score for this priority."
    if composite >= 0.7:
        return f"Strong performer (composite {composite:.2f})."
    if composite >= 0.4:
        return f"Moderate composite score ({composite:.2f}); trade-offs between speed and quality."
    return f"Lower composite score ({composite:.2f}); may have limited data or lower metrics."


async def rank_by_priority(
    model: str,
    gpu_name: str | None = None,
    priority: Literal["speed", "quality", "balance", "efficiency"] = "balance",
    limit: int = 20,
) -> RankedQuantizations:
    """Rank all tested quantizations for a model by a composite score.

    USE THIS TOOL when the user explicitly asks for a ranked list or wants to know
    which quantization is "best" by a specific metric (speed, quality, efficiency).
    For a single-config assessment, use get_combined_scores instead. For a
    VRAM-aware recommendation, use recommend_quantization instead.

    Unlike recommend_quantization (which requires VRAM and user count), this tool
    ranks all tested quants purely on measured performance and/or quality metrics.
    Useful when the GPU is known from the dataset and you want an overview ranking.

    Priority weights:
      - speed:      TPS=1.0
      - quality:    context_rot=0.4, tool_accuracy=0.2, mt_bench=0.2, TPS=0.2
      - balance:    TPS=0.5, context_rot=0.3, tool_accuracy=0.1, mt_bench=0.1
      - efficiency: TPS=0.7, tokens_per_watt=0.3 (falls back to balance if no power data)

    NOTE: Only quantizations with actual benchmark rows for this (model, gpu) combo
    are ranked — formula-only estimates are excluded. Do NOT pass "null" for gpu_name.

    Args:
        model: Partial match on model name (required).
        gpu_name: Optional partial match on GPU name.
        priority: Composite score weighting scheme (default: "balance").
        limit: Maximum number of ranked quantizations to return (default 20).
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    gpu_display = gpu_name if not is_blank(gpu_name) else ""
    chosen_model = model

    # ── Collect quantitative data ──
    quant_df = df
    if "run_type" in quant_df.columns:
        quant_df = quant_df[quant_df["run_type"] != "qualitative"]
    if not is_blank(model) and "model_base" in quant_df.columns:
        quant_df = quant_df[
            quant_df["model_base"].astype(str).str.contains(model, case=False, na=False)
        ]
    if not is_blank(gpu_name) and "gpu_name" in quant_df.columns:
        quant_df = quant_df[
            quant_df["gpu_name"].astype(str).str.contains(gpu_name, case=False, na=False)
        ]
    if not quant_df.empty and "model_base" in quant_df.columns:
        first_m = quant_df["model_base"].dropna()
        if not first_m.empty:
            chosen_model = str(first_m.iloc[0])
    if not quant_df.empty and not gpu_display and "gpu_name" in quant_df.columns:
        first_g = quant_df["gpu_name"].dropna()
        if not first_g.empty:
            gpu_display = str(first_g.iloc[0])

    # Best TPS per quant (and tokens_per_watt if available).
    quant_stats: dict[str, dict] = {}
    if (
        not quant_df.empty
        and "quant" in quant_df.columns
        and "throughput_tok_s" in quant_df.columns
    ):
        for quant_label, group in quant_df.groupby("quant"):
            tps_vals = group["throughput_tok_s"].dropna()
            if tps_vals.empty:
                continue
            best_idx = tps_vals.idxmax()
            best_row = group.loc[best_idx]
            tps = float(best_row["throughput_tok_s"])
            tpw: float | None = None
            if "avg_power_w" in best_row.index:
                power = _opt_float(best_row.get("avg_power_w"))
                if power and power > 0:
                    tpw = tps / power
            quant_stats[str(quant_label)] = {"tps": tps, "tpw": tpw}

    # ── Collect qualitative data ──
    qual_sub = filter_qualitative(df, model=model, gpu_name=gpu_name)
    qual_stats: dict[str, dict] = {}
    if not qual_sub.empty and "quant" in qual_sub.columns:
        for quant_label, group in qual_sub.groupby("quant"):

            def pick(col: str, g=group) -> float | None:
                if col not in g.columns:
                    return None
                vals = g[col].dropna()
                return float(vals.iloc[0]) if not vals.empty else None

            qual_stats[str(quant_label)] = {
                "context_rot": pick("context_rot_score"),
                "tool_accuracy": pick("overall_tool_accuracy"),
                "mt_bench": pick("mt_bench_score"),
            }

    # ── Union of all tested quants ──
    all_quants = sorted(set(quant_stats) | set(qual_stats))
    if not all_quants:
        return RankedQuantizations(
            model=chosen_model,
            gpu_name=gpu_display,
            priority=priority,
            rows=[],
        )

    # Build raw metric vectors.
    tps_raw = [quant_stats.get(q, {}).get("tps") for q in all_quants]
    tpw_raw = [quant_stats.get(q, {}).get("tpw") for q in all_quants]
    cr_raw = [qual_stats.get(q, {}).get("context_rot") for q in all_quants]
    ta_raw = [qual_stats.get(q, {}).get("tool_accuracy") for q in all_quants]
    mb_raw = [qual_stats.get(q, {}).get("mt_bench") for q in all_quants]

    def norm_vec(vals: list) -> list[float]:
        filled = [v if v is not None else 0.0 for v in vals]
        return _normalize_series(filled)

    tps_n = norm_vec(tps_raw)
    tpw_n = norm_vec(tpw_raw)
    cr_n = norm_vec(cr_raw)
    ta_n = norm_vec(ta_raw)
    mb_n = norm_vec(mb_raw)

    has_tpw = any(v is not None for v in tpw_raw)

    composites: list[float] = []
    for i in range(len(all_quants)):
        if priority == "speed":
            score = tps_n[i]
        elif priority == "quality":
            score = 0.4 * cr_n[i] + 0.2 * ta_n[i] + 0.2 * mb_n[i] + 0.2 * tps_n[i]
        elif priority == "efficiency":
            if has_tpw:
                score = 0.7 * tps_n[i] + 0.3 * tpw_n[i]
            else:
                score = 0.5 * tps_n[i] + 0.3 * cr_n[i] + 0.1 * ta_n[i] + 0.1 * mb_n[i]
        else:  # balance
            score = 0.5 * tps_n[i] + 0.3 * cr_n[i] + 0.1 * ta_n[i] + 0.1 * mb_n[i]
        composites.append(round(score, 4))

    # Sort by composite descending.
    ordering = sorted(range(len(all_quants)), key=lambda i: composites[i], reverse=True)

    rows: list[RankedConfig] = []
    for rank_pos, idx in enumerate(ordering[:limit], start=1):
        quant = all_quants[idx]
        tps = tps_raw[idx]
        cr = cr_raw[idx]
        mb = mb_raw[idx]
        ta = ta_raw[idx]
        comp = composites[idx]
        rows.append(
            RankedConfig(
                quantization=quant,
                rank=rank_pos,
                tokens_per_second=round(tps, 2) if tps is not None else None,
                context_rot_score=cr,
                mt_bench_score=mb,
                overall_tool_accuracy=ta,
                composite_score=comp,
                insight=_row_insight(rank_pos, quant, comp, tps),
            )
        )

    return RankedQuantizations(
        model=chosen_model,
        gpu_name=gpu_display,
        priority=priority,
        rows=rows,
    )
