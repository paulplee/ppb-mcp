"""recommend_hardware tool — budget-aware GPU recommendation."""

from __future__ import annotations

import math
from typing import Literal

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import HardwareOption, HardwareRecommendation
from ppb_mcp.tools._filters import is_blank

# ---------------------------------------------------------------------------
# Approximate GPU MSRPs (USD) at launch.  Kept here as a static reference so
# the tool can compute tokens-per-dollar.  Users can still get a recommendation
# without budget filtering; the prices are only used for the ranking signal.
# ---------------------------------------------------------------------------

_GPU_MSRP_USD: dict[str, float] = {
    # NVIDIA consumer GPUs
    "rtx 5090": 1_999.0,
    "rtx 5080": 999.0,
    "rtx 5070 ti": 749.0,
    "rtx 5070": 549.0,
    "rtx 4090": 1_599.0,
    "rtx 4080 super": 999.0,
    "rtx 4080": 1_199.0,
    "rtx 4070 ti super": 799.0,
    "rtx 4070 ti": 799.0,
    "rtx 4070 super": 599.0,
    "rtx 4070": 599.0,
    "rtx 3090 ti": 1_999.0,
    "rtx 3090": 1_499.0,
    "rtx 3080 ti": 1_199.0,
    "rtx 3080": 699.0,
    # NVIDIA professional
    "a100 80gb": 10_000.0,
    "a100 40gb": 6_500.0,
    "h100 80gb": 30_000.0,
    # Apple Silicon (system price proxy — base model only)
    "apple m4 max": 1_999.0,
    "apple m4 pro": 1_599.0,
    "apple m4": 599.0,
    "apple m3 max": 1_999.0,
    "apple m3 pro": 1_599.0,
    "apple m3": 599.0,
    "apple m2 ultra": 3_999.0,
    "apple m2 max": 1_999.0,
    "apple m2 pro": 1_299.0,
    "apple m1 ultra": 3_999.0,
    "apple m1 max": 1_999.0,
}


def _lookup_msrp(gpu_name: str) -> float | None:
    """Return the approximate MSRP for a GPU by fuzzy key lookup."""
    key = gpu_name.lower()
    for pattern, price in _GPU_MSRP_USD.items():
        if pattern in key:
            return price
    return None


def _confidence(n: int) -> Literal["high", "medium", "low"]:
    if n >= 5:
        return "high"
    if n >= 2:
        return "medium"
    return "low"


async def recommend_hardware(
    target_model: str,
    target_quantization: str | None = None,
    concurrent_users: int = 1,
    budget_usd: float | None = None,
    priority: Literal["speed", "efficiency", "value"] = "value",
) -> HardwareRecommendation:
    """Recommend the best GPU for running a specific model.

    USE THIS when a user asks what GPU to buy, which hardware can run a model,
    or wants to compare hardware options for a workload.

    This tool looks at real benchmark data to find which GPUs have actually been
    tested with the requested model and quantization, then ranks them by your
    priority (speed, efficiency, or value-for-money).

    Args:
        target_model: Model name to look up, e.g. "Qwen3.5-27B".  Partial match.
        target_quantization: Optional exact quantization label, e.g. "Q4_K_M".
            If omitted, all quants for the model are considered and the best
            performing quant per GPU is selected.
        concurrent_users: Target concurrency level (1–32).  Affects which runner
            rows are considered — results at this user count are preferred.
        budget_usd: Optional budget cap in USD.  GPUs with known MSRPs above this
            are excluded.  GPUs with unknown prices are always included.
        priority: Ranking strategy:
            - "speed"      → highest tokens/second first
            - "efficiency" → highest tokens/watt first (requires power data)
            - "value"      → highest tokens/dollar (MSRP-adjusted) first

    Returns:
        A ranked list of GPU options with throughput, power, and pricing estimates.

    Example:
        recommend_hardware(target_model="Qwen3.5-27B", concurrent_users=4, budget_usd=1500)
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    # --- Filter to the requested model / quant ----------------------------
    if "model_base" not in df.columns:
        return HardwareRecommendation(
            target_model=target_model,
            target_quantization=target_quantization,
            concurrent_users=concurrent_users,
            budget_usd=budget_usd,
            priority=priority,
            recommended=[],
            insight="No benchmark data available.",
        )

    sub = df[df["model_base"].astype(str).str.contains(target_model, case=False, na=False)]
    if not is_blank(target_quantization) and "quant" in sub.columns:
        sub = sub[sub["quant"] == target_quantization]

    if sub.empty:
        return HardwareRecommendation(
            target_model=target_model,
            target_quantization=target_quantization,
            concurrent_users=concurrent_users,
            budget_usd=budget_usd,
            priority=priority,
            recommended=[],
            insight=f"No benchmark data found for '{target_model}'"
            + (f" / {target_quantization}" if target_quantization else "")
            + ".",
        )

    # --- Prefer rows near the requested concurrent_users ------------------
    if "concurrent_users" in sub.columns:
        tested_cu = sorted(sub["concurrent_users"].dropna().unique().astype(int).tolist())
        if tested_cu:
            candidates = [v for v in tested_cu if v <= concurrent_users]
            best_cu = max(candidates) if candidates else min(tested_cu)
            cu_sub = sub[sub["concurrent_users"] == best_cu]
            if not cu_sub.empty:
                sub = cu_sub

    # --- Aggregate per GPU (mean throughput, power) -----------------------
    group_cols = ["gpu_name"]
    if "gpu_total_vram_gb" in sub.columns:
        group_cols.append("gpu_total_vram_gb")
    elif "gpu_vram_gb" in sub.columns:
        group_cols.append("gpu_vram_gb")

    agg: dict[str, str | list] = {"throughput_tok_s": ["mean", "count"]}
    if "avg_power_w" in sub.columns:
        agg["avg_power_w"] = "mean"
    if "quant" in sub.columns:
        agg["quant"] = "first"

    grouped = sub.groupby(group_cols).agg(agg)
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]
    grouped = grouped.reset_index()

    # Normalise column names
    col_map = {
        "throughput_tok_s_mean": "mean_tps",
        "throughput_tok_s_count": "sample_count",
        "avg_power_w_mean": "mean_power_w",
        "quant_first": "best_quant",
    }
    grouped = grouped.rename(columns={k: v for k, v in col_map.items() if k in grouped.columns})

    vram_col = "gpu_total_vram_gb" if "gpu_total_vram_gb" in grouped.columns else "gpu_vram_gb"

    # --- Attach MSRP and compute derived metrics -------------------------
    grouped["msrp"] = grouped["gpu_name"].apply(lambda n: _lookup_msrp(str(n)))

    # Budget filter (exclude only when MSRP is known AND exceeds budget)
    if budget_usd is not None:
        grouped = grouped[grouped["msrp"].isna() | (grouped["msrp"] <= budget_usd)]

    if grouped.empty:
        return HardwareRecommendation(
            target_model=target_model,
            target_quantization=target_quantization,
            concurrent_users=concurrent_users,
            budget_usd=budget_usd,
            priority=priority,
            recommended=[],
            insight=f"No hardware found within ${budget_usd:,.0f} budget for this workload.",
        )

    if "mean_tps" in grouped.columns and "msrp" in grouped.columns:
        grouped["tpd"] = grouped.apply(
            lambda r: (r["mean_tps"] / r["msrp"]) if r["msrp"] and r["msrp"] > 0 else None,
            axis=1,
        )
    if "mean_tps" in grouped.columns and "mean_power_w" in grouped.columns:
        grouped["tpw"] = grouped.apply(
            lambda r: (
                (r["mean_tps"] / r["mean_power_w"])
                if r.get("mean_power_w") and r["mean_power_w"] > 0
                else None
            ),
            axis=1,
        )

    # --- Sort by priority -------------------------------------------------
    if priority == "speed":
        sort_col = "mean_tps"
    elif priority == "efficiency" and "tpw" in grouped.columns:
        sort_col = "tpw"
    else:
        sort_col = "tpd" if "tpd" in grouped.columns else "mean_tps"

    grouped = grouped.sort_values(sort_col, ascending=False, na_position="last")

    # --- Build output --------------------------------------------------------
    options: list[HardwareOption] = []
    for _, row in grouped.head(10).iterrows():
        tps = row.get("mean_tps")
        if tps is None or (isinstance(tps, float) and math.isnan(tps)):
            continue

        vram_v = row.get(vram_col)
        vram_f = (
            float(vram_v)
            if vram_v is not None and not (isinstance(vram_v, float) and math.isnan(vram_v))
            else 0.0
        )
        msrp_v = row.get("msrp")
        msrp_f = (
            float(msrp_v)
            if msrp_v is not None and not (isinstance(msrp_v, float) and math.isnan(msrp_v))
            else None
        )
        tpd_v = row.get("tpd")
        tpd_f = (
            float(tpd_v)
            if tpd_v is not None and not (isinstance(tpd_v, float) and math.isnan(tpd_v))
            else None
        )
        pwr_v = row.get("mean_power_w")
        pwr_f = (
            float(pwr_v)
            if pwr_v is not None and not (isinstance(pwr_v, float) and math.isnan(pwr_v))
            else None
        )
        tpw_v = row.get("tpw")
        tpw_f = (
            float(tpw_v)
            if tpw_v is not None and not (isinstance(tpw_v, float) and math.isnan(tpw_v))
            else None
        )
        sc_v = row.get("sample_count", 1)
        sc = int(sc_v) if sc_v is not None else 1

        options.append(
            HardwareOption(
                gpu_name=str(row["gpu_name"]),
                vram_gb=vram_f,
                estimated_msrp_usd=msrp_f,
                tokens_per_second=round(float(tps), 2),
                tokens_per_dollar=round(tpd_f, 4) if tpd_f else None,
                avg_power_w=round(pwr_f, 1) if pwr_f else None,
                tokens_per_watt=round(tpw_f, 4) if tpw_f else None,
                supports_concurrent_users=concurrent_users,
                sample_count=sc,
                confidence=_confidence(sc),
            )
        )

    # Build insight text
    if options:
        top = options[0]
        if priority == "value" and top.tokens_per_dollar:
            insight = (
                f"Best value for '{target_model}' is the {top.gpu_name} at "
                f"~{top.tokens_per_second:.1f} tok/s ({top.tokens_per_dollar:.4f} tok/s per dollar)."
            )
        elif priority == "efficiency" and top.tokens_per_watt:
            insight = (
                f"Most power-efficient option is the {top.gpu_name}: "
                f"{top.tokens_per_watt:.3f} tok/s/W at {top.tokens_per_second:.1f} tok/s."
            )
        else:
            insight = (
                f"Fastest option for '{target_model}' is the {top.gpu_name} "
                f"at ~{top.tokens_per_second:.1f} tok/s."
            )
        if budget_usd:
            insight += f"  All options shown are within ${budget_usd:,.0f} budget."
    else:
        insight = "No viable hardware options found for this configuration."

    return HardwareRecommendation(
        target_model=target_model,
        target_quantization=target_quantization,
        concurrent_users=concurrent_users,
        budget_usd=budget_usd,
        priority=priority,
        recommended=options,
        insight=insight,
    )
