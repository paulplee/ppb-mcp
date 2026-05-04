"""explain_result tool — contextual explanation of a single benchmark result."""

from __future__ import annotations

import math
from typing import Literal

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import ResultExplanation
from ppb_mcp.tools._vram import estimate_total_vram_gb


def _nan_to_none(v: object) -> float | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return float(v)  # type: ignore[arg-type]


def _vram_pressure(
    usage_gb: float | None, total_gb: float | None
) -> Literal["low", "medium", "high", "unknown"]:
    if usage_gb is None or total_gb is None or total_gb <= 0:
        return "unknown"
    ratio = usage_gb / total_gb
    if ratio < 0.60:
        return "low"
    if ratio < 0.85:
        return "medium"
    return "high"


async def explain_result(
    gpu_name: str,
    model: str,
    quantization: str,
    concurrent_users: int = 1,
    n_ctx: int | None = None,
) -> ResultExplanation:
    """Explain why a benchmark result is what it is.

    USE THIS TOOL when a user wants to understand:
    - Why a particular GPU + model combination is fast or slow
    - How VRAM pressure affects performance
    - How this config compares to similar setups in the dataset
    - What the PCIe / architecture context means for throughput

    The tool looks up real benchmark rows for the requested config, computes
    how the result compares to all other tested configs for the same model+quant,
    and generates a plain-language insight.

    Args:
        gpu_name: GPU name (partial match, case-insensitive), e.g. "RTX 5090" or "M4 Max".
        model: Model base name (partial match), e.g. "Qwen3.5-27B".
        quantization: Exact quantization label, e.g. "Q4_K_M".
        concurrent_users: Concurrency level to look up (default 1).
        n_ctx: Optional context length filter.  If omitted, all context lengths
            for the config are averaged.

    Returns:
        A structured explanation with VRAM analysis, latency profile, hardware
        context, relative percentile ranking, and a plain-language insight.
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    # --- Find the specific result -----------------------------------------
    sub = df.copy()
    if "gpu_name" in sub.columns:
        sub = sub[sub["gpu_name"].astype(str).str.contains(gpu_name, case=False, na=False)]
    if "model_base" in sub.columns:
        sub = sub[sub["model_base"].astype(str).str.contains(model, case=False, na=False)]
    if "quant" in sub.columns:
        sub = sub[sub["quant"] == quantization]
    if "concurrent_users" in sub.columns:
        sub = sub[sub["concurrent_users"] == concurrent_users]
    if n_ctx is not None and "n_ctx" in sub.columns:
        sub = sub[sub["n_ctx"] == n_ctx]

    if sub.empty:
        return ResultExplanation(
            gpu_name=gpu_name,
            model=model,
            quantization=quantization,
            concurrent_users=concurrent_users,
            tokens_per_second=None,
            vram_total_gb=None,
            vram_estimated_usage_gb=None,
            vram_headroom_gb=None,
            vram_pressure="unknown",
            avg_ttft_ms=None,
            p99_ttft_ms=None,
            avg_itl_ms=None,
            pcie_gen=None,
            pcie_width=None,
            unified_memory=None,
            avg_power_w=None,
            percentile_rank_throughput=None,
            faster_than_pct=None,
            insight=f"No benchmark data found for {gpu_name} / {model} / {quantization} @ {concurrent_users} users.",
        )

    # Average across repeated runs for robustness
    row = sub.mean(numeric_only=True)
    first = sub.iloc[0]  # for string columns

    tps = _nan_to_none(row.get("throughput_tok_s"))

    # --- VRAM analysis ---------------------------------------------------
    vram_col = "gpu_total_vram_gb" if "gpu_total_vram_gb" in sub.columns else "gpu_vram_gb"
    vram_total = _nan_to_none(row.get(vram_col))
    gpu_name_str = str(first.get("gpu_name") or gpu_name)
    quant_str = str(first.get("quant") or quantization)
    model_base_str = str(first.get("model_base") or model)
    vram_est = estimate_total_vram_gb(model_base_str, quant_str, concurrent_users)
    vram_headroom = (vram_total - vram_est) if (vram_total and vram_est) else None
    pressure = _vram_pressure(vram_est, vram_total)

    # --- Latency profile -------------------------------------------------
    avg_ttft = _nan_to_none(row.get("avg_ttft_ms"))
    p99_ttft = _nan_to_none(row.get("p99_ttft_ms"))
    avg_itl = _nan_to_none(row.get("avg_itl_ms"))

    # --- Hardware context ------------------------------------------------
    pcie_gen_v = row.get("gpu_pcie_gen")
    pcie_gen = (
        int(pcie_gen_v)
        if pcie_gen_v is not None and not (isinstance(pcie_gen_v, float) and math.isnan(pcie_gen_v))
        else None
    )
    pcie_width_v = row.get("gpu_pcie_width")
    pcie_width = (
        int(pcie_width_v)
        if pcie_width_v is not None
        and not (isinstance(pcie_width_v, float) and math.isnan(pcie_width_v))
        else None
    )

    um_v = first.get("unified_memory")
    if um_v is None or (isinstance(um_v, float) and math.isnan(um_v)):
        unified_mem: bool | None = None
    elif isinstance(um_v, bool):
        unified_mem = um_v
    else:
        unified_mem = bool(um_v)

    avg_power = _nan_to_none(row.get("avg_power_w"))

    # --- Percentile ranking across all rows for same model+quant ---------
    baseline = df.copy()
    if "model_base" in baseline.columns:
        baseline = baseline[
            baseline["model_base"].astype(str).str.contains(model, case=False, na=False)
        ]
    if "quant" in baseline.columns:
        baseline = baseline[baseline["quant"] == quantization]
    if "concurrent_users" in baseline.columns:
        baseline = baseline[baseline["concurrent_users"] == concurrent_users]

    percentile_rank: float | None = None
    faster_than_pct: float | None = None
    if tps is not None and not baseline.empty and "throughput_tok_s" in baseline.columns:
        valid = baseline["throughput_tok_s"].dropna()
        if len(valid) > 0:
            faster_than = (valid < tps).sum()
            faster_than_pct = round(float(faster_than) / len(valid) * 100, 1)
            percentile_rank = faster_than_pct

    # --- Build insight text ----------------------------------------------
    parts: list[str] = []
    if tps is not None:
        parts.append(
            f"{gpu_name_str} runs {model_base_str} {quant_str} at {tps:.1f} tok/s with {concurrent_users} user(s)."
        )

    if pressure == "high":
        parts.append(
            f"VRAM is under pressure (~{vram_est:.1f} GB estimated in {vram_total:.1f} GB total), which likely throttles throughput."
        )
    elif pressure == "medium" and vram_headroom is not None:
        parts.append(f"VRAM usage is moderate — ~{vram_headroom:.1f} GB headroom remains.")
    elif pressure == "low" and vram_headroom is not None:
        parts.append(
            f"Ample VRAM headroom (~{vram_headroom:.1f} GB free) — this model should fit comfortably."
        )

    if unified_mem:
        parts.append(
            "This is a unified-memory system (Apple Silicon): CPU and GPU share the same memory pool."
        )
    elif pcie_gen and pcie_width:
        bw_note = "high-bandwidth" if (pcie_gen >= 4 and pcie_width >= 16) else "standard"
        parts.append(f"Connected via PCIe Gen{pcie_gen} ×{pcie_width} ({bw_note} link).")

    if faster_than_pct is not None:
        parts.append(
            f"This result is faster than {faster_than_pct:.0f}% of comparable tested configs."
        )

    if avg_power is not None:
        parts.append(f"Average GPU power draw: {avg_power:.0f} W.")

    insight = (
        "  ".join(parts)
        if parts
        else "Benchmark data found but could not generate a detailed explanation."
    )

    return ResultExplanation(
        gpu_name=gpu_name_str,
        model=model_base_str,
        quantization=quant_str,
        concurrent_users=concurrent_users,
        tokens_per_second=round(tps, 2) if tps else None,
        vram_total_gb=round(vram_total, 1) if vram_total else None,
        vram_estimated_usage_gb=round(vram_est, 2) if vram_est else None,
        vram_headroom_gb=round(vram_headroom, 2) if vram_headroom else None,
        vram_pressure=pressure,
        avg_ttft_ms=round(avg_ttft, 1) if avg_ttft else None,
        p99_ttft_ms=round(p99_ttft, 1) if p99_ttft else None,
        avg_itl_ms=round(avg_itl, 2) if avg_itl else None,
        pcie_gen=pcie_gen,
        pcie_width=pcie_width,
        unified_memory=unified_mem,
        avg_power_w=round(avg_power, 1) if avg_power else None,
        percentile_rank_throughput=percentile_rank,
        faster_than_pct=faster_than_pct,
        insight=insight,
    )
