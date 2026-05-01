"""get_combined_scores tool."""

from __future__ import annotations

import math

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import CombinedScores
from ppb_mcp.tools._filters import is_blank
from ppb_mcp.tools._qualitative import filter_qualitative


def _opt_float(v: object) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _build_insight(
    tps: float | None,
    vram: float | None,
    context_rot: float | None,
    tool_acc: float | None,
    has_quant: bool,
    has_qual: bool,
    mt_bench_score: float | None = None,
) -> str:
    if not has_quant and not has_qual:
        return "No benchmark data found for this configuration."
    if has_quant and has_qual:
        speed_str = f"~{tps:.0f} tok/s" if tps is not None else "unknown speed"
        parts: list[str] = [f"Runs at {speed_str}"]
        if context_rot is not None:
            quality_label = (
                "good" if context_rot >= 0.8 else ("moderate" if context_rot >= 0.5 else "limited")
            )
            parts.append(f"context recall score {context_rot:.2f} ({quality_label})")
        if tool_acc is not None:
            ta_label = "good" if tool_acc >= 0.8 else ("moderate" if tool_acc >= 0.5 else "poor")
            parts.append(f"tool accuracy {tool_acc:.2f} ({ta_label})")
        suffix = ""
        if mt_bench_score is None:
            suffix = " MT-Bench score not yet available for this configuration."
        return "; ".join(parts) + "." + suffix
    if has_quant:
        speed_str = f"~{tps:.0f} tok/s" if tps is not None else "speed measured"
        vram_str = f" using ~{vram:.1f} GB VRAM" if vram is not None else ""
        return f"Speed data available: {speed_str}{vram_str}. No qualitative benchmarks run for this config yet."
    # has_qual only
    if context_rot is not None:
        quality_label = (
            "good" if context_rot >= 0.8 else ("moderate" if context_rot >= 0.5 else "limited")
        )
        return f"Qualitative data available (context recall: {context_rot:.2f}, {quality_label}). No speed benchmarks for this config yet."
    return "Qualitative data available but no speed benchmarks for this config yet."


async def get_combined_scores(
    model: str,
    quantization: str,
    gpu_name: str | None = None,
) -> CombinedScores:
    """Get both quantitative (speed/VRAM) and qualitative (accuracy/quality) scores
    for a single (gpu, model, quant) configuration in one call.

    USE THIS TOOL when a user asks for an overall assessment of a model+quant combo,
    or wants to understand both the speed and quality trade-offs simultaneously.

    NOTE: Qualitative data is sparse — if has_qualitative_data=False, only speed metrics
    are available. Do NOT pass "null" as a string for gpu_name; omit it instead.

    Args:
        model: Partial match on model name (required).
        quantization: Exact quantization label (required).
        gpu_name: Optional partial match on GPU name.
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    gpu_display = gpu_name if not is_blank(gpu_name) else ""

    # ── Quantitative: best llama-server-loadtest row by TPS, else any row ──
    quant_sub = df
    if "run_type" in quant_sub.columns:
        quant_sub = quant_sub[quant_sub["run_type"] != "qualitative"]
    if not is_blank(model) and "model_base" in quant_sub.columns:
        quant_sub = quant_sub[
            quant_sub["model_base"].astype(str).str.contains(model, case=False, na=False)
        ]
    if not is_blank(quantization) and "quant" in quant_sub.columns:
        quant_sub = quant_sub[quant_sub["quant"] == quantization]
    if not is_blank(gpu_name) and "gpu_name" in quant_sub.columns:
        quant_sub = quant_sub[
            quant_sub["gpu_name"].astype(str).str.contains(gpu_name, case=False, na=False)
        ]

    tokens_per_second: float | None = None
    avg_ttft_ms: float | None = None
    p50_itl_ms: float | None = None
    vram_gb: float | None = None
    runner_type_used: str | None = None
    has_quantitative_data = False
    chosen_gpu = gpu_display
    chosen_model = model

    if not quant_sub.empty and "throughput_tok_s" in quant_sub.columns:
        # Prefer llama-server-loadtest
        loadtest = quant_sub
        if "runner_type" in quant_sub.columns:
            lt = quant_sub[
                quant_sub["runner_type"].astype(str).str.contains("loadtest", case=False, na=False)
            ]
            if not lt.empty:
                loadtest = lt
        best = loadtest.sort_values("throughput_tok_s", ascending=False).iloc[0]
        tokens_per_second = _opt_float(best.get("throughput_tok_s"))
        avg_ttft_ms = _opt_float(best.get("avg_ttft_ms"))
        p50_itl_ms = _opt_float(best.get("p50_itl_ms"))
        vram_col = "gpu_total_vram_gb" if "gpu_total_vram_gb" in best.index else "gpu_vram_gb"
        vram_gb = _opt_float(best.get(vram_col))
        runner_type_used = str(best.get("runner_type") or "") or None
        if not chosen_gpu and best.get("gpu_name"):
            chosen_gpu = str(best["gpu_name"])
        if best.get("model_base"):
            chosen_model = str(best["model_base"])
        has_quantitative_data = True

    # ── Qualitative ──
    qual_sub = filter_qualitative(df, model=model, quantization=quantization, gpu_name=gpu_name)

    context_rot_score: float | None = None
    overall_tool_accuracy: float | None = None
    quality_composite_score: float | None = None
    mt_bench_score: float | None = None
    has_qualitative_data = False

    if not qual_sub.empty:
        has_qualitative_data = True
        if not chosen_gpu and "gpu_name" in qual_sub.columns:
            first_gpu = qual_sub["gpu_name"].dropna()
            if not first_gpu.empty:
                chosen_gpu = str(first_gpu.iloc[0])
        for col, _target in [
            ("context_rot_score", "context_rot"),
            ("overall_tool_accuracy", "tool"),
            ("quality_composite_score", "quality"),
            ("mt_bench_score", "mt"),
        ]:
            if col in qual_sub.columns:
                val = qual_sub[col].dropna()
                if not val.empty:
                    if col == "context_rot_score":
                        context_rot_score = _opt_float(val.iloc[0])
                    elif col == "overall_tool_accuracy":
                        overall_tool_accuracy = _opt_float(val.iloc[0])
                    elif col == "quality_composite_score":
                        quality_composite_score = _opt_float(val.iloc[0])
                    elif col == "mt_bench_score":
                        mt_bench_score = _opt_float(val.iloc[0])

    insight = _build_insight(
        tps=tokens_per_second,
        vram=vram_gb,
        context_rot=context_rot_score,
        tool_acc=overall_tool_accuracy,
        has_quant=has_quantitative_data,
        has_qual=has_qualitative_data,
        mt_bench_score=mt_bench_score,
    )

    return CombinedScores(
        gpu_name=chosen_gpu,
        model=chosen_model,
        quantization=quantization,
        tokens_per_second=round(tokens_per_second, 2) if tokens_per_second is not None else None,
        avg_ttft_ms=round(avg_ttft_ms, 2) if avg_ttft_ms is not None else None,
        p50_itl_ms=round(p50_itl_ms, 2) if p50_itl_ms is not None else None,
        vram_gb=round(vram_gb, 2) if vram_gb is not None else None,
        runner_type=runner_type_used,
        context_rot_score=context_rot_score,
        overall_tool_accuracy=overall_tool_accuracy,
        quality_composite_score=quality_composite_score,
        mt_bench_score=mt_bench_score,
        has_quantitative_data=has_quantitative_data,
        has_qualitative_data=has_qualitative_data,
        insight=insight,
    )
