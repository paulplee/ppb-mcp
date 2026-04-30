"""query_ppb_results tool."""

from __future__ import annotations

import math

import pandas as pd

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import BenchmarkRow, QueryResult
from ppb_mcp.tools._filters import is_blank


def _row_to_model(r: pd.Series) -> BenchmarkRow:
    def _opt_float(key: str) -> float | None:
        v = r.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)

    def _opt_int(key: str) -> int | None:
        v = r.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return int(v)

    def _opt_str(key: str) -> str | None:
        v = r.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return str(v)

    vram = r.get("gpu_total_vram_gb")
    if vram is None or (isinstance(vram, float) and math.isnan(vram)):
        vram = r.get("gpu_vram_gb")
    vram_f = (
        float(vram)
        if vram is not None and not (isinstance(vram, float) and math.isnan(vram))
        else 0.0
    )

    cu_raw = r.get("concurrent_users")
    cu = (
        int(cu_raw)
        if cu_raw is not None and not (isinstance(cu_raw, float) and math.isnan(cu_raw))
        else 1
    )

    return BenchmarkRow(
        gpu_name=str(r.get("gpu_name") or ""),
        vram_gb=vram_f,
        model=str(r.get("model_base") or ""),
        model_org=_opt_str("model_org"),
        model_full_path=_opt_str("model"),
        quantization=str(r.get("quant") or ""),
        concurrent_users=cu,
        tokens_per_second=float(r.get("throughput_tok_s") or 0.0),
        avg_ttft_ms=_opt_float("avg_ttft_ms"),
        p50_itl_ms=_opt_float("p50_itl_ms"),
        n_ctx=_opt_int("n_ctx"),
        backend=_opt_str("backends"),
        runner_type=_opt_str("runner_type"),
        submitter=_opt_str("submitter"),
        timestamp=_opt_str("timestamp"),
    )


def _apply_filters(
    df: pd.DataFrame,
    *,
    gpu_name: str | None,
    vram_gb_min: float | None,
    vram_gb_max: float | None,
    model: str | None,
    quantization: str | None,
    backend: str | None,
    runner_type: str | None,
    concurrent_users: int | None,
) -> pd.DataFrame:
    out = df
    if not is_blank(gpu_name) and "gpu_name" in out.columns:
        out = out[out["gpu_name"].astype(str).str.contains(gpu_name, case=False, na=False)]  # type: ignore[arg-type]
    vram_col = "gpu_total_vram_gb" if "gpu_total_vram_gb" in out.columns else "gpu_vram_gb"
    if vram_col in out.columns:
        if vram_gb_min is not None:
            out = out[out[vram_col].fillna(-1) >= vram_gb_min]
        if vram_gb_max is not None:
            out = out[out[vram_col].fillna(float("inf")) <= vram_gb_max]
    if not is_blank(model) and "model_base" in out.columns:
        out = out[out["model_base"].astype(str).str.contains(model, case=False, na=False)]  # type: ignore[arg-type]
    if not is_blank(quantization) and "quant" in out.columns:
        out = out[out["quant"] == quantization]
    if not is_blank(backend) and "backends" in out.columns:
        out = out[out["backends"].astype(str).str.contains(backend, case=False, na=False)]  # type: ignore[arg-type]
    if not is_blank(runner_type) and "runner_type" in out.columns:
        out = out[out["runner_type"].astype(str).str.contains(runner_type, case=False, na=False)]  # type: ignore[arg-type]
    if concurrent_users is not None and "concurrent_users" in out.columns:
        out = out[out["concurrent_users"] == concurrent_users]
    return out


def _stratified_sample(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Return one representative row per (gpu_name, model_base, quant), capped at limit."""
    needed = [c for c in ("gpu_name", "model_base", "quant") if c in df.columns]
    if not needed:
        return df.head(limit)
    grouped = df.drop_duplicates(subset=needed, keep="first")
    return grouped.head(limit)


async def query_ppb_results(
    gpu_name: str | None = None,
    vram_gb_min: float | None = None,
    vram_gb_max: float | None = None,
    model: str | None = None,
    quantization: str | None = None,
    backend: str | None = None,
    runner_type: str | None = None,
    concurrent_users: int | None = None,
    limit: int = 50,
) -> QueryResult:
    """Filter raw benchmark rows from PPB.

    USE THIS TOOL when you need raw throughput numbers, TTFT, inter-token latency,
    or VRAM figures for a specific GPU, model, or quantization.

    IMPORTANT — always filter by runner_type when comparing speeds:
      - "llama-bench"            → raw throughput (tok/s), no concurrency
      - "llama-server-loadtest"  → real concurrent-user throughput (lower numbers, more realistic)

    NOTE: To omit a filter, EXCLUDE the parameter entirely. Do NOT pass the string
    "null" — it will not match any GPU and returns zero results.

    All filters are optional and AND-combined. String filters are case-insensitive
    partial matches except `quantization` (exact). When called with no filters,
    returns a stratified diverse sample (one row per gpu × model × quant combo).
    Never raises on empty results — returns rows=[].

    Args:
        gpu_name: Partial match on GPU name (case-insensitive).
        vram_gb_min: Minimum total VRAM in GB.
        vram_gb_max: Maximum total VRAM in GB.
        model: Partial match on model_base, e.g. "Qwen3.5-9B".
        quantization: Exact match on quantization label, e.g. "Q4_K_M".
        backend: Partial match on backend, e.g. "CUDA" or "Metal".
        runner_type: Filter by benchmark runner. Use "llama-bench" for raw throughput,
            "llama-server-loadtest" for real concurrent-user throughput (these numbers are
            NOT comparable — always filter by runner_type when comparing speeds).
        concurrent_users: Exact match on concurrent_users (1, 2, 4, 8, 16, or 32).
        limit: Max rows to return (1–500).

    Example calls:
        query_ppb_results(gpu_name="Apple M4 Pro", model="Qwen3.5-27B", runner_type="llama-server-loadtest")
        query_ppb_results(model="Qwen3.5-9B", quantization="Q4_K_M", concurrent_users=4)
    """
    limit = max(1, min(int(limit), 500))
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    total = len(df)
    no_filters = all(
        v is None
        for v in (
            gpu_name,
            vram_gb_min,
            vram_gb_max,
            model,
            quantization,
            backend,
            runner_type,
            concurrent_users,
        )
    )

    filtered = _apply_filters(
        df,
        gpu_name=gpu_name,
        vram_gb_min=vram_gb_min,
        vram_gb_max=vram_gb_max,
        model=model,
        quantization=quantization,
        backend=backend,
        runner_type=runner_type,
        concurrent_users=concurrent_users,
    )
    filtered_count = len(filtered)

    if no_filters:
        result = _stratified_sample(filtered, limit)
    else:
        result = filtered.head(limit)

    rows = [_row_to_model(r) for _, r in result.iterrows()]
    return QueryResult(rows=rows, total_count=total, filtered_count=filtered_count)
