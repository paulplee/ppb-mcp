"""Shared helpers for qualitative tools."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def opt_float(v: Any) -> float | None:
    if v is None or _is_nan(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def opt_int(v: Any) -> int | None:
    if v is None or _is_nan(v):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def opt_str(v: Any) -> str | None:
    if v is None or _is_nan(v):
        return None
    return str(v)


def parse_json_dict(v: Any) -> dict | None:
    """Parse a value that may be a JSON string, an already-decoded dict, or null."""
    if v is None or _is_nan(v):
        return None
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def filter_qualitative(
    df: pd.DataFrame,
    *,
    model: str | None = None,
    quantization: str | None = None,
    gpu_name: str | None = None,
    runner_type: str | None = None,
) -> pd.DataFrame:
    if "run_type" not in df.columns:
        return df.iloc[0:0]
    out = df[df["run_type"] == "qualitative"]
    if model and "model_base" in out.columns:
        out = out[out["model_base"].astype(str).str.contains(model, case=False, na=False)]
    if quantization and "quant" in out.columns:
        out = out[out["quant"] == quantization]
    if gpu_name and "gpu_name" in out.columns:
        out = out[out["gpu_name"].astype(str).str.contains(gpu_name, case=False, na=False)]
    if runner_type and "runner_type" in out.columns:
        out = out[out["runner_type"] == runner_type]
    return out


def first_non_null(series: pd.Series) -> Any:
    for v in series:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        return v
    return None
