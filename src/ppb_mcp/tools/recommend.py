"""recommend_quantization tool — three-tier empirical-first algorithm."""
from __future__ import annotations

from typing import Literal

import pandas as pd

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import QuantizationRecommendation
from ppb_mcp.tools._vram import bits_per_weight, estimate_vram_per_user_gb

HEADROOM_FRACTION = 0.90  # leave 10% headroom for OS / driver overhead


def _floor_concurrency(df: pd.DataFrame, requested: int) -> int:
    """Return the largest tested concurrent_users value <= requested.

    Falls back to the smallest tested value if none is <= requested.
    """
    if "concurrent_users" not in df.columns:
        return requested
    tested = sorted(df["concurrent_users"].dropna().unique().astype(int).tolist())
    if not tested:
        return requested
    candidates = [v for v in tested if v <= requested]
    if candidates:
        return max(candidates)
    return min(tested)


def _normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def _rank(df: pd.DataFrame, priority: str) -> pd.DataFrame:
    """Rank rows by priority. Adds a `_score` col and sorts descending."""
    if df.empty:
        return df
    bpw_series = df["quant"].map(bits_per_weight)
    if priority == "quality":
        return df.assign(_score=bpw_series).sort_values("_score", ascending=False)
    if priority == "speed":
        return df.assign(_score=df["throughput_tok_s"]).sort_values("_score", ascending=False)
    # balance
    bpw_n = _normalize(bpw_series)
    tps_n = _normalize(df["throughput_tok_s"])
    return df.assign(_score=0.5 * bpw_n + 0.5 * tps_n).sort_values("_score", ascending=False)


def _build_reasoning(
    *,
    tier: int,
    quant: str,
    gpu_label: str,
    user_vram: float,
    users: int,
    model_label: str,
    per_user: float,
    total: float,
    headroom: float,
    tps: float,
    n_rows: int,
    surrogate_gpu: str | None = None,
) -> str:
    if tier == 1:
        return (
            f"{quant} is recommended for your {gpu_label} ({user_vram:.0f} GB) running "
            f"{users} concurrent user{'s' if users != 1 else ''} of {model_label}. "
            f"Based on {n_rows} measured benchmark run{'s' if n_rows != 1 else ''}, "
            f"it uses ~{per_user:.1f} GB per user (~{total:.1f} GB total), "
            f"leaving ~{headroom:.1f} GB headroom and delivering ~{tps:.0f} tokens/sec per request."
        )
    if tier == 2:
        return (
            f"{quant} is the likely best choice for your {user_vram:.0f} GB GPU running "
            f"{users} concurrent user{'s' if users != 1 else ''} of {model_label}. "
            f"We don't have direct measurements at this VRAM tier, but the same model+quant tested on "
            f"{surrogate_gpu} delivered ~{tps:.0f} tokens/sec; "
            f"estimated headroom on your card is ~{headroom:.1f} GB."
        )
    # tier 3
    return (
        f"{quant} is the estimated best choice for a {user_vram:.0f} GB GPU running "
        f"{users} concurrent user{'s' if users != 1 else ''} of {model_label}. "
        f"No direct benchmark exists for this combination — based on model size and quantization, "
        f"it should require ~{per_user:.1f} GB per user (~{total:.1f} GB total), "
        f"leaving ~{headroom:.1f} GB headroom. Treat the throughput estimate (~{tps:.0f} tokens/sec) as approximate."
    )


def _empty_recommendation(
    gpu_vram_gb: float,
    concurrent_users: int,
    model_label: str,
    reason: str,
) -> QuantizationRecommendation:
    return QuantizationRecommendation(
        recommended_quantization="(none)",
        model=model_label,
        gpu_vram_gb=gpu_vram_gb,
        concurrent_users=concurrent_users,
        estimated_vram_usage_gb=0.0,
        estimated_vram_per_user_gb=0.0,
        estimated_tokens_per_second=0.0,
        headroom_gb=gpu_vram_gb,
        confidence="low",
        reasoning=reason,
        alternatives=[],
    )


async def recommend_quantization(
    gpu_vram_gb: float,
    concurrent_users: int,
    gpu_name: str | None = None,
    model: str | None = None,
    priority: Literal["quality", "speed", "balance"] = "balance",
) -> QuantizationRecommendation:
    """Recommend the best quantization for a given GPU VRAM budget and user count.

    Three-tier empirical-first algorithm:
      • Tier 1 (high confidence): direct benchmark rows on a GPU at-or-below the user's
        VRAM budget that were measured at the requested concurrent_users.
      • Tier 2 (medium): same (model_base, quant) tested on a different GPU at the
        requested user count — uses that throughput; scales headroom against user's VRAM.
      • Tier 3 (low): formula-only extrapolation (params × bits_per_weight / 8 × 1.15).

    Args:
        gpu_vram_gb: Total VRAM available on the GPU in GB.
        concurrent_users: Simultaneous inference requests to support (1–32).
        gpu_name: Optional GPU name (partial, case-insensitive). If supplied, prefers
            rows matching this exact GPU.
        model: Optional model_base partial match, e.g. "Qwen3.5-9B".
        priority: "quality" | "speed" | "balance" (default).
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    model_label = model or "any tested model"
    gpu_label = gpu_name or f"{gpu_vram_gb:.0f} GB GPU"

    if df.empty or "quant" not in df.columns or "throughput_tok_s" not in df.columns:
        return _empty_recommendation(
            gpu_vram_gb,
            concurrent_users,
            model_label,
            "No benchmark data available.",
        )

    # Apply model filter (partial, case-insensitive on model_base).
    base = df
    if model and "model_base" in base.columns:
        base = base[base["model_base"].astype(str).str.contains(model, case=False, na=False)]

    # Drop rows missing essentials.
    base = base.dropna(subset=["quant", "throughput_tok_s"])

    # ── Tier 1: empirical match on GPU at-or-below user VRAM, at requested users ──
    vram_col = "gpu_total_vram_gb" if "gpu_total_vram_gb" in base.columns else "gpu_vram_gb"
    tier1 = base.copy()
    if vram_col in tier1.columns:
        tier1 = tier1[tier1[vram_col].fillna(0) <= gpu_vram_gb]
    if gpu_name and "gpu_name" in tier1.columns:
        tier1 = tier1[tier1["gpu_name"].astype(str).str.contains(gpu_name, case=False, na=False)]
    effective_users_t1 = _floor_concurrency(tier1, concurrent_users)
    if "concurrent_users" in tier1.columns:
        tier1 = tier1[tier1["concurrent_users"] == effective_users_t1]

    if not tier1.empty:
        ranked = _rank(tier1, priority)
        top = ranked.iloc[0]
        # Pick best row per quant for alternatives.
        best_per_quant = ranked.drop_duplicates(subset=["quant"], keep="first")
        chosen_quant = str(top["quant"])
        n_matches = int((tier1["quant"] == chosen_quant).sum())
        confidence = "high" if n_matches >= 3 else "medium"

        chosen_model_for_vram = str(top.get("model_base") or model_label)
        per_user = estimate_vram_per_user_gb(chosen_model_for_vram, chosen_quant)
        if per_user is None:
            # Formula fallback: distribute the GPU's total VRAM across users.
            per_user = float(top.get(vram_col, gpu_vram_gb)) / max(concurrent_users, 1)
        total_used = per_user * concurrent_users
        headroom = gpu_vram_gb - total_used
        tps = float(top["throughput_tok_s"])
        chosen_model = chosen_model_for_vram

        reasoning = _build_reasoning(
            tier=1,
            quant=chosen_quant,
            gpu_label=str(top.get("gpu_name") or gpu_label),
            user_vram=gpu_vram_gb,
            users=concurrent_users,
            model_label=chosen_model,
            per_user=per_user,
            total=total_used,
            headroom=headroom,
            tps=tps,
            n_rows=n_matches,
        )
        if effective_users_t1 != concurrent_users:
            reasoning += (
                f" (Note: no data for {concurrent_users} concurrent users; "
                f"recommendation uses {effective_users_t1}-user measurements as the "
                "nearest tested benchmark.)"
            )
        alternatives = [str(q) for q in best_per_quant["quant"].tolist() if str(q) != chosen_quant][:2]
        return QuantizationRecommendation(
            recommended_quantization=chosen_quant,
            model=chosen_model,
            gpu_vram_gb=gpu_vram_gb,
            concurrent_users=concurrent_users,
            estimated_vram_usage_gb=round(total_used, 2),
            estimated_vram_per_user_gb=round(per_user, 2),
            estimated_tokens_per_second=round(tps, 1),
            headroom_gb=round(headroom, 2),
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
        )

    # ── Tier 2: empirical-near (any GPU, requested users, same model+quant universe) ──
    tier2 = base.copy()
    effective_users_t2 = _floor_concurrency(tier2, concurrent_users)
    if "concurrent_users" in tier2.columns:
        tier2 = tier2[tier2["concurrent_users"] == effective_users_t2]

    if not tier2.empty:
        # For each (model_base, quant) pair, take the best surrogate row by priority.
        ranked = _rank(tier2, priority)
        top = ranked.iloc[0]
        chosen_quant = str(top["quant"])
        chosen_model = str(top.get("model_base") or model_label)
        surrogate_gpu = str(top.get("gpu_name") or "another GPU")
        surrogate_vram = float(top.get(vram_col, gpu_vram_gb))
        tps = float(top["throughput_tok_s"])

        per_user_est = estimate_vram_per_user_gb(chosen_model, chosen_quant)
        if per_user_est is None:
            per_user_est = surrogate_vram / max(concurrent_users, 1)
        total_est = per_user_est * concurrent_users
        # Skip if the formula says it doesn't fit at all on the user's GPU.
        if total_est <= gpu_vram_gb * HEADROOM_FRACTION:
            headroom = gpu_vram_gb - total_est
            best_per_quant = ranked.drop_duplicates(subset=["quant"], keep="first")
            alternatives = [str(q) for q in best_per_quant["quant"].tolist() if str(q) != chosen_quant][:2]
            reasoning = _build_reasoning(
                tier=2,
                quant=chosen_quant,
                gpu_label=gpu_label,
                user_vram=gpu_vram_gb,
                users=concurrent_users,
                model_label=chosen_model,
                per_user=per_user_est,
                total=total_est,
                headroom=headroom,
                tps=tps,
                n_rows=0,
                surrogate_gpu=f"{surrogate_gpu} ({surrogate_vram:.0f} GB)",
            )
            if effective_users_t2 != concurrent_users:
                reasoning += (
                    f" (Note: no data for {concurrent_users} concurrent users; "
                    f"recommendation uses {effective_users_t2}-user measurements as the "
                    "nearest tested benchmark.)"
                )
            return QuantizationRecommendation(
                recommended_quantization=chosen_quant,
                model=chosen_model,
                gpu_vram_gb=gpu_vram_gb,
                concurrent_users=concurrent_users,
                estimated_vram_usage_gb=round(total_est, 2),
                estimated_vram_per_user_gb=round(per_user_est, 2),
                estimated_tokens_per_second=round(tps, 1),
                headroom_gb=round(headroom, 2),
                confidence="medium",
                reasoning=reasoning,
                alternatives=alternatives,
            )

    # ── Tier 3: formula extrapolation across all known quants ──
    if not model:
        return _empty_recommendation(
            gpu_vram_gb,
            concurrent_users,
            model_label,
            f"No benchmark data fits a {gpu_vram_gb:.0f} GB GPU at {concurrent_users} concurrent users, "
            "and no specific model was supplied to extrapolate from.",
        )
    candidates: list[tuple[str, float, float]] = []  # (quant, per_user, bpw)
    for quant in sorted(set(base["quant"].dropna().astype(str).tolist()) or {"Q4_K_M"}):
        per_user = estimate_vram_per_user_gb(model, quant)
        if per_user is None:
            continue
        total = per_user * concurrent_users
        if total <= gpu_vram_gb * HEADROOM_FRACTION:
            candidates.append((quant, per_user, bits_per_weight(quant)))

    if not candidates:
        return _empty_recommendation(
            gpu_vram_gb,
            concurrent_users,
            model_label,
            f"No quantization of {model_label} is estimated to fit on a {gpu_vram_gb:.0f} GB GPU "
            f"at {concurrent_users} concurrent users.",
        )

    # Rank by priority on formula-derived values.
    if priority == "quality":
        candidates.sort(key=lambda c: c[2], reverse=True)
    elif priority == "speed":
        # Smaller quants are usually faster; rank by lower bpw first.
        candidates.sort(key=lambda c: c[2])
    else:  # balance — prefer middle bpw closest to 4.5
        candidates.sort(key=lambda c: abs(c[2] - 4.5))

    chosen_quant, per_user, _ = candidates[0]
    total = per_user * concurrent_users
    headroom = gpu_vram_gb - total
    # Throughput is unknown — estimate as 100 tok/s as a placeholder; downgrade lang accordingly.
    tps_estimate = 100.0
    alternatives = [c[0] for c in candidates[1:3]]
    reasoning = _build_reasoning(
        tier=3,
        quant=chosen_quant,
        gpu_label=gpu_label,
        user_vram=gpu_vram_gb,
        users=concurrent_users,
        model_label=model_label,
        per_user=per_user,
        total=total,
        headroom=headroom,
        tps=tps_estimate,
        n_rows=0,
    )
    return QuantizationRecommendation(
        recommended_quantization=chosen_quant,
        model=model_label,
        gpu_vram_gb=gpu_vram_gb,
        concurrent_users=concurrent_users,
        estimated_vram_usage_gb=round(total, 2),
        estimated_vram_per_user_gb=round(per_user, 2),
        estimated_tokens_per_second=tps_estimate,
        headroom_gb=round(headroom, 2),
        confidence="low",
        reasoning=reasoning,
        alternatives=alternatives,
    )
