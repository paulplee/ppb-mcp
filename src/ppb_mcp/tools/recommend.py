"""recommend_quantization tool — three-tier empirical-first algorithm."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import QuantizationRecommendation
from ppb_mcp.tools._filters import is_blank
from ppb_mcp.tools._vram import bits_per_weight, estimate_total_vram_gb, estimate_vram_per_user_gb

HEADROOM_FRACTION = 0.90  # leave 10% headroom for OS / driver overhead


def _filter_viable(ranked: pd.DataFrame, gpu_vram_gb: float, users: int) -> pd.DataFrame:
    """Drop candidates whose two-term VRAM estimate exceeds the GPU budget."""

    def fits(row: pd.Series) -> bool:
        model = str(row.get("model_base") or "")
        quant = str(row.get("quant") or "")
        total = estimate_total_vram_gb(model, quant, users)
        if total is None:
            return True  # can't rule out; keep and note uncertainty
        return total <= (gpu_vram_gb * HEADROOM_FRACTION)

    mask = ranked.apply(fits, axis=1)
    return ranked[mask]


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
    if priority == "efficiency":
        # tokens_per_watt = throughput / avg_power_w. Rows with no power data score 0.
        # If NO rows have power data, fall back to balance ranking silently.
        if "avg_power_w" in df.columns:
            power = df["avg_power_w"].fillna(0).clip(lower=0)
            has_power = power > 0
            tpw = pd.Series(0.0, index=df.index)
            tpw[has_power] = df.loc[has_power, "throughput_tok_s"] / power[has_power]
            if tpw.sum() > 0:
                return df.assign(_score=tpw).sort_values("_score", ascending=False)
        # Fallback: balance.
        bpw_n = _normalize(bpw_series)
        tps_n = _normalize(df["throughput_tok_s"])
        return df.assign(_score=0.5 * bpw_n + 0.5 * tps_n).sort_values("_score", ascending=False)
    # balance (default)
    bpw_n = _normalize(bpw_series)
    tps_n = _normalize(df["throughput_tok_s"])
    return df.assign(_score=0.5 * bpw_n + 0.5 * tps_n).sort_values("_score", ascending=False)


def _compute_tokens_per_watt(row: pd.Series, tps: float) -> float | None:
    """Return tokens/sec per watt from a benchmark row, or None if no power data."""
    if "avg_power_w" not in row.index:
        return None
    power = row.get("avg_power_w")
    if power is None or pd.isna(power) or float(power) <= 0:
        return None
    return round(tps / float(power), 3)


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
    tokens_per_watt: float | None = None,
) -> str:
    if tier == 1:
        base = (
            f"{quant} is recommended for your {gpu_label} ({user_vram:.0f} GB) running "
            f"{users} concurrent user{'s' if users != 1 else ''} of {model_label}. "
            f"Based on {n_rows} measured benchmark run{'s' if n_rows != 1 else ''}, "
            f"it uses ~{per_user:.1f} GB per user (~{total:.1f} GB total), "
            f"leaving ~{headroom:.1f} GB headroom and delivering ~{tps:.0f} tokens/sec per request."
        )
        if tokens_per_watt is not None:
            base += f" Power efficiency: ~{tokens_per_watt:.2f} tokens/sec/watt."
        return base
    if tier == 2:
        base = (
            f"{quant} is the likely best choice for your {user_vram:.0f} GB GPU running "
            f"{users} concurrent user{'s' if users != 1 else ''} of {model_label}. "
            f"We don't have direct measurements at this VRAM tier, but the same model+quant tested on "
            f"{surrogate_gpu} delivered ~{tps:.0f} tokens/sec; "
            f"estimated headroom on your card is ~{headroom:.1f} GB."
        )
        if tokens_per_watt is not None:
            base += f" Power efficiency: ~{tokens_per_watt:.2f} tokens/sec/watt."
        return base
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
    concurrent_users: int,
    gpu_vram_gb: float | None = None,
    gpu_name: str | None = None,
    model: str | None = None,
    priority: Literal["quality", "speed", "balance", "efficiency"] = "balance",
) -> QuantizationRecommendation:
    """Recommend the best quantization for a given GPU VRAM budget and user count.

    USE THIS TOOL when a user asks "what quantization fits in X GB?" or "what's the
    best quant for Y concurrent users on Z GPU?".

    Three-tier empirical-first algorithm:
      • Tier 1 (high confidence): direct benchmark rows on a GPU at-or-below the user's
        VRAM budget that were measured at the requested concurrent_users.
      • Tier 2 (medium): same (model_base, quant) tested on a different GPU at the
        requested user count — uses that throughput; scales headroom against user's VRAM.
      • Tier 3 (low): formula-only extrapolation (params × bits_per_weight / 8 × 1.1).

    IMPORTANT — headroom_gb will always be >= 0. Any candidate that doesn't physically
    fit (after applying the two-term VRAM formula) is excluded before ranking.

    NOTE: To omit a filter, EXCLUDE the parameter entirely. Do NOT pass "null".

    Args:
        concurrent_users: Simultaneous inference requests to support (1–32).
        gpu_vram_gb: Total VRAM available on the GPU in GB. Optional when gpu_name is
            supplied — the VRAM will be looked up from benchmark rows automatically.
        gpu_name: Optional GPU name (partial, case-insensitive). If supplied, prefers
            rows matching this exact GPU and auto-resolves VRAM when gpu_vram_gb is omitted.
        model: Optional model_base partial match, e.g. "Qwen3.5-9B".
        priority: "quality" | "speed" | "balance" | "efficiency" (default: "balance").
            "efficiency" ranks by tokens-per-watt — best for always-on or
            power-constrained setups. Falls back to "balance" when no power
            data exists for the candidate rows.
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    # Resolve gpu_vram_gb from benchmark data when only gpu_name is supplied.
    if gpu_vram_gb is None:
        if is_blank(gpu_name):
            return _empty_recommendation(
                0.0,
                concurrent_users,
                model or "any tested model",
                "Either gpu_vram_gb or gpu_name must be provided.",
            )
        vram_col_lookup = (
            "gpu_total_vram_gb" if "gpu_total_vram_gb" in df.columns else "gpu_vram_gb"
        )
        gpu_rows = df
        if "gpu_name" in df.columns:
            gpu_rows = df[df["gpu_name"].astype(str).str.contains(gpu_name, case=False, na=False)]
        if gpu_rows.empty or vram_col_lookup not in gpu_rows.columns:
            return _empty_recommendation(
                0.0,
                concurrent_users,
                model or "any tested model",
                f"GPU {gpu_name!r} not found in the dataset; cannot determine VRAM capacity.",
            )
        resolved = float(gpu_rows[vram_col_lookup].dropna().max() or 0.0)
        if resolved <= 0.0:
            return _empty_recommendation(
                0.0,
                concurrent_users,
                model or "any tested model",
                f"GPU {gpu_name!r} found but VRAM capacity is unknown.",
            )
        gpu_vram_gb = resolved

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
    if not is_blank(model) and "model_base" in base.columns:
        base = base[base["model_base"].astype(str).str.contains(model, case=False, na=False)]

    # Drop rows missing essentials.
    base = base.dropna(subset=["quant", "throughput_tok_s"])

    # ── Tier 1: empirical match on GPU at-or-below user VRAM, at requested users ──
    vram_col = "gpu_total_vram_gb" if "gpu_total_vram_gb" in base.columns else "gpu_vram_gb"
    tier1 = base.copy()
    if vram_col in tier1.columns:
        tier1 = tier1[tier1[vram_col].fillna(0) <= gpu_vram_gb]
    if not is_blank(gpu_name) and "gpu_name" in tier1.columns:
        tier1 = tier1[tier1["gpu_name"].astype(str).str.contains(gpu_name, case=False, na=False)]
    effective_users_t1 = _floor_concurrency(tier1, concurrent_users)
    if "concurrent_users" in tier1.columns:
        tier1 = tier1[tier1["concurrent_users"] == effective_users_t1]

    if not tier1.empty:
        ranked = _rank(tier1, priority)
        ranked = _filter_viable(ranked, gpu_vram_gb, concurrent_users)
        if not ranked.empty:
            top = ranked.iloc[0]
            # Pick best row per quant for alternatives.
            best_per_quant = ranked.drop_duplicates(subset=["quant"], keep="first")
            chosen_quant = str(top["quant"])
            n_matches = int((tier1["quant"] == chosen_quant).sum())
            confidence = "high" if n_matches >= 3 else "medium"

            chosen_model_for_vram = str(top.get("model_base") or model_label)
            # Two-term formula: fixed weights + per-user KV cache (does not replicate weights)
            total_used = estimate_total_vram_gb(
                chosen_model_for_vram, chosen_quant, concurrent_users
            )
            if total_used is None:
                # Formula fallback: distribute the GPU's total VRAM across users.
                per_user = float(top.get(vram_col, gpu_vram_gb)) / max(concurrent_users, 1)
                total_used = per_user * concurrent_users
            else:
                per_user = estimate_vram_per_user_gb(chosen_model_for_vram, chosen_quant) or (
                    total_used / max(concurrent_users, 1)
                )
            headroom = gpu_vram_gb - total_used
            tps = float(top["throughput_tok_s"])
            tokens_per_watt = _compute_tokens_per_watt(top, tps)
            chosen_model = chosen_model_for_vram

            # Fix GPU label: distinguish benchmark GPU from user's GPU
            benchmark_gpu = str(top.get("gpu_name") or gpu_label)
            if benchmark_gpu.lower() != gpu_label.lower():
                gpu_display = f"{gpu_label} (benchmark from {benchmark_gpu})"
            else:
                gpu_display = gpu_label

            reasoning = _build_reasoning(
                tier=1,
                quant=chosen_quant,
                gpu_label=gpu_display,
                user_vram=gpu_vram_gb,
                users=concurrent_users,
                model_label=chosen_model,
                per_user=per_user,
                total=total_used,
                headroom=headroom,
                tps=tps,
                n_rows=n_matches,
                tokens_per_watt=tokens_per_watt,
            )
            if effective_users_t1 != concurrent_users:
                reasoning += (
                    f" (Note: no data for {concurrent_users} concurrent users; "
                    f"recommendation uses {effective_users_t1}-user measurements as the "
                    "nearest tested benchmark.)"
                )
            # Flag when the matched model differs from what the user asked for.
            user_model_clean = (model or "").strip().lower()
            if user_model_clean and user_model_clean not in chosen_model.strip().lower():
                reasoning += (
                    f" Note: the requested model '{model}' was matched to '{chosen_model}'"
                    " in the dataset — verify this is the intended model before using"
                    " this recommendation."
                )
            alternatives = [
                str(q) for q in best_per_quant["quant"].tolist() if str(q) != chosen_quant
            ][:2]
            return QuantizationRecommendation(
                recommended_quantization=chosen_quant,
                model=chosen_model,
                gpu_vram_gb=gpu_vram_gb,
                concurrent_users=concurrent_users,
                estimated_vram_usage_gb=round(total_used, 2),
                estimated_vram_per_user_gb=round(per_user, 2),
                estimated_tokens_per_second=round(tps, 1),
                estimated_tokens_per_watt=tokens_per_watt,
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
        ranked = _filter_viable(ranked, gpu_vram_gb, concurrent_users)
        if not ranked.empty:
            top = ranked.iloc[0]
            chosen_quant = str(top["quant"])
            chosen_model = str(top.get("model_base") or model_label)
            surrogate_gpu = str(top.get("gpu_name") or "another GPU")
            surrogate_vram = float(top.get(vram_col, gpu_vram_gb))
            tps = float(top["throughput_tok_s"])
            tokens_per_watt = _compute_tokens_per_watt(top, tps)

            # Two-term formula for total VRAM; fall back to old estimate if needed
            total_est = estimate_total_vram_gb(chosen_model, chosen_quant, concurrent_users)
            if total_est is None:
                per_user_est = surrogate_vram / max(concurrent_users, 1)
                total_est = per_user_est * concurrent_users
            else:
                per_user_est = estimate_vram_per_user_gb(chosen_model, chosen_quant) or (
                    total_est / max(concurrent_users, 1)
                )
            headroom = gpu_vram_gb - total_est
            best_per_quant = ranked.drop_duplicates(subset=["quant"], keep="first")
            alternatives = [
                str(q) for q in best_per_quant["quant"].tolist() if str(q) != chosen_quant
            ][:2]
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
                tokens_per_watt=tokens_per_watt,
            )
            if effective_users_t2 != concurrent_users:
                reasoning += (
                    f" (Note: no data for {concurrent_users} concurrent users; "
                    f"recommendation uses {effective_users_t2}-user measurements as the "
                    "nearest tested benchmark.)"
                )
            # Flag when the matched model differs from what the user asked for.
            user_model_clean = (model or "").strip().lower()
            if user_model_clean and user_model_clean not in chosen_model.strip().lower():
                reasoning += (
                    f" Note: the requested model '{model}' was matched to '{chosen_model}'"
                    " in the dataset — verify this is the intended model before using"
                    " this recommendation."
                )
            return QuantizationRecommendation(
                recommended_quantization=chosen_quant,
                model=chosen_model,
                gpu_vram_gb=gpu_vram_gb,
                concurrent_users=concurrent_users,
                estimated_vram_usage_gb=round(total_est, 2),
                estimated_vram_per_user_gb=round(per_user_est, 2),
                estimated_tokens_per_second=round(tps, 1),
                estimated_tokens_per_watt=tokens_per_watt,
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
    candidates: list[tuple[str, float, float, float]] = []  # (quant, per_user, total, bpw)
    for quant in sorted(set(base["quant"].dropna().astype(str).tolist()) or {"Q4_K_M"}):
        total = estimate_total_vram_gb(model, quant, concurrent_users)
        if total is None:
            # Fallback to old formula if params unparseable
            per_user = estimate_vram_per_user_gb(model, quant)
            if per_user is None:
                continue
            total = per_user * concurrent_users
        else:
            per_user = estimate_vram_per_user_gb(model, quant) or (total / max(concurrent_users, 1))
        if total <= gpu_vram_gb * HEADROOM_FRACTION:
            candidates.append((quant, per_user, total, bits_per_weight(quant)))

    if not candidates:
        # Second pass without headroom guard — find quants that physically fit but are risky.
        risky_candidates: list[tuple[str, float, float, float]] = []
        for quant in sorted(set(base["quant"].dropna().astype(str).tolist()) or {"Q4_K_M"}):
            total = estimate_total_vram_gb(model, quant, concurrent_users)
            if total is None:
                per_user = estimate_vram_per_user_gb(model, quant)
                if per_user is None:
                    continue
                total = per_user * concurrent_users
            else:
                per_user = estimate_vram_per_user_gb(model, quant) or (
                    total / max(concurrent_users, 1)
                )
            if total <= gpu_vram_gb:
                risky_candidates.append((quant, per_user, total, bits_per_weight(quant)))

        if risky_candidates:
            # Rank risky candidates by priority and pick the best.
            if priority == "quality":
                risky_candidates.sort(key=lambda c: c[3], reverse=True)
            elif priority == "speed":
                risky_candidates.sort(key=lambda c: c[3])
            else:
                risky_candidates.sort(key=lambda c: abs(c[3] - 4.5))
            chosen_quant, per_user, total, _ = risky_candidates[0]
            headroom = gpu_vram_gb - total
            tps_estimate = 100.0
            risky_alternatives = [c[0] for c in risky_candidates[1:3]]
            risky_reasoning = _build_reasoning(
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
            risky_reasoning += (
                f" WARNING: headroom is only {headroom:.1f} GB \u2014 OOM is likely at long context"
                " lengths or with large system prompts. Test carefully before deploying."
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
                reasoning=risky_reasoning,
                alternatives=risky_alternatives,
            )

        return _empty_recommendation(
            gpu_vram_gb,
            concurrent_users,
            model_label,
            f"No quantization of {model_label} is estimated to fit on a {gpu_vram_gb:.0f} GB GPU "
            f"at {concurrent_users} concurrent users.",
        )

    # Rank by priority on formula-derived values.
    if priority == "quality":
        candidates.sort(key=lambda c: c[3], reverse=True)
    elif priority == "speed":
        # Smaller quants are usually faster; rank by lower bpw first.
        candidates.sort(key=lambda c: c[3])
    else:  # balance — prefer middle bpw closest to 4.5
        candidates.sort(key=lambda c: abs(c[3] - 4.5))

    chosen_quant, per_user, total, _ = candidates[0]
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
