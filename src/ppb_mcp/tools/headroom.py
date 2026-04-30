"""get_gpu_headroom tool."""

from __future__ import annotations

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import GPUHeadroom
from ppb_mcp.tools._filters import is_blank
from ppb_mcp.tools._vram import estimate_total_vram_gb, estimate_vram_per_user_gb


async def get_gpu_headroom(
    gpu_name: str,
    quantization: str,
    model: str,
    concurrent_users: int = 1,
) -> GPUHeadroom:
    """Sanity-check VRAM usage for a specific (gpu, quant, model, users) config.

    USE THIS TOOL when a user wants to verify whether a specific model+quant combination
    will physically fit on a GPU, or to find the maximum safe concurrent user count.

    Uses empirical benchmark rows when an exact match exists; falls back to the
    two-term formula: weights_gb (shared) + kv_cache_per_user_gb × N (per-user).
    Never raises on missing rows — returns is_viable=False with an explanatory
    warning instead.

    NOTE: Do NOT pass "null" for gpu_name or other string parameters — use the exact
    GPU name from list_tested_configs() output.
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    df = await store.get_df()

    vram_col = "gpu_total_vram_gb" if "gpu_total_vram_gb" in df.columns else "gpu_vram_gb"

    # Determine the GPU's VRAM capacity from any matching row.
    gpu_match = df
    if not is_blank(gpu_name) and "gpu_name" in df.columns:
        gpu_match = df[df["gpu_name"].astype(str).str.contains(gpu_name, case=False, na=False)]
    vram_available = 0.0
    if not gpu_match.empty and vram_col in gpu_match.columns:
        vram_available = float(gpu_match[vram_col].dropna().max() or 0.0)

    if vram_available <= 0.0:
        return GPUHeadroom(
            gpu_name=gpu_name,
            quantization=quantization,
            model=model,
            vram_required_gb=0.0,
            vram_available_gb=0.0,
            headroom_gb=0.0,
            max_safe_concurrent_users=0,
            is_viable=False,
            warning=f"GPU {gpu_name!r} not found in the dataset; cannot determine VRAM capacity.",
        )

    # Use the two-term formula: fixed weights + per-user KV cache.
    # For N=1 the difference from the old formula is small; for N>1 it avoids
    # incorrectly multiplying model weights by the user count.
    vram_required = estimate_total_vram_gb(model, quantization, concurrent_users)
    per_user = estimate_vram_per_user_gb(model, quantization)
    if vram_required is None:
        # Fallback: use the column value if any matching row exists. This is a rough
        # ceiling (the GPU's VRAM) and only kicks in when the model name is unparseable.
        if all(c in df.columns for c in ("gpu_name", "quant", "model_base")):
            exact = df[
                df["gpu_name"].astype(str).str.contains(gpu_name, case=False, na=False)
                & (df["quant"] == quantization)
                & df["model_base"].astype(str).str.contains(model, case=False, na=False)
            ]
            if not exact.empty:
                row_vram = exact[vram_col].dropna()
                if not row_vram.empty:
                    per_user = float(row_vram.iloc[0]) / max(concurrent_users, 1)
        if per_user is None:
            return GPUHeadroom(
                gpu_name=gpu_name,
                quantization=quantization,
                model=model,
                vram_required_gb=0.0,
                vram_available_gb=vram_available,
                headroom_gb=vram_available,
                max_safe_concurrent_users=0,
                is_viable=False,
                warning=(
                    f"Could not extract parameter count from model name {model!r}; "
                    "no empirical row matched either."
                ),
            )
        vram_required = per_user * concurrent_users

    if per_user is None:
        per_user = vram_required / max(concurrent_users, 1)

    headroom = vram_available - vram_required
    is_viable = headroom >= 0

    # Find max safe concurrent users using the two-term formula.
    max_safe = 0
    for n in range(1, 33):
        total_n = estimate_total_vram_gb(model, quantization, n)
        if total_n is None:
            total_n = per_user * n
        if total_n <= vram_available:
            max_safe = n
        else:
            break

    warning: str | None = None
    if not is_viable:
        warning = (
            f"Configuration is NOT viable: estimated {vram_required:.1f} GB required "
            f"exceeds {vram_available:.1f} GB available (deficit {-headroom:.1f} GB)."
        )
    elif headroom < 1.0:
        warning = (
            f"Tight headroom: only {headroom:.2f} GB free. Risk of OOM under context-length spikes."
        )

    return GPUHeadroom(
        gpu_name=gpu_name,
        quantization=quantization,
        model=model,
        vram_required_gb=round(vram_required, 2),
        vram_available_gb=round(vram_available, 2),
        headroom_gb=round(headroom, 2),
        max_safe_concurrent_users=max_safe,
        is_viable=is_viable,
        warning=warning,
    )
