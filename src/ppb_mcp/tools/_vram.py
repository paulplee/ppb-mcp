"""
VRAM-estimation helpers shared by `recommend.py` and `headroom.py`.

When no benchmark row exists for an exact (gpu, model, quant, users) combo,
we fall back to a formula:

    vram_per_user_gb ≈ (params_billions × bits_per_weight / 8) × 1.15

The 1.15 multiplier is a 15% overhead for KV cache + runtime. Total VRAM
at N concurrent users is approximated as `vram_per_user_gb × N` (the spec's
simplification — overestimates because model weights don't scale with users,
but it errs on the safe side, which is desirable for a recommendation tool).
"""

from __future__ import annotations

import re

# Bits-per-weight for every quant label observed in the dataset (32 unique).
# Numbers reflect approximate effective storage cost incl. small overheads.
BITS_PER_WEIGHT: dict[str, float] = {
    "BF16": 16.0,
    "F16": 16.0,
    "Q8_0": 8.5,
    "Q8_K_XL": 8.5,
    "Q6_K": 6.6,
    "Q6_K_S": 6.6,
    "Q6_K_XL": 6.6,
    "Q5_K_M": 5.5,
    "Q5_K_S": 5.5,
    "Q5_K_XL": 5.5,
    "Q4_K_M": 4.5,
    "Q4_K_S": 4.5,
    "Q4_K_L": 4.5,
    "Q4_K_XL": 4.5,
    "Q4_0": 4.5,
    "Q4_1": 4.5,
    "IQ4_NL": 4.5,
    "IQ4_NL_XL": 4.5,
    "IQ4_XS": 4.25,
    "Q3_K_M": 3.5,
    "Q3_K_S": 3.5,
    "Q3_K_XL": 3.5,
    "IQ3_S": 3.5,
    "IQ3_XXS": 3.0,
    "Q2_K": 2.6,
    "Q2_K_L": 2.6,
    "Q2_K_XL": 2.6,
    "IQ2_M": 2.6,
    "IQ2_XXS": 2.2,
    "IQ1_M": 1.6,
    # Unsloth Dynamic profile labels — treat as ~mid-quality (Q4 equivalent).
    "Balanced": 4.5,
    "Quality": 5.5,
}

DEFAULT_BPW = 4.5


def bits_per_weight(quant: str) -> float:
    """Return effective bits-per-weight for a quant label (default 4.5)."""
    return BITS_PER_WEIGHT.get(quant, DEFAULT_BPW)


# Match `-0.8B`, `-2B`, `-9B`, `-27B`, `-35B-A3B`, `E2B`, `E4B`, `20b`, `32B`, etc.
_PARAM_RE = re.compile(r"(?:^|[-_\.])([Ee]?)(\d+(?:\.\d+)?)[Bb](?![a-zA-Z])")


def extract_params_billions(model_name: str) -> float | None:
    """Extract approximate parameter count in billions from a model_base name.

    Handles formats:
      Qwen3.5-9B           -> 9.0
      Qwen3.5-0.8B         -> 0.8
      gemma-4-E4B-it       -> 4.0   (the 'E' prefix is dropped)
      gpt-oss-20b          -> 20.0
      Qwen3.5-35B-A3B      -> 35.0  (first match wins; A3B = active 3B for MoE)
      DeepSeek-R1-Distill-Qwen-32B -> 32.0
    """
    if not model_name:
        return None
    m = _PARAM_RE.search(model_name)
    if not m:
        return None
    try:
        return float(m.group(2))
    except ValueError:
        return None


def estimate_vram_per_user_gb(model_name: str, quant: str) -> float | None:
    """Formula-based VRAM-per-request estimate. Returns None if params unparseable."""
    params_b = extract_params_billions(model_name)
    if params_b is None:
        return None
    bpw = bits_per_weight(quant)
    return (params_b * bpw / 8.0) * 1.15


# Approximate KV cache GB per concurrent user at n_ctx=8192 (conservative upper bound).
# Derivation: 4 × num_layers × num_kv_heads × head_dim × n_ctx × 2 bytes (GQA, fp16) / 1e9
# Using 0.25 GB/B is deliberately conservative (overestimates) to err on the safe side.
_KV_CACHE_GB_PER_B_PARAMS: float = 0.25  # at n_ctx=8192; intentional overestimate for safety


def estimate_total_vram_gb(
    model_name: str,
    quant: str,
    concurrent_users: int,
    n_ctx: int = 8192,
) -> float | None:
    """Two-term VRAM estimate: fixed weights + linear KV cache per user.

    Unlike estimate_vram_per_user_gb × N (which over-counts by replicating weights),
    this correctly models shared weights plus per-user KV cache scaling.
    Returns None if the parameter count cannot be parsed from model_name.
    """
    params_b = extract_params_billions(model_name)
    if params_b is None:
        return None
    bpw = bits_per_weight(quant)
    weights_gb = params_b * bpw / 8.0 * 1.1  # 10% runtime overhead on weights
    # KV cache scales with n_ctx; 8192 is the reference baseline
    ctx_scale = n_ctx / 8192.0
    kv_per_user = params_b * _KV_CACHE_GB_PER_B_PARAMS * ctx_scale
    return weights_gb + kv_per_user * concurrent_users
