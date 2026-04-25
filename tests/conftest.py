"""Pytest fixtures: synthetic PPB DataFrame mirroring the real schema."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest

from ppb_mcp.data import PPBDataStore


def _row(
    *,
    gpu_name: str,
    vram: float,
    model_base: str,
    quant: str,
    users: int,
    tps: float,
    backend: str = "CUDA",
    org: str = "unsloth",
) -> dict:
    return {
        "model": f"{org}/{model_base}-GGUF/{model_base}-{quant}.gguf",
        "model_base": model_base,
        "quant": quant,
        "model_org": org,
        "model_repo": f"{org}/{model_base}-GGUF",
        "runner_type": "llama-server",
        "llm_engine_name": "llama.cpp",
        "llm_engine_version": None,
        "gpu_name": gpu_name,
        "gpu_vram_gb": vram,
        "gpu_total_vram_gb": vram,
        "gpu_count": 1,
        "gpu_names": gpu_name,
        "backends": backend,
        "n_ctx": 4096,
        "n_batch": 512,
        "split_mode": None,
        "tensor_split": None,
        "concurrent_users": float(users),
        "task_type": "text-generation",
        "prompt_dataset": "sharegpt-v3",
        "num_prompts": 10 * users,
        "n_predict": 256,
        "throughput_tok_s": tps,
        "avg_ttft_ms": 50.0,
        "p50_ttft_ms": 48.0,
        "p99_ttft_ms": 80.0,
        "avg_itl_ms": 12.0,
        "p50_itl_ms": 11.5,
        "p99_itl_ms": 25.0,
        "quality_score": None,
        "tags": None,
        "submitter": "test-fixture",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "submitted_at": "2026-01-01T00:00:00+00:00",
        "schema_version": "0.1.0",
        "benchmark_version": "0.1.0",
        "row_id": f"{gpu_name}-{model_base}-{quant}-{users}",
    }


def _build_fixture_df() -> pd.DataFrame:
    rows: list[dict] = []
    # GPU A: 8 GB — only Q4_K_M Llama-2B fits at 1 user (edge case)
    rows.append(
        _row(
            gpu_name="TestGPU-8GB",
            vram=8.0,
            model_base="Llama-2B",
            quant="Q4_K_M",
            users=1,
            tps=120.0,
        )
    )
    rows.append(
        _row(
            gpu_name="TestGPU-8GB",
            vram=8.0,
            model_base="Llama-2B",
            quant="Q4_K_M",
            users=2,
            tps=180.0,
        )
    )
    # Smaller quant of bigger model on 8 GB
    rows.append(
        _row(
            gpu_name="TestGPU-8GB",
            vram=8.0,
            model_base="Llama-7B",
            quant="Q4_K_M",
            users=1,
            tps=42.0,
        )
    )

    # GPU B: 16 GB — Q4 / Q5 / Q8 of Llama-7B at users=1, 2
    for quant, tps_factor in [("Q4_K_M", 1.0), ("Q5_K_M", 0.85), ("Q8_0", 0.6)]:
        for users, multiplier in [(1, 1.0), (2, 1.4)]:
            rows.append(
                _row(
                    gpu_name="TestGPU-16GB",
                    vram=16.0,
                    model_base="Llama-7B",
                    quant=quant,
                    users=users,
                    tps=80.0 * tps_factor * multiplier,
                )
            )
    # Plus some Llama-13B on 16 GB
    rows.append(
        _row(
            gpu_name="TestGPU-16GB",
            vram=16.0,
            model_base="Llama-13B",
            quant="Q4_K_M",
            users=1,
            tps=35.0,
        )
    )

    # GPU C: 24 GB — full sweep + a Mistral model family
    for quant, tps_factor in [("Q4_K_M", 1.0), ("Q5_K_M", 0.9), ("Q8_0", 0.7)]:
        for users in (1, 2, 4, 8):
            rows.append(
                _row(
                    gpu_name="TestGPU-24GB",
                    vram=24.0,
                    model_base="Llama-7B",
                    quant=quant,
                    users=users,
                    tps=120.0 * tps_factor * (users**0.5),
                )
            )
            rows.append(
                _row(
                    gpu_name="TestGPU-24GB",
                    vram=24.0,
                    model_base="Mistral-7B",
                    quant=quant,
                    users=users,
                    tps=110.0 * tps_factor * (users**0.5),
                    org="mudler",
                )
            )
    rows.append(
        _row(
            gpu_name="TestGPU-24GB",
            vram=24.0,
            model_base="Llama-13B",
            quant="Q4_K_M",
            users=1,
            tps=55.0,
        )
    )
    rows.append(
        _row(
            gpu_name="TestGPU-24GB",
            vram=24.0,
            model_base="Llama-13B",
            quant="Q4_K_M",
            users=2,
            tps=82.0,
        )
    )

    return pd.DataFrame(rows)


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    return _build_fixture_df()


@pytest.fixture
def store(fixture_df: pd.DataFrame) -> Iterator[PPBDataStore]:
    """A PPBDataStore wired with a synthetic loader; installed as the global singleton."""
    s = PPBDataStore(loader=lambda: fixture_df)
    s.load_sync()
    PPBDataStore.set_instance(s)
    try:
        yield s
    finally:
        PPBDataStore.set_instance(None)
