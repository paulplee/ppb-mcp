"""list_tested_configs tool."""
from __future__ import annotations

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import TestedConfigs


async def list_tested_configs() -> TestedConfigs:
    """List every tested GPU, model, and quantization in the PPB dataset.

    Call this first to orient yourself before issuing a targeted query.
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    return TestedConfigs(
        gpus=store.get_all_gpus(),
        models=store.get_all_models(),
        quantizations=store.get_all_quantizations(),
        total_benchmark_rows=store.row_count(),
        last_updated=store.get_last_refreshed(),
    )
