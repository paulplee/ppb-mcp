"""list_tested_configs tool."""

from __future__ import annotations

from ppb_mcp.data import PPBDataStore
from ppb_mcp.models import TestedConfigs


async def list_tested_configs() -> TestedConfigs:
    """List every tested GPU, model, quantization, and runner type in the PPB dataset.

    USE THIS TOOL first to orient yourself before issuing a targeted query.
    The runner_types field shows valid values for the runner_type filter in
    query_ppb_results — always filter by runner_type when comparing speeds, as
    "llama-bench" and "llama-server-loadtest" numbers are NOT directly comparable.

    NOTE: Do NOT pass "null" for any filter parameter — omit it instead.
    """
    store = PPBDataStore.instance()
    await store.ensure_loaded()
    return TestedConfigs(
        gpus=store.get_all_gpus(),
        models=store.get_all_models(),
        quantizations=store.get_all_quantizations(),
        runner_types=store.get_all_runner_types(),
        total_benchmark_rows=store.row_count(),
        last_updated=store.get_last_refreshed(),
    )
