"""
ppb_mcp.data — HuggingFace data layer for Poor Paul's Benchmark.

Verified schema (paulplee/ppb-results, ~30,841 rows × 61 cols, 260 JSONL shards):

    Key columns (real → external/Pydantic name):
        throughput_tok_s  → tokens_per_second   (float64, no nulls)
        gpu_name          → gpu_name            (str; 3 values today: RTX 5090, M4 Pro, GB10)
        gpu_vram_gb       → vram_gb             (float64, 7265 nulls)
        gpu_total_vram_gb → vram_gb (canonical, multi-GPU-aware)
        model_base        → model               (str, 34 unique, e.g. "Qwen3.5-9B")
        model             → model_full_path     (str, raw GGUF path)
        model_org         → model_org           (str: unsloth, mudler, Jackrong)
        quant             → quantization        (str, 32 unique, 650 nulls)
        concurrent_users  → concurrent_users    (float64; values 1, 2, 4, 8, 16, 32)
        backends          → backend             (str: CUDA 13.0/13.1, Metal/Metal 4)

    Reserved-but-currently-all-null columns (whitelisted, no warning):
        llm_engine_version, split_mode, tensor_split, quality_score, tags

    There is no `vram_usage_gb` column — VRAM-per-request must be derived
    (see ppb_mcp.tools._vram for the formula fallback).

Loading note: `datasets.load_dataset()` FAILS on this dataset due to a
pyarrow null→string cast error across shards. We use the raw-JSONL path
via huggingface_hub instead.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "throughput_tok_s",
    "gpu_name",
    "gpu_total_vram_gb",
    "model_base",
    "quant",
    "concurrent_users",
}
RESERVED_NULL_COLUMNS = {
    "llm_engine_version",
    "split_mode",
    "tensor_split",
    "quality_score",
    "tags",
}


def _load_jsonl_shards(repo_id: str, *, force_redownload: bool = False) -> pd.DataFrame:
    """Load all JSONL shards from a HuggingFace dataset repo into a DataFrame."""
    api = HfApi()
    files = [
        f
        for f in api.list_repo_files(repo_id, repo_type="dataset")
        if f.endswith(".jsonl")
    ]
    if not files:
        raise RuntimeError(f"No .jsonl shards found in dataset {repo_id!r}")

    rows: list[dict[str, Any]] = []
    for shard in files:
        local_path = hf_hub_download(
            repo_id,
            shard,
            repo_type="dataset",
            force_download=force_redownload,
        )
        with open(local_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line in %s: %s", shard, exc)
    return pd.DataFrame(rows)


def _clear_repo_cache(repo_id: str) -> None:
    """Best-effort: delete the local HF cache snapshot for a dataset repo."""
    cache_root = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub"
    repo_dir_name = "datasets--" + repo_id.replace("/", "--")
    repo_dir = cache_root / repo_dir_name
    if repo_dir.exists():
        try:
            shutil.rmtree(repo_dir)
            logger.info("Cleared HF cache at %s", repo_dir)
        except OSError as exc:
            logger.warning("Failed to clear HF cache at %s: %s", repo_dir, exc)


class PPBDataStore:
    """Singleton store for the PPB benchmark DataFrame with background refresh."""

    _instance: PPBDataStore | None = None

    def __init__(
        self,
        dataset: str | None = None,
        refresh_interval_hours: float | None = None,
        loader: Any | None = None,
    ) -> None:
        self.dataset = dataset or os.environ.get("HF_DATASET", "paulplee/ppb-results")
        self.refresh_interval_hours = (
            refresh_interval_hours
            if refresh_interval_hours is not None
            else float(os.environ.get("REFRESH_INTERVAL_HOURS", "1"))
        )
        # `loader` is a hook for tests: a callable that returns a DataFrame.
        self._loader = loader
        self._df: pd.DataFrame = pd.DataFrame()
        self._lock = asyncio.Lock()
        self._last_refreshed: str | None = None
        self._loaded = False

    # ── singleton accessor ────────────────────────────────────────────────
    @classmethod
    def instance(cls) -> PPBDataStore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_instance(cls, store: PPBDataStore | None) -> None:
        """Replace (or clear) the global singleton — used by tests."""
        cls._instance = store

    # ── loading ────────────────────────────────────────────────────────────
    def _do_load(self, force_redownload: bool = False) -> pd.DataFrame:
        if self._loader is not None:
            return self._loader()
        if force_redownload:
            _clear_repo_cache(self.dataset)
        return _load_jsonl_shards(self.dataset, force_redownload=force_redownload)

    def _validate_schema(self, df: pd.DataFrame) -> None:
        cols = set(df.columns)
        missing = REQUIRED_COLUMNS - cols
        if missing:
            logger.warning(
                "Required columns missing from dataset: %s. Continuing with available columns.",
                sorted(missing),
            )
        # Reserved-null columns are tolerated silently if present-and-null.
        # Other novel columns are also fine; do not warn.

    def load_sync(self, *, force_redownload: bool = False) -> None:
        """Synchronously load (or reload) the dataset. Safe to call at startup."""
        df = self._do_load(force_redownload=force_redownload)
        self._validate_schema(df)
        self._df = df
        self._last_refreshed = datetime.now(UTC).isoformat()
        self._loaded = True
        logger.info(
            "Loaded PPB dataset %s: shape=%s, columns=%d",
            self.dataset,
            df.shape,
            len(df.columns),
        )

    async def ensure_loaded(self) -> None:
        """Lazy-load on first access if not already loaded."""
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            await anyio.to_thread.run_sync(self.load_sync)

    async def refresh(self) -> bool:
        """Force a refresh. Returns True on success; False (and keeps stale cache) on failure."""
        try:
            new_df = await anyio.to_thread.run_sync(
                lambda: self._do_load(force_redownload=True)
            )
        except (HfHubHTTPError, OSError, RuntimeError, ValueError) as exc:
            logger.error("Dataset refresh failed; serving stale cache. Error: %s", exc)
            return False
        async with self._lock:
            self._validate_schema(new_df)
            self._df = new_df
            self._last_refreshed = datetime.now(UTC).isoformat()
        logger.info("Refreshed PPB dataset: shape=%s", new_df.shape)
        return True

    async def run_refresh_loop(self) -> None:
        """Background task: refresh every `refresh_interval_hours`."""
        interval_s = self.refresh_interval_hours * 3600
        while True:
            await anyio.sleep(interval_s)
            await self.refresh()

    # ── accessors ──────────────────────────────────────────────────────────
    async def get_df(self) -> pd.DataFrame:
        """Returns the current cached DataFrame (thread-safe)."""
        await self.ensure_loaded()
        async with self._lock:
            return self._df

    def df_unsafe(self) -> pd.DataFrame:
        """Synchronous accessor (for sync tool implementations after ensure_loaded)."""
        return self._df

    def get_all_gpus(self) -> list[str]:
        if "gpu_name" not in self._df.columns:
            return []
        return sorted(self._df["gpu_name"].dropna().unique().tolist())

    def get_all_models(self) -> list[str]:
        if "model_base" not in self._df.columns:
            return []
        return sorted(self._df["model_base"].dropna().unique().tolist())

    def get_all_quantizations(self) -> list[str]:
        if "quant" not in self._df.columns:
            return []
        return sorted(self._df["quant"].dropna().unique().tolist())

    def get_last_refreshed(self) -> str:
        return self._last_refreshed or "never"

    def row_count(self) -> int:
        return len(self._df)
