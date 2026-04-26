"""
ppb_mcp.data — HuggingFace data layer for Poor Paul's Benchmark.

Verified schema (paulplee/ppb-results, ~30,841 rows × 61 cols, 260 JSONL shards):

    Key columns (real → external/Pydantic name):
        throughput_tok_s  → tokens_per_second
        gpu_name          → gpu_name
        gpu_total_vram_gb → vram_gb
        model_base        → model
        quant             → quantization
        concurrent_users  → concurrent_users
        backends          → backend

Loading is now backed by a local SQLite cache (`ppb_mcp.db.SQLiteCache`).
On startup, if the cache is fresh (synced within REFRESH_INTERVAL_HOURS),
we load directly from SQLite and skip HuggingFace entirely. Otherwise we
check the dataset's git commit SHA — if unchanged we still skip the
download; if changed (or first run / force_redownload) we fetch the
JSONL shards and incrementally upsert them into SQLite.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

import anyio
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from ppb_mcp.db import SQLiteCache

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

_COMMIT_KEY = "__commit__"


def _read_jsonl_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line in %s: %s", path, exc)
    return rows


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
        self._cache = SQLiteCache()

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

    # ── schema validation ─────────────────────────────────────────────────
    def _validate_schema(self, df: pd.DataFrame) -> None:
        cols = set(df.columns)
        missing = REQUIRED_COLUMNS - cols
        if missing:
            logger.warning(
                "Required columns missing from dataset: %s. Continuing with available columns.",
                sorted(missing),
            )

    # ── loading ────────────────────────────────────────────────────────────
    def load_sync(self, *, force_redownload: bool = False) -> None:
        """Synchronously load (or reload) the dataset. Safe to call at startup."""
        # Test hook bypasses SQLite entirely.
        if self._loader is not None:
            df = self._loader()
            self._validate_schema(df)
            self._df = df
            self._last_refreshed = datetime.now(UTC).isoformat()
            self._loaded = True
            logger.info("Loaded PPB dataset (loader hook): shape=%s", df.shape)
            return

        self._cache.setup()
        if not force_redownload and self._cache.is_fresh(self.refresh_interval_hours):
            df = self._cache.load_dataframe()
            self._validate_schema(df)
            self._df = df
            self._last_refreshed = self._cache.last_synced_at()
            self._loaded = True
            logger.info("Loaded from SQLite cache: shape=%s", df.shape)
            return

        self._incremental_sync(force_redownload=force_redownload)

    def _incremental_sync(self, *, force_redownload: bool = False) -> None:
        """Fetch only new/changed JSONL shards from HuggingFace, upsert into SQLite."""
        t0 = time.monotonic()
        api = HfApi()

        try:
            info = api.dataset_info(self.dataset)
            remote_commit = getattr(info, "sha", None)
        except Exception as exc:  # pragma: no cover - network fallback
            logger.warning(
                "Could not fetch dataset_info; proceeding without commit check: %s", exc
            )
            remote_commit = None

        stored_commit = self._cache.get_shard_etag(_COMMIT_KEY)

        if force_redownload:
            self._cache.clear_shard_meta()
            stored_commit = None

        commit_changed = (
            force_redownload
            or stored_commit is None
            or remote_commit is None
            or remote_commit != stored_commit
        )

        if not commit_changed:
            logger.info("Dataset commit unchanged (%s); skipping HF download.", remote_commit)
            df = self._cache.load_dataframe()
            self._validate_schema(df)
            self._df = df
            self._cache.write_sync_log(0, 0, time.monotonic() - t0)
            self._last_refreshed = self._cache.last_synced_at()
            self._loaded = True
            return

        repo_files = api.list_repo_files(self.dataset, repo_type="dataset")
        shards = [f for f in repo_files if f.endswith(".jsonl")]
        if not shards:
            raise RuntimeError(f"No .jsonl shards found in dataset {self.dataset!r}")

        rows_added = 0
        shards_synced = 0
        for shard in shards:
            already_synced_etag = self._cache.get_shard_etag(shard)
            if (
                not force_redownload
                and already_synced_etag is not None
                and remote_commit is not None
                and already_synced_etag == remote_commit
            ):
                continue
            local_path = hf_hub_download(
                self.dataset,
                shard,
                repo_type="dataset",
                force_download=force_redownload,
            )
            shard_rows = _read_jsonl_rows(local_path)
            rows_added += self._cache.upsert_rows(shard_rows, shard)
            self._cache.update_shard_meta(shard, remote_commit or "unknown")
            shards_synced += 1

        self._cache.update_shard_meta(_COMMIT_KEY, remote_commit or "unknown")

        duration = time.monotonic() - t0
        self._cache.write_sync_log(rows_added, shards_synced, duration)

        df = self._cache.load_dataframe()
        self._validate_schema(df)
        self._df = df
        self._last_refreshed = self._cache.last_synced_at()
        self._loaded = True
        logger.info(
            "Incremental sync complete: shards=%d rows_added=%d duration=%.1fs shape=%s",
            shards_synced,
            rows_added,
            duration,
            df.shape,
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
            if self._loader is not None:
                new_df = await anyio.to_thread.run_sync(self._loader)
            else:
                await anyio.to_thread.run_sync(
                    lambda: self._incremental_sync(force_redownload=True)
                )
                new_df = self._df
        except (HfHubHTTPError, OSError, RuntimeError, ValueError) as exc:
            logger.error("Dataset refresh failed; serving stale cache. Error: %s", exc)
            return False
        async with self._lock:
            self._validate_schema(new_df)
            self._df = new_df
            if self._loader is not None:
                self._last_refreshed = datetime.now(UTC).isoformat()
            else:
                self._last_refreshed = self._cache.last_synced_at()
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
