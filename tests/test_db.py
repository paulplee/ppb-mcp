"""Tests for the SQLiteCache db layer."""

from __future__ import annotations

import sqlite3

import pytest

from ppb_mcp.db import SQLiteCache


@pytest.fixture
def tmp_cache(tmp_path):
    cache = SQLiteCache(db_path=tmp_path / "test.db")
    cache.setup()
    return cache


def test_setup_creates_tables(tmp_cache, tmp_path):
    con = sqlite3.connect(tmp_path / "test.db")
    tables = {
        r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"rows", "shard_meta", "sync_log"}.issubset(tables)
    con.close()


def test_upsert_and_load(tmp_cache):
    row = {
        "run_type": "qualitative",
        "runner_type": "context-rot",
        "gpu_name": "M4 Pro",
        "model_base": "Qwen",
        "quant": "Q4_K_M",
        "suite_id": "s1",
        "benchmark_version": "0.1.0",
        "context_rot_score": 0.5,
        "timestamp": "2026-04-26T00:00:00Z",
    }
    n = tmp_cache.upsert_rows([row], "shard_001.jsonl")
    assert n == 1
    df = tmp_cache.load_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["context_rot_score"] == pytest.approx(0.5)


def test_upsert_is_idempotent(tmp_cache):
    row = {
        "run_type": "qualitative",
        "gpu_name": "GPU",
        "model_base": "M",
        "quant": "Q4",
        "suite_id": "s1",
        "runner_type": "ctx",
        "benchmark_version": "0.1.0",
        "timestamp": "2026-01-01T00:00:00Z",
    }
    tmp_cache.upsert_rows([row], "shard.jsonl")
    tmp_cache.upsert_rows([row], "shard.jsonl")
    df = tmp_cache.load_dataframe()
    assert len(df) == 1


def test_is_fresh_false_when_empty(tmp_cache):
    assert tmp_cache.is_fresh(1.0) is False


def test_shard_meta_roundtrip(tmp_cache):
    tmp_cache.update_shard_meta("shard_001.jsonl", "abc123")
    assert tmp_cache.get_shard_etag("shard_001.jsonl") == "abc123"
    assert tmp_cache.get_shard_etag("nonexistent.jsonl") is None


def test_load_sync_with_fresh_cache_skips_network(tmp_path, monkeypatch):
    """Verify that load_sync with a fresh SQLite cache doesn't hit HuggingFace."""
    from ppb_mcp.data import PPBDataStore

    db_path = tmp_path / "fresh.db"
    cache = SQLiteCache(db_path=db_path)
    cache.setup()
    cache.upsert_rows(
        [
            {
                "run_type": "real",
                "runner_type": "llama-server",
                "gpu_name": "G",
                "model_base": "M",
                "quant": "Q4_K_M",
                "throughput_tok_s": 10.0,
                "gpu_total_vram_gb": 24.0,
                "concurrent_users": 1,
                "timestamp": "2026-04-26T00:00:00Z",
                "suite_id": "s",
                "benchmark_version": "0.1.0",
            }
        ],
        "shard.jsonl",
    )
    cache.write_sync_log(1, 1, 0.1)

    # Build a store that points at this fresh cache.
    store = PPBDataStore(refresh_interval_hours=1.0)
    store._cache = cache  # inject the prepared cache

    # Patch HfApi to detect any HF use.
    calls = {"n": 0}

    class _Boom:
        def __init__(self, *a, **kw):
            calls["n"] += 1

        def list_repo_files(self, *a, **kw):  # pragma: no cover - should not happen
            calls["n"] += 1
            return []

        def dataset_info(self, *a, **kw):  # pragma: no cover
            calls["n"] += 1
            return None

    monkeypatch.setattr("ppb_mcp.data.HfApi", _Boom)

    store.load_sync()
    assert calls["n"] == 0
    assert store.row_count() == 1


def test_force_redownload_calls_sync(tmp_path, monkeypatch):
    """force_redownload=True bypasses freshness and triggers _incremental_sync."""
    from ppb_mcp.data import PPBDataStore

    db_path = tmp_path / "x.db"
    cache = SQLiteCache(db_path=db_path)
    cache.setup()
    cache.write_sync_log(0, 0, 0.0)

    store = PPBDataStore(refresh_interval_hours=24.0)
    store._cache = cache

    called = {"n": 0}

    def fake_sync(*, force_redownload=False):
        called["n"] += 1
        called["force"] = force_redownload
        store._df = __import__("pandas").DataFrame()
        store._loaded = True
        store._last_refreshed = "now"

    monkeypatch.setattr(store, "_incremental_sync", fake_sync)
    store.load_sync(force_redownload=True)
    assert called["n"] == 1
    assert called["force"] is True
