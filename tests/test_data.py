"""Tests for the PPBDataStore data layer."""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from ppb_mcp.data import PPBDataStore


def _empty_loader() -> pd.DataFrame:
    return pd.DataFrame()


def test_load_sync_populates_df(fixture_df: pd.DataFrame) -> None:
    s = PPBDataStore(loader=lambda: fixture_df)
    s.load_sync()
    assert s.row_count() == len(fixture_df)
    assert "TestGPU-24GB" in s.get_all_gpus()
    assert "Llama-7B" in s.get_all_models()
    assert "Q4_K_M" in s.get_all_quantizations()
    assert s.get_last_refreshed() != "never"


@pytest.mark.asyncio
async def test_get_df_lazy_loads(fixture_df: pd.DataFrame) -> None:
    s = PPBDataStore(loader=lambda: fixture_df)
    df = await s.get_df()
    assert len(df) == len(fixture_df)


@pytest.mark.asyncio
async def test_refresh_failure_keeps_stale_data(fixture_df: pd.DataFrame, caplog) -> None:
    calls = {"n": 0}

    def flaky_loader() -> pd.DataFrame:
        calls["n"] += 1
        if calls["n"] == 1:
            return fixture_df
        raise RuntimeError("simulated network failure")

    s = PPBDataStore(loader=flaky_loader)
    s.load_sync()
    pre_count = s.row_count()
    with caplog.at_level(logging.ERROR, logger="ppb_mcp.data"):
        ok = await s.refresh()
    assert ok is False
    assert s.row_count() == pre_count
    assert any("refresh failed" in r.message.lower() for r in caplog.records)


def test_schema_validation_warns_on_missing_required(caplog) -> None:
    bad_df = pd.DataFrame([{"foo": 1, "bar": 2}])
    s = PPBDataStore(loader=lambda: bad_df)
    with caplog.at_level(logging.WARNING, logger="ppb_mcp.data"):
        s.load_sync()
    assert any("required columns missing" in r.message.lower() for r in caplog.records)


def test_reserved_null_columns_do_not_warn(fixture_df: pd.DataFrame, caplog) -> None:
    s = PPBDataStore(loader=lambda: fixture_df)
    with caplog.at_level(logging.WARNING, logger="ppb_mcp.data"):
        s.load_sync()
    # The fixture has tags=None, quality_score=None — no warning should fire about them.
    for record in caplog.records:
        assert "tags" not in record.message.lower()
        assert "quality_score" not in record.message.lower()
