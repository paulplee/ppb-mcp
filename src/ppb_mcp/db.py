"""SQLite-backed incremental cache for the PPB dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _row_id(row: dict) -> str:
    """Derive a stable, deterministic ID for a benchmark row."""
    key = json.dumps(
        {
            "suite_id": row.get("suite_id") or "",
            "runner_type": row.get("runner_type") or "",
            "run_type": row.get("run_type") or "",
            "gpu_name": row.get("gpu_name") or "",
            "model_base": row.get("model_base") or "",
            "quant": row.get("quant") or "",
            "timestamp": row.get("timestamp") or "",
        },
        sort_keys=True,
    )
    return hashlib.sha1(key.encode()).hexdigest()


def _json_default(value: Any) -> Any:
    """Fallback JSON encoder for numpy/pandas scalars."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


def _clean_for_json(row: dict) -> dict:
    """Replace NaN floats with None so the row round-trips safely through JSON."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


_DDL = [
    """
    CREATE TABLE IF NOT EXISTS rows (
        row_id            TEXT PRIMARY KEY,
        run_type          TEXT,
        runner_type       TEXT,
        gpu_name          TEXT,
        model_base        TEXT,
        quant             TEXT,
        suite_id          TEXT,
        benchmark_version TEXT,
        data              TEXT NOT NULL,
        ingested_at       TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_run_type    ON rows(run_type)",
    "CREATE INDEX IF NOT EXISTS idx_gpu_model   ON rows(gpu_name, model_base)",
    "CREATE INDEX IF NOT EXISTS idx_runner_type ON rows(runner_type)",
    """
    CREATE TABLE IF NOT EXISTS shard_meta (
        filename    TEXT PRIMARY KEY,
        etag        TEXT,
        last_synced TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sync_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        synced_at     TEXT NOT NULL,
        rows_added    INTEGER NOT NULL,
        shards_synced INTEGER NOT NULL,
        duration_s    REAL NOT NULL
    )
    """,
]


class SQLiteCache:
    """Local SQLite cache for PPB benchmark rows + sync metadata."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            env = os.environ.get("PPB_DB_PATH")
            db_path = Path(env) if env else (Path.cwd() / "ppb_cache.db")
        self.db_path = Path(db_path)

    # ── connection helpers ────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db_path))
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    # ── lifecycle ─────────────────────────────────────────────────────────
    def setup(self) -> None:
        with self._connect() as con:
            for stmt in _DDL:
                con.execute(stmt)
            con.commit()

    # ── freshness / metadata ──────────────────────────────────────────────
    def is_fresh(self, max_age_hours: float) -> bool:
        last = self.last_synced_at()
        if last is None:
            return False
        try:
            ts = datetime.fromisoformat(last)
        except ValueError:
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age_h = (datetime.now(UTC) - ts).total_seconds() / 3600.0
        return age_h < max_age_hours

    def get_shard_etag(self, filename: str) -> str | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT etag FROM shard_meta WHERE filename = ?", (filename,)
            ).fetchone()
        return row[0] if row else None

    def update_shard_meta(self, filename: str, etag: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as con:
            con.execute(
                "INSERT INTO shard_meta(filename, etag, last_synced) VALUES (?, ?, ?) "
                "ON CONFLICT(filename) DO UPDATE SET etag = excluded.etag, "
                "last_synced = excluded.last_synced",
                (filename, etag, now),
            )
            con.commit()

    def clear_shard_meta(self) -> None:
        """Remove all shard_meta rows. Used by force_redownload."""
        with self._connect() as con:
            con.execute("DELETE FROM shard_meta")
            con.commit()

    # ── data ingest / read ────────────────────────────────────────────────
    def upsert_rows(self, rows: list[dict], shard_filename: str) -> int:
        if not rows:
            return 0
        now = datetime.now(UTC).isoformat()
        added = 0
        with self._connect() as con:
            for raw in rows:
                row = _clean_for_json(raw)
                rid = _row_id(row)
                exists = con.execute(
                    "SELECT 1 FROM rows WHERE row_id = ?", (rid,)
                ).fetchone()
                blob = json.dumps(row, default=_json_default)
                con.execute(
                    "INSERT OR REPLACE INTO rows("
                    "row_id, run_type, runner_type, gpu_name, model_base, "
                    "quant, suite_id, benchmark_version, data, ingested_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rid,
                        row.get("run_type"),
                        row.get("runner_type"),
                        row.get("gpu_name"),
                        row.get("model_base"),
                        row.get("quant"),
                        row.get("suite_id"),
                        row.get("benchmark_version"),
                        blob,
                        now,
                    ),
                )
                if not exists:
                    added += 1
            con.commit()
        logger.debug("upserted %d rows from %s (%d new)", len(rows), shard_filename, added)
        return added

    def write_sync_log(
        self, rows_added: int, shards_synced: int, duration_s: float
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as con:
            con.execute(
                "INSERT INTO sync_log(synced_at, rows_added, shards_synced, duration_s) "
                "VALUES (?, ?, ?, ?)",
                (now, int(rows_added), int(shards_synced), float(duration_s)),
            )
            con.commit()

    def load_dataframe(self) -> pd.DataFrame:
        with self._connect() as con:
            cursor = con.execute("SELECT data FROM rows")
            payloads = [json.loads(r[0]) for r in cursor.fetchall()]
        return pd.DataFrame(payloads)

    def row_count(self) -> int:
        try:
            with self._connect() as con:
                n = con.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
            return int(n)
        except sqlite3.OperationalError:
            return 0

    def last_synced_at(self) -> str | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT synced_at FROM sync_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None
