"""FastMCP server entrypoint for ppb-mcp."""

from __future__ import annotations

import collections
import json
import logging
import math
import os
import sys
import time
from contextlib import asynccontextmanager

import anyio
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.types import ASGIApp, Receive, Scope, Send

from ppb_mcp import __version__
from ppb_mcp.data import PPBDataStore
from ppb_mcp.tools.combined import get_combined_scores
from ppb_mcp.tools.compare_quantitative import compare_quants_quantitative
from ppb_mcp.tools.compare_quants import compare_quants_qualitative
from ppb_mcp.tools.context_rot import get_context_rot_breakdown
from ppb_mcp.tools.explain_result import explain_result
from ppb_mcp.tools.headroom import get_gpu_headroom
from ppb_mcp.tools.list_configs import list_tested_configs
from ppb_mcp.tools.qualitative_query import query_qualitative_results
from ppb_mcp.tools.qualitative_summary import get_qualitative_summary
from ppb_mcp.tools.query import query_ppb_results
from ppb_mcp.tools.rank import rank_by_priority
from ppb_mcp.tools.recommend import recommend_quantization
from ppb_mcp.tools.recommend_hardware import recommend_hardware
from ppb_mcp.tools.tool_accuracy import get_tool_accuracy_breakdown

logger = logging.getLogger("ppb_mcp")


class _AcceptPatchMiddleware:
    """Patch the Accept header on /mcp requests to include text/event-stream.

    The MCP streamable-HTTP spec requires clients to send
    ``Accept: application/json, text/event-stream``. Many clients only send
    ``Accept: application/json``, which causes the mcp SDK to return 406.
    This middleware silently adds ``text/event-stream`` when absent, making
    the server tolerant of non-compliant clients without altering any other
    behaviour.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path", "").rstrip("/") == "/mcp":
            accept_bytes = next((v for k, v in scope["headers"] if k == b"accept"), None)
            accept = accept_bytes.decode("latin-1") if accept_bytes is not None else ""
            # MCP streamable-HTTP requires Accept to contain BOTH application/json AND
            # text/event-stream. Patch whichever values are missing so non-compliant
            # clients (that send only one of the two) are not rejected with 406/400.
            needs_json = "application/json" not in accept
            needs_sse = "text/event-stream" not in accept
            if needs_json or needs_sse:
                parts = [accept] if accept else []
                if needs_json:
                    parts.append("application/json")
                if needs_sse:
                    parts.append("text/event-stream")
                new_accept = ", ".join(p for p in parts if p)
                new_headers = [(k, v) for k, v in scope["headers"] if k != b"accept"]
                new_headers.append((b"accept", new_accept.encode("latin-1")))
                scope = dict(scope)
                scope["headers"] = new_headers
        await self.app(scope, receive, send)


def _safe_dump(obj: object) -> object:
    """Recursively replace NaN/Inf float values with None for JSON-safe serialization.

    Starlette's JSONResponse uses json.dumps(..., allow_nan=False), which raises
    ValueError on NaN/Inf.  Some raw dataset rows have NaN for numeric columns
    (e.g. qualitative rows have no throughput_tok_s value), which propagate through
    the DataFrame → Pydantic model → model_dump() pipeline as Python float NaN.
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _safe_dump(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_dump(v) for v in obj]
    return obj


def _parse_rate_limit() -> tuple[int, int]:
    """Parse RATE_LIMIT env var (e.g. '60/minute', '200/hour'). Returns (max_requests, window_seconds)."""
    raw = os.environ.get("RATE_LIMIT", "60/minute")
    try:
        count_str, period = raw.split("/", 1)
        count = int(count_str.strip())
        period = period.strip().lower()
        if period in ("second", "sec", "s"):
            window = 1
        elif period in ("hour", "hr", "h"):
            window = 3600
        else:  # minute (default)
            window = 60
        return count, window
    except (ValueError, AttributeError):
        return 60, 60


class _RateLimitMiddleware:
    """Sliding-window IP rate limiter for /api/ endpoints.

    Reads the RATE_LIMIT environment variable (e.g. ``"60/minute"``, ``"200/hour"``).
    Applies only to paths starting with ``/api/``; the MCP protocol path (``/mcp``)
    and ``/health`` are intentionally left unthrottled.

    Because asyncio is single-threaded cooperative multitasking, the per-IP
    sliding-window updates are atomic without an additional lock.
    """

    def __init__(self, app: ASGIApp, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._store: dict[str, collections.deque] = {}

    def _client_ip(self, scope: Scope) -> str:
        headers = {k: v for k, v in scope.get("headers", [])}
        xff = headers.get(b"x-forwarded-for", b"").decode("latin-1")
        if xff:
            return xff.split(",")[0].strip()
        client = scope.get("client")
        return client[0] if client else "unknown"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path", "").startswith("/api/"):
            ip = self._client_ip(scope)
            now = time.monotonic()
            window = self._store.setdefault(ip, collections.deque())
            cutoff = now - self.window_seconds
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - window[0])))
                body = json.dumps({"error": "Rate limit exceeded. Try again later."}).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 429,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"retry-after", str(retry_after).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
            window.append(now)
        await self.app(scope, receive, send)


def _configure_logging(transport: str) -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    # In stdio mode, logs MUST go to stderr only — stdout is the MCP wire.
    stream = sys.stderr
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=stream,
        force=True,
    )


@asynccontextmanager
async def _lifespan(server):  # noqa: ANN001 - FastMCP passes its own server obj
    """Load dataset on startup; run background refresh for the server's lifetime."""
    store = PPBDataStore.instance()
    try:
        async with anyio.create_task_group() as tg:
            await store.ensure_loaded()
            tg.start_soon(store.run_refresh_loop)
            yield
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Lifespan error: %s", exc)
        yield  # still serve, just without background refresh


app: FastMCP = FastMCP(
    name="Poor Paul's MCP",
    instructions=(
        "Queryable GPU inference benchmarks from Poor Paul's Benchmark (PPB). "
        "Quantitative tools: recommend_quantization, query_ppb_results, "
        "get_gpu_headroom, list_tested_configs, compare_quants_quantitative, "
        "get_combined_scores, rank_by_priority, recommend_hardware, explain_result. "
        "Qualitative tools: get_qualitative_summary, query_qualitative_results, "
        "get_context_rot_breakdown, get_tool_accuracy_breakdown, compare_quants_qualitative. "
        "Data source: https://huggingface.co/datasets/paulplee/ppb-results"
    ),
    version=__version__,
    lifespan=_lifespan,
)

# Register the quantitative tools.
app.tool(list_tested_configs)
app.tool(query_ppb_results)
app.tool(recommend_quantization)
app.tool(get_gpu_headroom)
app.tool(compare_quants_quantitative)
app.tool(get_combined_scores)
app.tool(rank_by_priority)
app.tool(recommend_hardware)
app.tool(explain_result)

# Register the qualitative tools.
app.tool(get_qualitative_summary)
app.tool(query_qualitative_results)
app.tool(get_context_rot_breakdown)
app.tool(get_tool_accuracy_breakdown)
app.tool(compare_quants_qualitative)


# REST API endpoints (HTTP transport only).
try:
    from pydantic import ValidationError
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    from ppb_mcp.rest_schemas import (
        CompareQueryParams,
        ContextRotQueryParams,
        QualitativeQueryParams,
        ResultsQueryParams,
        ToolAccuracyQueryParams,
    )

    # ── Allowed origins for CORS ─────────────────────────────────────────────
    _CORS_ORIGINS = [
        "https://poorpaul.dev",
        "https://www.poorpaul.dev",
        "https://mcp.poorpaul.dev",
    ]

    def _cors_headers(request: Request) -> dict[str, str]:
        origin = request.headers.get("origin", "")
        allowed = origin if origin in _CORS_ORIGINS or origin.startswith("http://localhost") else ""
        headers: dict[str, str] = {
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Cache-Control": "public, max-age=60",
        }
        if allowed:
            headers["Access-Control-Allow-Origin"] = allowed
        return headers

    # ── /health ──────────────────────────────────────────────────────────────

    @app.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        store = PPBDataStore.instance()
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "dataset": store.dataset,
                "dataset_rows": store.row_count(),
                "last_refreshed": store.get_last_refreshed(),
                "db_path": str(store._cache.db_path),
                "cache_row_count": store._cache.row_count(),
            },
            headers=_cors_headers(request),
        )

    # ── /api/v1/summary ──────────────────────────────────────────────────────

    @app.custom_route("/api/v1/summary", methods=["GET"])
    async def api_summary(request: Request) -> JSONResponse:
        """List all tested GPUs, models, quantizations, runner types, and row count."""
        result = await list_tested_configs()
        return JSONResponse(_safe_dump(result.model_dump()), headers=_cors_headers(request))

    # ── /api/v1/hardware ─────────────────────────────────────────────────────

    @app.custom_route("/api/v1/hardware", methods=["GET"])
    async def api_hardware(request: Request) -> JSONResponse:
        """List GPUs with VRAM and result counts."""
        store = PPBDataStore.instance()
        await store.ensure_loaded()
        df = await store.get_df()
        rows: list[dict] = []
        if not df.empty and "gpu_name" in df.columns:
            vram_col = "gpu_total_vram_gb" if "gpu_total_vram_gb" in df.columns else "gpu_vram_gb"
            quant_df = (
                df[df.get("run_type", "quantitative") != "qualitative"]
                if "run_type" in df.columns
                else df
            )
            for gpu, grp in quant_df.groupby("gpu_name", dropna=True):
                vram = None
                if vram_col in grp.columns:
                    v = grp[vram_col].dropna()
                    if not v.empty:
                        vram = round(float(v.iloc[0]), 1)
                rows.append(
                    {
                        "gpu_name": str(gpu),
                        "gpu_vram_gb": vram,
                        "result_count": int(len(grp)),
                    }
                )
            rows.sort(key=lambda r: r["result_count"], reverse=True)
        return JSONResponse({"hardware": rows}, headers=_cors_headers(request))

    # ── /api/v1/models ───────────────────────────────────────────────────────

    @app.custom_route("/api/v1/models", methods=["GET"])
    async def api_models(request: Request) -> JSONResponse:
        """List models with available quantizations and result counts."""
        store = PPBDataStore.instance()
        await store.ensure_loaded()
        df = await store.get_df()
        rows: list[dict] = []
        if not df.empty and "model_base" in df.columns:
            quant_df = df[df["run_type"] != "qualitative"] if "run_type" in df.columns else df
            for model, grp in quant_df.groupby("model_base", dropna=True):
                quants = (
                    sorted(grp["quant"].dropna().unique().tolist())
                    if "quant" in grp.columns
                    else []
                )
                runner_types = (
                    sorted(grp["runner_type"].dropna().unique().tolist())
                    if "runner_type" in grp.columns
                    else []
                )
                model_org = None
                if "model_org" in grp.columns:
                    orgs = grp["model_org"].dropna()
                    if not orgs.empty:
                        model_org = str(orgs.iloc[0])
                rows.append(
                    {
                        "model": str(model),
                        "model_org": model_org,
                        "quantizations": quants,
                        "runner_types": runner_types,
                        "result_count": int(len(grp)),
                    }
                )
            rows.sort(key=lambda r: r["result_count"], reverse=True)
        return JSONResponse({"models": rows}, headers=_cors_headers(request))

    # ── /api/v1/results ──────────────────────────────────────────────────────

    @app.custom_route("/api/v1/results", methods=["GET"])
    async def api_results(request: Request) -> JSONResponse:
        """Query benchmark results with optional filters.

        Query params (all optional): gpu, model, quant, runner_type,
        concurrent_users (int, one of 1/2/4/8/16/32), vram_min (float ≥ 0),
        vram_max (float ≥ 0), unified_memory (true/false),
        run_after (ISO8601), run_before (ISO8601), limit (int 1–5000, default 100).
        """
        try:
            params = ResultsQueryParams(**dict(request.query_params))
        except ValidationError as exc:
            return JSONResponse(
                {"error": f"Invalid parameters: {exc}"},
                status_code=400,
                headers=_cors_headers(request),
            )
        # Allow a larger page when both gpu and model are specified — the
        # client fetches the full user slice in one shot for the Insights page.
        has_filter = bool(params.gpu and params.model)
        effective_limit = min(params.limit, 5000 if has_filter else 500)

        result = await query_ppb_results(
            gpu_name=params.gpu or None,
            vram_gb_min=params.vram_min,
            vram_gb_max=params.vram_max,
            model=params.model or None,
            quantization=params.quant or None,
            backend=None,
            runner_type=params.runner_type or None,
            concurrent_users=params.concurrent_users,
            run_after=params.run_after or None,
            run_before=params.run_before or None,
            unified_memory=params.unified_memory,
            limit=effective_limit,
        )
        return JSONResponse(_safe_dump(result.model_dump()), headers=_cors_headers(request))

    # ── /api/v1/qualitative ──────────────────────────────────────────────────

    @app.custom_route("/api/v1/qualitative", methods=["GET"])
    async def api_qualitative(request: Request) -> JSONResponse:
        """Query qualitative benchmark results with optional filters.

        Query params: model, quant (exact), gpu, runner_type,
        min_context_rot (0–1), min_tool_accuracy (0–1), min_mt_bench (0–1),
        limit (int 1–200, default 50).
        """
        try:
            params = QualitativeQueryParams(**dict(request.query_params))
        except ValidationError as exc:
            return JSONResponse(
                {"error": f"Invalid parameters: {exc}"},
                status_code=400,
                headers=_cors_headers(request),
            )
        result = await query_qualitative_results(
            model=params.model or None,
            quantization=params.quant or None,
            gpu_name=params.gpu or None,
            runner_type=params.runner_type or None,
            min_context_rot_score=params.min_context_rot,
            min_overall_tool_accuracy=params.min_tool_accuracy,
            min_mt_bench_score=params.min_mt_bench,
            limit=params.limit,
        )
        return JSONResponse(_safe_dump(result.model_dump()), headers=_cors_headers(request))

    # ── /api/v1/compare/quants ───────────────────────────────────────────────

    @app.custom_route("/api/v1/compare/quants", methods=["GET"])
    async def api_compare_quants(request: Request) -> JSONResponse:
        """Compare quantizations for a model across quantitative + qualitative metrics.

        Query params: model (required), gpu, runner_type,
        concurrent_users (int, one of 1/2/4/8/16/32).
        """
        try:
            params = CompareQueryParams(**dict(request.query_params))
        except ValidationError as exc:
            return JSONResponse(
                {"error": f"Invalid parameters: {exc}"},
                status_code=400,
                headers=_cors_headers(request),
            )
        quant_result = await compare_quants_quantitative(
            model=params.model,
            gpu_name=params.gpu or None,
            runner_type=params.runner_type or None,
            concurrent_users=params.concurrent_users,
        )
        return JSONResponse(_safe_dump(quant_result.model_dump()), headers=_cors_headers(request))

    # ── /api/v1/context-rot ──────────────────────────────────────────────────

    @app.custom_route("/api/v1/context-rot", methods=["GET"])
    async def api_context_rot(request: Request) -> JSONResponse:
        """Get context-rot breakdown for a model × quant × GPU.

        Query params: model (required), quant (required, exact), gpu.
        """
        try:
            params = ContextRotQueryParams(**dict(request.query_params))
        except ValidationError as exc:
            return JSONResponse(
                {"error": f"Invalid parameters: {exc}"},
                status_code=400,
                headers=_cors_headers(request),
            )
        result = await get_context_rot_breakdown(
            model=params.model,
            quantization=params.quant,
            gpu_name=params.gpu or None,
        )
        return JSONResponse(_safe_dump(result.model_dump()), headers=_cors_headers(request))

    # ── /api/v1/tool-accuracy ────────────────────────────────────────────────

    @app.custom_route("/api/v1/tool-accuracy", methods=["GET"])
    async def api_tool_accuracy(request: Request) -> JSONResponse:
        """Get tool-accuracy breakdown for a model × quant × GPU.

        Query params: model (required), quant (required, exact), gpu.
        """
        try:
            params = ToolAccuracyQueryParams(**dict(request.query_params))
        except ValidationError as exc:
            return JSONResponse(
                {"error": f"Invalid parameters: {exc}"},
                status_code=400,
                headers=_cors_headers(request),
            )
        result = await get_tool_accuracy_breakdown(
            model=params.model,
            quantization=params.quant,
            gpu_name=params.gpu or None,
        )
        return JSONResponse(_safe_dump(result.model_dump()), headers=_cors_headers(request))

    # ── /api/v1/docs ────────────────────────────────────────────────

    @app.custom_route("/api/v1/docs", methods=["GET"])
    async def api_docs(request: Request) -> JSONResponse:
        """Return documentation for all available REST endpoints."""
        rate_limit_env = os.environ.get("RATE_LIMIT", "60/minute")
        docs = {
            "version": __version__,
            "rate_limit": f"{rate_limit_env} per IP (applies to /api/ paths only)",
            "endpoints": [
                {
                    "path": "/health",
                    "method": "GET",
                    "description": "Server health check. Returns version, row counts, and last refresh time.",
                    "params": [],
                    "example": "/health",
                },
                {
                    "path": "/api/v1/summary",
                    "method": "GET",
                    "description": "List all tested GPUs, models, quantizations, and runner types.",
                    "params": [],
                    "example": "/api/v1/summary",
                },
                {
                    "path": "/api/v1/hardware",
                    "method": "GET",
                    "description": "List GPUs with VRAM capacity and result counts.",
                    "params": [],
                    "example": "/api/v1/hardware",
                },
                {
                    "path": "/api/v1/models",
                    "method": "GET",
                    "description": "List models with available quantizations and result counts.",
                    "params": [],
                    "example": "/api/v1/models",
                },
                {
                    "path": "/api/v1/results",
                    "method": "GET",
                    "description": (
                        "Query quantitative benchmark results with optional filters. "
                        "Providing both gpu and model raises the row cap to 5000."
                    ),
                    "schema": ResultsQueryParams.model_json_schema(),
                    "example": "/api/v1/results?gpu=RTX+4090&model=Qwen3.5-8B&limit=50",
                },
                {
                    "path": "/api/v1/qualitative",
                    "method": "GET",
                    "description": "Query qualitative results (context-rot, tool-accuracy, MT-Bench).",
                    "schema": QualitativeQueryParams.model_json_schema(),
                    "example": "/api/v1/qualitative?min_context_rot=0.7&limit=20",
                },
                {
                    "path": "/api/v1/compare/quants",
                    "method": "GET",
                    "description": "Compare quantizations for a model across quantitative and qualitative metrics.",
                    "schema": CompareQueryParams.model_json_schema(),
                    "example": "/api/v1/compare/quants?model=Qwen3.5-8B&gpu=RTX+4090",
                },
                {
                    "path": "/api/v1/context-rot",
                    "method": "GET",
                    "description": "Get context-rot score breakdown for a model × quant × GPU.",
                    "schema": ContextRotQueryParams.model_json_schema(),
                    "example": "/api/v1/context-rot?model=Qwen3.5-8B&quant=Q4_K_M",
                },
                {
                    "path": "/api/v1/tool-accuracy",
                    "method": "GET",
                    "description": "Get tool-accuracy breakdown for a model × quant × GPU.",
                    "schema": ToolAccuracyQueryParams.model_json_schema(),
                    "example": "/api/v1/tool-accuracy?model=Qwen3.5-8B&quant=Q4_K_M",
                },
                {
                    "path": "/api/v1/docs",
                    "method": "GET",
                    "description": "This endpoint. Returns documentation for all REST endpoints.",
                    "params": [],
                    "example": "/api/v1/docs",
                },
            ],
        }
        return JSONResponse(docs, headers=_cors_headers(request))

except ImportError:
    # starlette is pulled in by fastmcp; if it's missing, REST endpoints are unavailable but stdio still works.
    pass


def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    port = int(os.environ.get("PORT", "9933"))
    host = os.environ.get("HOST", "0.0.0.0")
    _configure_logging(transport)

    if transport == "stdio":
        app.run(transport="stdio")
    else:
        max_requests, window_seconds = _parse_rate_limit()
        app.run(
            transport="streamable-http",
            host=host,
            port=port,
            middleware=[
                Middleware(_AcceptPatchMiddleware),
                Middleware(
                    _RateLimitMiddleware,
                    max_requests=max_requests,
                    window_seconds=window_seconds,
                ),
            ],
        )


if __name__ == "__main__":
    main()
