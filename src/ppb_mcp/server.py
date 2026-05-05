"""FastMCP server entrypoint for ppb-mcp."""

from __future__ import annotations

import logging
import os
import sys
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
            if "text/event-stream" not in accept:
                new_accept = (accept + ", text/event-stream").lstrip(", ")
                new_headers = [(k, v) for k, v in scope["headers"] if k != b"accept"]
                new_headers.append((b"accept", new_accept.encode("latin-1")))
                scope = dict(scope)
                scope["headers"] = new_headers
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
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    # ── Allowed origins for CORS ─────────────────────────────────────────────
    _CORS_ORIGINS = [
        "https://poorpaul.dev",
        "https://www.poorpaul.dev",
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
        return JSONResponse(result.model_dump(), headers=_cors_headers(request))

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
                rows.append(
                    {
                        "model": str(model),
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
        concurrent_users (int), vram_min (float), vram_max (float),
        unified_memory (true/false), run_after (ISO8601), run_before (ISO8601),
        limit (int, max 500, default 100).
        """
        p = request.query_params
        concurrent_users: int | None = None
        if p.get("concurrent_users"):
            try:
                concurrent_users = int(p["concurrent_users"])
            except ValueError:
                pass
        vram_gb_min: float | None = None
        if p.get("vram_min"):
            try:
                vram_gb_min = float(p["vram_min"])
            except ValueError:
                pass
        vram_gb_max: float | None = None
        if p.get("vram_max"):
            try:
                vram_gb_max = float(p["vram_max"])
            except ValueError:
                pass
        unified_memory: bool | None = None
        if p.get("unified_memory"):
            unified_memory = p["unified_memory"].lower() in ("true", "1", "yes")
        # Allow a larger page when both gpu and model are specified — the
        # client fetches the full user slice in one shot for the Insights page.
        has_filter = bool(p.get("gpu") and p.get("model"))
        max_limit = 5000 if has_filter else 500
        limit = min(int(p.get("limit", 100)), max_limit)

        result = await query_ppb_results(
            gpu_name=p.get("gpu") or None,
            vram_gb_min=vram_gb_min,
            vram_gb_max=vram_gb_max,
            model=p.get("model") or None,
            quantization=p.get("quant") or None,
            backend=None,
            runner_type=p.get("runner_type") or None,
            concurrent_users=concurrent_users,
            run_after=p.get("run_after") or None,
            run_before=p.get("run_before") or None,
            unified_memory=unified_memory,
            limit=limit,
        )
        return JSONResponse(result.model_dump(), headers=_cors_headers(request))

    # ── /api/v1/qualitative ──────────────────────────────────────────────────

    @app.custom_route("/api/v1/qualitative", methods=["GET"])
    async def api_qualitative(request: Request) -> JSONResponse:
        """Query qualitative benchmark results with optional filters.

        Query params: model, quant (exact), gpu, runner_type,
        min_context_rot, min_tool_accuracy, min_mt_bench,
        limit (int, max 200, default 50).
        """
        p = request.query_params
        min_context_rot: float | None = None
        if p.get("min_context_rot"):
            try:
                min_context_rot = float(p["min_context_rot"])
            except ValueError:
                pass
        min_tool_accuracy: float | None = None
        if p.get("min_tool_accuracy"):
            try:
                min_tool_accuracy = float(p["min_tool_accuracy"])
            except ValueError:
                pass
        min_mt_bench: float | None = None
        if p.get("min_mt_bench"):
            try:
                min_mt_bench = float(p["min_mt_bench"])
            except ValueError:
                pass
        limit = min(int(p.get("limit", 50)), 200)

        result = await query_qualitative_results(
            model=p.get("model") or None,
            quantization=p.get("quant") or None,
            gpu_name=p.get("gpu") or None,
            runner_type=p.get("runner_type") or None,
            min_context_rot_score=min_context_rot,
            min_overall_tool_accuracy=min_tool_accuracy,
            min_mt_bench_score=min_mt_bench,
            limit=limit,
        )
        return JSONResponse(result.model_dump(), headers=_cors_headers(request))

    # ── /api/v1/compare/quants ───────────────────────────────────────────────

    @app.custom_route("/api/v1/compare/quants", methods=["GET"])
    async def api_compare_quants(request: Request) -> JSONResponse:
        """Compare quantizations for a model across quantitative + qualitative metrics.

        Query params: model (required), gpu, runner_type, concurrent_users (int).
        """
        p = request.query_params
        model = p.get("model") or ""
        if not model:
            return JSONResponse(
                {"error": "model parameter is required"},
                status_code=400,
                headers=_cors_headers(request),
            )
        concurrent_users: int | None = None
        if p.get("concurrent_users"):
            try:
                concurrent_users = int(p["concurrent_users"])
            except ValueError:
                pass

        quant_result = await compare_quants_quantitative(
            model=model,
            gpu_name=p.get("gpu") or None,
            runner_type=p.get("runner_type") or None,
            concurrent_users=concurrent_users,
        )
        return JSONResponse(quant_result.model_dump(), headers=_cors_headers(request))

    # ── /api/v1/context-rot ──────────────────────────────────────────────────

    @app.custom_route("/api/v1/context-rot", methods=["GET"])
    async def api_context_rot(request: Request) -> JSONResponse:
        """Get context-rot breakdown for a model × quant × GPU.

        Query params: model (required), quant (required, exact), gpu.
        """
        p = request.query_params
        model = p.get("model") or ""
        quant = p.get("quant") or ""
        if not model or not quant:
            return JSONResponse(
                {"error": "model and quant parameters are required"},
                status_code=400,
                headers=_cors_headers(request),
            )
        result = await get_context_rot_breakdown(
            model=model,
            quantization=quant,
            gpu_name=p.get("gpu") or None,
        )
        return JSONResponse(result.model_dump(), headers=_cors_headers(request))

    # ── /api/v1/tool-accuracy ────────────────────────────────────────────────

    @app.custom_route("/api/v1/tool-accuracy", methods=["GET"])
    async def api_tool_accuracy(request: Request) -> JSONResponse:
        """Get tool-accuracy breakdown for a model × quant × GPU.

        Query params: model (required), quant (required, exact), gpu.
        """
        p = request.query_params
        model = p.get("model") or ""
        quant = p.get("quant") or ""
        if not model or not quant:
            return JSONResponse(
                {"error": "model and quant parameters are required"},
                status_code=400,
                headers=_cors_headers(request),
            )
        result = await get_tool_accuracy_breakdown(
            model=model,
            quantization=quant,
            gpu_name=p.get("gpu") or None,
        )
        return JSONResponse(result.model_dump(), headers=_cors_headers(request))

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
        app.run(
            transport="streamable-http",
            host=host,
            port=port,
            middleware=[Middleware(_AcceptPatchMiddleware)],
        )


if __name__ == "__main__":
    main()
