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
from ppb_mcp.tools.headroom import get_gpu_headroom
from ppb_mcp.tools.list_configs import list_tested_configs
from ppb_mcp.tools.query import query_ppb_results
from ppb_mcp.tools.recommend import recommend_quantization

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
        "Use recommend_quantization to find the best quantization for your GPU "
        "and concurrent user count. Data source: https://huggingface.co/datasets/paulplee/ppb-results"
    ),
    version=__version__,
    lifespan=_lifespan,
)

# Register the four tools.
app.tool(list_tested_configs)
app.tool(query_ppb_results)
app.tool(recommend_quantization)
app.tool(get_gpu_headroom)


# /health endpoint for Docker healthcheck and Lightsail monitoring (HTTP transport only).
try:
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @app.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> JSONResponse:
        store = PPBDataStore.instance()
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "dataset": store.dataset,
                "dataset_rows": store.row_count(),
                "last_refreshed": store.get_last_refreshed(),
            }
        )
except ImportError:
    # starlette is pulled in by fastmcp; if it's missing, /health is unavailable but stdio still works.
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
