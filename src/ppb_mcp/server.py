"""FastMCP server entrypoint for ppb-mcp."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

import anyio
from fastmcp import FastMCP

from ppb_mcp import __version__
from ppb_mcp.data import PPBDataStore
from ppb_mcp.tools.headroom import get_gpu_headroom
from ppb_mcp.tools.list_configs import list_tested_configs
from ppb_mcp.tools.query import query_ppb_results
from ppb_mcp.tools.recommend import recommend_quantization

logger = logging.getLogger("ppb_mcp")


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
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    _configure_logging(transport)

    if transport == "stdio":
        app.run(transport="stdio")
    else:
        app.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
