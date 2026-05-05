"""Tests for the FastMCP server registration."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_four_tools_registered() -> None:
    from ppb_mcp.server import app

    tools = await app.list_tools()
    names = {t.name for t in tools}
    assert {
        "list_tested_configs",
        "query_ppb_results",
        "recommend_quantization",
        "get_gpu_headroom",
    } <= names


@pytest.mark.asyncio
async def test_tool_schemas_have_descriptions() -> None:
    from ppb_mcp.server import app

    tools = await app.list_tools()
    by_name = {t.name: t for t in tools}
    for name in (
        "list_tested_configs",
        "query_ppb_results",
        "recommend_quantization",
        "get_gpu_headroom",
    ):
        assert by_name[name].description, f"{name} is missing a description"


# ── REST API endpoint tests ───────────────────────────────────────────────────


def test_rest_health(store) -> None:  # noqa: ANN001 - fixture
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "dataset_rows" in data
    assert "last_refreshed" in data


def test_rest_summary(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/summary")
    assert r.status_code == 200
    data = r.json()
    assert "gpus" in data
    assert "models" in data
    assert "quantizations" in data
    assert "total_benchmark_rows" in data
    assert len(data["gpus"]) > 0


def test_rest_hardware(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/hardware")
    assert r.status_code == 200
    data = r.json()
    assert "hardware" in data
    assert len(data["hardware"]) > 0
    first = data["hardware"][0]
    assert "gpu_name" in first
    assert "result_count" in first


def test_rest_models(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert len(data["models"]) > 0
    first = data["models"][0]
    assert "model" in first
    assert "quantizations" in first
    assert "result_count" in first


def test_rest_results(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/results")
    assert r.status_code == 200
    data = r.json()
    assert "rows" in data
    assert "total_count" in data
    assert "filtered_count" in data
    assert len(data["rows"]) > 0


def test_rest_results_with_gpu_filter(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/results?gpu=16GB")
    assert r.status_code == 200
    data = r.json()
    assert all("16GB" in row["gpu_name"] for row in data["rows"])


def test_rest_compare_quants_requires_model(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/compare/quants")
    assert r.status_code == 400


def test_rest_compare_quants(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/compare/quants?model=Llama-7B")
    assert r.status_code == 200
    data = r.json()
    assert "rows" in data
    assert "insight" in data


def test_rest_context_rot_requires_params(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/context-rot?model=Llama-7B")
    assert r.status_code == 400


def test_rest_tool_accuracy_requires_params(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/tool-accuracy?model=Llama-7B")
    assert r.status_code == 400


def test_rest_cors_header_for_poorpaul(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/summary", headers={"Origin": "https://poorpaul.dev"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://poorpaul.dev"


def test_rest_no_cors_for_unknown_origin(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/summary", headers={"Origin": "https://evil.example.com"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers
