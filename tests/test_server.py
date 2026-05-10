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


# ── Input validation tests ────────────────────────────────────────────────────


def test_results_invalid_concurrent_users(store) -> None:
    """concurrent_users=3 is not in {1,2,4,8,16,32} — should return 400."""
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/results?concurrent_users=3")
    assert r.status_code == 400
    data = r.json()
    assert "error" in data
    assert "concurrent_users" in data["error"]


def test_results_valid_concurrent_users(store) -> None:
    """concurrent_users=4 is valid — should return 200."""
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/results?concurrent_users=4")
    assert r.status_code == 200


def test_results_invalid_limit_too_large(store) -> None:
    """limit=9999 exceeds schema max of 5000 — should return 400."""
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/results?limit=9999")
    assert r.status_code == 400
    assert "error" in r.json()


def test_results_invalid_limit_zero(store) -> None:
    """limit=0 is below schema min of 1 — should return 400."""
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/results?limit=0")
    assert r.status_code == 400
    assert "error" in r.json()


def test_results_invalid_vram_negative(store) -> None:
    """vram_min=-1 is negative — should return 400."""
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/results?vram_min=-1")
    assert r.status_code == 400
    assert "error" in r.json()


def test_qualitative_invalid_score(store) -> None:
    """min_context_rot=1.5 exceeds max of 1.0 — should return 400."""
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/qualitative?min_context_rot=1.5")
    assert r.status_code == 400
    assert "error" in r.json()


def test_compare_quants_invalid_concurrent_users(store) -> None:
    """concurrent_users=5 is not valid — should return 400."""
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/compare/quants?model=Llama-7B&concurrent_users=5")
    assert r.status_code == 400
    assert "error" in r.json()


# ── /api/v1/docs tests ───────────────────────────────────────────────────────


def test_docs_endpoint(store) -> None:
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/docs")
    assert r.status_code == 200
    data = r.json()
    assert "endpoints" in data
    assert "version" in data
    assert "rate_limit" in data
    paths = {e["path"] for e in data["endpoints"]}
    assert "/api/v1/results" in paths
    assert "/api/v1/docs" in paths
    assert "/api/v1/qualitative" in paths


def test_docs_results_schema_present(store) -> None:
    """The /api/v1/results entry should include a Pydantic-generated JSON schema."""
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/docs")
    assert r.status_code == 200
    results_entry = next(e for e in r.json()["endpoints"] if e["path"] == "/api/v1/results")
    assert "schema" in results_entry
    assert "properties" in results_entry["schema"]


# ── CORS expansion tests ──────────────────────────────────────────────────────


def test_cors_mcp_subdomain(store) -> None:
    """mcp.poorpaul.dev should now receive CORS headers."""
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/summary", headers={"Origin": "https://mcp.poorpaul.dev"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://mcp.poorpaul.dev"


def test_cors_localhost_allowed(store) -> None:
    """Any localhost origin should receive CORS headers (existing behaviour preserved)."""
    from starlette.testclient import TestClient

    from ppb_mcp.server import app

    with TestClient(app.http_app()) as client:
        r = client.get("/api/v1/summary", headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ── Rate limiter unit tests ───────────────────────────────────────────────────


def test_parse_rate_limit_minute() -> None:
    import os

    from ppb_mcp.server import _parse_rate_limit

    os.environ["RATE_LIMIT"] = "60/minute"
    assert _parse_rate_limit() == (60, 60)


def test_parse_rate_limit_hour() -> None:
    import os

    from ppb_mcp.server import _parse_rate_limit

    os.environ["RATE_LIMIT"] = "200/hour"
    assert _parse_rate_limit() == (200, 3600)


def test_parse_rate_limit_bad_value() -> None:
    import os

    from ppb_mcp.server import _parse_rate_limit

    os.environ["RATE_LIMIT"] = "not-a-number"
    result = _parse_rate_limit()
    # Falls back to default (60, 60)
    assert result == (60, 60)
