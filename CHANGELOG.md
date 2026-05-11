# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-10

### Added
- MCP server with 17 tools covering quantitative benchmarks, qualitative results, VRAM headroom, hardware recommendations, context-rot breakdown, and tool-accuracy scores
- REST API at `/api/v1/` with OpenAPI docs at `/api/v1/docs`
- Pydantic input validation on all REST endpoints with structured 400 error responses
- Rate limiting middleware (configurable via `RATE_LIMIT` env var, default 60/minute)
- CORS support for browser-based MCP clients
- `_AcceptPatchMiddleware` for compatibility with MCP clients that omit `application/json` in Accept headers
- `HF_DATASET` env var to point at custom dataset forks
- `PPB_DB_PATH` env var for custom SQLite cache location
- Streamable-HTTP and stdio transports selectable via `MCP_TRANSPORT`
- Safe JSON serialisation: NaN/Inf values serialised as `null`
- Lifespan dataset pre-load with configurable `REFRESH_INTERVAL_HOURS`
