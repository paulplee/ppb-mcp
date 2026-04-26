# ppb-mcp

> An MCP server that exposes [Poor Paul's Benchmark](https://huggingface.co/datasets/paulplee/ppb-results) GPU inference data — quantization × throughput × VRAM × concurrent users — as queryable tools to any LLM client.

[![CI](https://github.com/paulplee/ppb-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/paulplee/ppb-mcp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ppb-mcp.svg)](https://pypi.org/project/ppb-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hosted instance:** `https://mcp.poorpaul.dev/` (streamable-http transport, no auth)

## What it does

Connect any MCP-aware client (Claude Desktop, Cline, Continue, etc.) to ask questions like:

- _"What's the best quantization for a 32 GB GPU running Qwen3.5-9B with 8 concurrent users?"_
- _"Show me every model tested at Q4_K_M on the RTX 5090."_
- _"Will Llama-13B at Q5_K_M fit on a 24 GB GPU at 4 concurrent users?"_

It exposes **nine tools** backed by 30,000+ real benchmark rows:

### Quantitative tools

| Tool                     | What it does                                                                      |
| ------------------------ | --------------------------------------------------------------------------------- |
| `list_tested_configs`    | Lists every tested GPU, model, and quantization (call this first)                 |
| `query_ppb_results`      | Filters raw benchmark rows by GPU / VRAM / model / quant / users / backend        |
| `recommend_quantization` | Three-tier empirical-first recommendation engine (high / medium / low confidence) |
| `get_gpu_headroom`       | Sanity-checks a (gpu, model, quant, users) configuration for VRAM headroom        |

### Qualitative tools

| Tool                          | What it does                                                                        |
| ----------------------------- | ----------------------------------------------------------------------------------- |
| `get_qualitative_summary`     | All available qualitative scores (context-rot, tool accuracy, quality, MT-Bench)    |
| `query_qualitative_results`   | Filter qualitative rows by phase, model, quant, GPU, or minimum score thresholds    |
| `get_context_rot_breakdown`   | Long-context recall scores by length, depth, and needle type                        |
| `get_tool_accuracy_breakdown` | Tool-call accuracy: selection, parameters, hallucination rate, parse success        |
| `compare_quants_qualitative`  | Side-by-side qualitative comparison across quantizations with deterministic insight |

### Data & caching

Benchmark rows are mirrored into a local SQLite cache (`./ppb_cache.db` by
default; override with `PPB_DB_PATH`). On startup the server loads from
SQLite and only contacts HuggingFace when the dataset's git commit SHA
has changed — making subsequent restarts fast and offline-friendly.

## Install

### 1) Use the hosted instance (zero setup)

Add to your MCP client config (Claude Desktop example, `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ppb": {
      "transport": { "type": "http", "url": "https://mcp.poorpaul.dev/mcp" }
    }
  }
}
```

### 2) `pip install` and run locally (stdio)

```bash
pip install ppb-mcp
MCP_TRANSPORT=stdio ppb-mcp
```

Claude Desktop config:

```json
{
  "mcpServers": {
    "ppb": {
      "command": "ppb-mcp",
      "env": { "MCP_TRANSPORT": "stdio" }
    }
  }
}
```

### 3) Docker

```bash
docker run --rm -p 9933:9933 \
  -e MCP_TRANSPORT=streamable-http \
  -v ppb-hf-cache:/data/huggingface \
  ghcr.io/paulplee/ppb-mcp:latest
```

### 4) From source

```bash
git clone https://github.com/paulplee/ppb-mcp
cd ppb-mcp
pip install -e ".[dev]"
ppb-mcp           # streamable-http on :9933
```

## Connect Your LLM Client

All clients use the same hosted endpoint: `https://mcp.poorpaul.dev/mcp`

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "ppb": {
      "transport": { "type": "http", "url": "https://mcp.poorpaul.dev/mcp" }
    }
  }
}
```

Restart Claude Desktop after saving.

### Cursor

Edit `~/.cursor/mcp.json` (create if it doesn't exist):

```json
{
  "mcpServers": {
    "ppb": {
      "url": "https://mcp.poorpaul.dev/mcp",
      "type": "http"
    }
  }
}
```

Or via UI: **Settings → Tools & Integrations → MCP → Add Server**.

### Windsurf

Edit `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "ppb": {
      "serverUrl": "https://mcp.poorpaul.dev/mcp",
      "transport": "http"
    }
  }
}
```

### VS Code (GitHub Copilot Agent Mode)

Add to your `.vscode/mcp.json` (workspace) or User `settings.json`:

```json
{
  "mcp": {
    "servers": {
      "ppb": {
        "type": "http",
        "url": "https://mcp.poorpaul.dev/mcp"
      }
    }
  }
}
```

### Zed

Add to `~/.config/zed/settings.json` under `"context_servers"`:

```json
{
  "context_servers": {
    "ppb": {
      "command": {
        "path": "env",
        "args": ["MCP_TRANSPORT=stdio", "uvx", "ppb-mcp"]
      }
    }
  }
}
```

### Cline (VS Code extension)

Open the Cline panel → **MCP Servers** tab → **Add Server** → select **SSE/HTTP** → paste `https://mcp.poorpaul.dev/mcp`.

### Continue.dev

Add to `~/.continue/config.yaml`:

```yaml
mcpServers:
  - name: ppb
    transport:
      type: http
      url: https://mcp.poorpaul.dev/mcp
```

### OpenCode

Add to `~/.config/opencode/config.json`:

```json
{
  "mcp": {
    "ppb": {
      "type": "remote",
      "url": "https://mcp.poorpaul.dev/mcp"
    }
  }
}
```

### Goose (Block)

```bash
goose mcp add ppb --transport http --url https://mcp.poorpaul.dev/mcp
```

### Any stdio-compatible client

```bash
# Zero-install (requires uv):
env MCP_TRANSPORT=stdio uvx ppb-mcp

# After pip install:
env MCP_TRANSPORT=stdio ppb-mcp
```

> **Note on transport key names**: MCP clients are not yet fully standardised on JSON key names for the HTTP transport. If your client doesn't connect with `"type": "http"`, try `"transport": "http"`, `"type": "sse"`, or `"transport": "streamable-http"`. The endpoint URL is the same regardless.

## Example session

```text
> list_tested_configs
{ "gpus": ["Apple M4 Pro", "NVIDIA GB10", "NVIDIA GeForce RTX 5090"],
  "models": ["Qwen3.5-9B", ...], "quantizations": ["Q4_K_M", ...] }

> recommend_quantization(gpu_vram_gb=32, concurrent_users=8, model="Qwen3.5-9B", priority="balance")
{ "recommended_quantization": "Q5_K_M",
  "estimated_vram_usage_gb": 27.8,
  "estimated_tokens_per_second": 142.0,
  "headroom_gb": 4.2,
  "confidence": "high",
  "reasoning": "Q5_K_M is recommended for your NVIDIA GeForce RTX 5090 (32 GB) ...",
  "alternatives": ["Q4_K_M", "Q8_0"] }
```

## Configuration

| Env var                  | Default                | Notes                        |
| ------------------------ | ---------------------- | ---------------------------- |
| `HF_DATASET`             | `paulplee/ppb-results` | HuggingFace dataset ID       |
| `REFRESH_INTERVAL_HOURS` | `1`                    | Background refresh cadence   |
| `MCP_TRANSPORT`          | `streamable-http`      | `stdio` or `streamable-http` |
| `HOST`                   | `0.0.0.0`              | HTTP bind host               |
| `PORT`                   | `9933`                 | HTTP bind port               |
| `LOG_LEVEL`              | `INFO`                 | Python `logging` level       |

## Self-hosting (Lightsail / any Ubuntu VPS)

```bash
git clone https://github.com/paulplee/ppb-mcp /tmp/ppb-mcp
cd /tmp/ppb-mcp
DOMAIN=mcp.example.com EMAIL=you@example.com ./deploy/deploy.sh
```

This installs Docker, builds the image, registers a systemd unit, configures nginx, and runs certbot.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -v
```

Integration tests against the live HuggingFace dataset are gated behind `PPB_RUN_INTEGRATION=1` to keep CI offline-clean.

## How recommendations work

1. **Tier 1 — empirical exact match (high confidence).** ≥3 measured runs on a GPU at-or-below your VRAM budget at the requested concurrency.
2. **Tier 2 — empirical-near (medium).** Same `(model, quant)` benchmarked on a different GPU at the same concurrency; throughput borrowed, VRAM scaled to your card.
3. **Tier 3 — formula extrapolation (low).** `vram_per_user ≈ (params_B × bits_per_weight / 8) × 1.15`; viable iff total ≤ 90 % of your VRAM.

## License

MIT — see [LICENSE](LICENSE).
