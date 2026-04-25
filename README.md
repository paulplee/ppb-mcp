# ppb-mcp

> An MCP server that exposes [Poor Paul's Benchmark](https://huggingface.co/datasets/paulplee/ppb-results) GPU inference data — quantization × throughput × VRAM × concurrent users — as queryable tools to any LLM client.

[![CI](https://github.com/paulplee/ppb-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/paulplee/ppb-mcp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ppb-mcp.svg)](https://pypi.org/project/ppb-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hosted instance:** `https://mcp.poorpaul.dev/` (streamable-http transport, no auth)

## What it does

Connect any MCP-aware client (Claude Desktop, Cline, Continue, etc.) to ask questions like:

- *"What's the best quantization for a 32 GB GPU running Qwen3.5-9B with 8 concurrent users?"*
- *"Show me every model tested at Q4_K_M on the RTX 5090."*
- *"Will Llama-13B at Q5_K_M fit on a 24 GB GPU at 4 concurrent users?"*

It exposes **four tools** backed by 30,000+ real benchmark rows:

| Tool | What it does |
| --- | --- |
| `list_tested_configs` | Lists every tested GPU, model, and quantization (call this first) |
| `query_ppb_results` | Filters raw benchmark rows by GPU / VRAM / model / quant / users / backend |
| `recommend_quantization` | Three-tier empirical-first recommendation engine (high / medium / low confidence) |
| `get_gpu_headroom` | Sanity-checks a (gpu, model, quant, users) configuration for VRAM headroom |

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
docker run --rm -p 8000:8000 \
  -e MCP_TRANSPORT=streamable-http \
  -v ppb-hf-cache:/data/huggingface \
  ghcr.io/paulplee/ppb-mcp:latest
```

### 4) From source

```bash
git clone https://github.com/paulplee/ppb-mcp
cd ppb-mcp
pip install -e ".[dev]"
ppb-mcp           # streamable-http on :8000
```

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

| Env var | Default | Notes |
| --- | --- | --- |
| `HF_DATASET` | `paulplee/ppb-results` | HuggingFace dataset ID |
| `REFRESH_INTERVAL_HOURS` | `1` | Background refresh cadence |
| `MCP_TRANSPORT` | `streamable-http` | `stdio` or `streamable-http` |
| `HOST` | `0.0.0.0` | HTTP bind host |
| `PORT` | `8000` | HTTP bind port |
| `LOG_LEVEL` | `INFO` | Python `logging` level |

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
