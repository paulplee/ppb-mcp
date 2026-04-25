# ppb-mcp Implementation Prompt

### For: GitHub Copilot with Claude Opus 4.7

### Project: Poor Paul's MCP Server (`ppb-mcp`)

---

## How to Use This Prompt

Paste this document in full as your first message to Copilot. Opus 4.7's extended thinking works best when given the complete picture upfront. Do **not** break this into smaller prompts — the interdependencies between the components (schema, tools, packaging, tests, deployment) require the model to hold the full spec in context.

After pasting, say: **"Begin with Step 1. Confirm your understanding of the project structure before writing any code."**

This forces Opus to think through the architecture before generating, which produces far fewer revisions.

---

## SYSTEM CONTEXT

You are a senior Python engineer building `ppb-mcp`, a production-ready, open-source MCP (Model Context Protocol) server. This server gives LLM clients structured, queryable access to Poor Paul's Benchmark data — empirical GPU inference performance results for consumer and prosumer hardware stored on HuggingFace.

The primary use case is: a user asks an LLM "which quantization should I run on my RTX 3090 for 2 concurrent users?" and the LLM calls `ppb-mcp` tools to answer with real benchmark data rather than guessing.

You are building this to production quality: tested, packaged for PyPI, Dockerized, deployable as a systemd service behind nginx, and documented for both zero-setup hosted use and self-hosting. The code will be open-sourced and is a brand asset — quality and clarity matter as much as correctness.

---

## PROJECT OVERVIEW

**Repository name**: `ppb-mcp`  
**PyPI package name**: `ppb-mcp`  
**Docker image**: `paulplee/ppb-mcp`  
**Hosted endpoint** (production, managed separately): `https://mcp.poorpaul.dev`  
**Data source**: HuggingFace dataset `paulplee/ppb-results` (public, no auth required)  
**Framework**: FastMCP (Python) — use the `fastmcp` PyPI package  
**Transport**: Streamable HTTP (SSE) for the hosted/remote deployment; stdio transport for local `uvx` usage  
**Python version**: 3.11+

---

## STEP 1 — INSPECT THE DATA FIRST

Before writing any tool logic, fetch and inspect the HuggingFace dataset to understand its exact schema.

> ⚠️ **Known issue (verified):** `load_dataset("paulplee/ppb-results")` **fails** with `TypeError: Couldn't cast array of type string to null` because some columns are all-null in early shards but typed in later ones, breaking pyarrow schema unification across the 260 JSONL shards. Use the raw-JSONL loader path described in Step 5 instead. Do **not** add the `datasets` package as a dependency.

Use this script (via `huggingface_hub`) to inspect the schema:

```python
import json
from huggingface_hub import HfApi, hf_hub_download
import pandas as pd

api = HfApi()
files = [f for f in api.list_repo_files("paulplee/ppb-results", repo_type="dataset") if f.endswith(".jsonl")]
rows = []
for f in files:
    p = hf_hub_download("paulplee/ppb-results", f, repo_type="dataset")
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
df = pd.DataFrame(rows)
print(df.shape, df.dtypes, df.head(3))
```

### Verified schema (as of 30,841 rows × 61 columns, 260 shards)

**Key columns** (these are the ground truth — do not assume names from the spec):

| Concept                      | Real column                                                                                                     | Type    | Notes                                                     |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------- |
| Tokens/sec (generation)      | `throughput_tok_s`                                                                                              | float64 | No nulls. Primary speed metric.                           |
| GPU name                     | `gpu_name`                                                                                                      | str     | 3 unique values today                                     |
| GPU VRAM                     | `gpu_vram_gb`                                                                                                   | float64 | Has 7,265 nulls; fall back to `gpu_total_vram_gb`         |
| Total VRAM (multi-GPU aware) | `gpu_total_vram_gb`                                                                                             | float64 | Use as canonical for forward-compat                       |
| Clean model name             | `model_base`                                                                                                    | str     | 34 unique, e.g. `Qwen3.5-9B`, `gpt-oss-20b`               |
| Full model path              | `model`                                                                                                         | str     | Raw GGUF path; expose as `model_full_path` in API         |
| Model org                    | `model_org`                                                                                                     | str     | `unsloth`, `mudler`, `Jackrong`                           |
| Quantization                 | `quant`                                                                                                         | str     | 32 unique, e.g. `Q4_K_M`, `IQ4_XS`, `Q4_K_XL`. 650 nulls. |
| Concurrent users             | `concurrent_users`                                                                                              | float64 | **Real column.** Values: 1, 2, 4, 8, 16, 32               |
| Backend                      | `backends`                                                                                                      | str     | `CUDA 13.0`, `CUDA 13.1`, `CUDA`, `Metal`, `Metal 4`      |
| Latency                      | `avg_ttft_ms`, `p50_ttft_ms`, `p99_ttft_ms`, `avg_itl_ms`, `p50_itl_ms`, `p99_itl_ms`                           | float64 | All populated                                             |
| Power                        | `avg_power_w`, `max_power_w`                                                                                    | float64 | Has nulls                                                 |
| Context                      | `n_ctx`, `n_batch`                                                                                              | int64   |                                                           |
| Provenance                   | `submitter`, `timestamp`, `submitted_at`, `schema_version`, `benchmark_version`                                 | str     |                                                           |
| Run identity                 | `row_id`, `submission_id`, `run_fingerprint`, `result_fingerprint`, `machine_fingerprint`, `source_file_sha256` | str     |                                                           |

**Reserved-but-currently-all-null** (whitelist these in schema validation; do not warn): `llm_engine_version`, `split_mode`, `tensor_split`, `quality_score`, `tags`.

**There is no `vram_usage_gb` column.** VRAM-per-request must be derived (see Step 7).

**GPUs in the dataset today** (test fixtures may mock 8/16/24 GB GPUs, but real data has only these):

| GPU                       | VRAM     | Rows   |
| ------------------------- | -------- | ------ |
| `NVIDIA GeForce RTX 5090` | 31.8 GB  | 15,892 |
| `Apple M4 Pro`            | 64.0 GB  | 6,359  |
| `NVIDIA GB10`             | 119.6 GB | 1,325  |

**Document the verified schema as a comment block at the top of `src/ppb_mcp/data.py`.** Every downstream tool depends on this schema — do not assume column names.

---

## STEP 2 — REPOSITORY STRUCTURE

Create exactly this layout. Do not deviate.

```
ppb-mcp/
├── src/
│   └── ppb_mcp/
│       ├── __init__.py           # version = "0.1.0", exports app
│       ├── server.py             # FastMCP app definition + tool registration
│       ├── data.py               # HuggingFace loading, caching, refresh logic
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── _vram.py          # shared VRAM-estimation helpers (param parser + bpw table)
│       │   ├── list_configs.py   # list_tested_configs tool
│       │   ├── query.py          # query_ppb_results tool
│       │   ├── recommend.py      # recommend_quantization tool (core logic)
│       │   └── headroom.py       # get_gpu_headroom tool
│       └── models.py             # Pydantic response models
├── tests/
│   ├── conftest.py               # pytest fixtures, sample DataFrame
│   ├── test_data.py              # data loading and refresh logic
│   ├── test_tools.py             # one test class per tool, happy path + edge cases
│   └── test_server.py            # MCP tool registration and schema tests
├── deploy/
│   ├── ppb-mcp.service           # systemd unit file
│   ├── nginx-ppb-mcp.conf        # nginx server block
│   └── deploy.sh                 # one-shot Lightsail deploy script
├── .github/
│   └── workflows/
│       ├── ci.yml                # run tests on PR
│       └── publish.yml           # publish to PyPI on version tag
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
└── .env.example
```

---

## STEP 3 — DEPENDENCIES AND PACKAGING (`pyproject.toml`)

Use `pyproject.toml` with Hatch as the build backend. This enables `uvx ppb-mcp` to work directly.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ppb-mcp"
version = "0.1.0"
description = "Poor Paul's MCP Server — queryable GPU inference benchmarks for LLM clients"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "Paul Lee", email = "paul@poorpaul.dev" }]
keywords = ["mcp", "llm", "gpu", "benchmark", "quantization", "local-llm", "model-context-protocol"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
]
requires-python = ">=3.11"
dependencies = [
    "fastmcp>=2.0",
    "huggingface_hub>=0.20",
    "pandas>=2.2",
    "pydantic>=2.0",
    "anyio>=4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
    "httpx>=0.27",  # used by docker healthcheck and integration tests
]

[project.scripts]
ppb-mcp = "ppb_mcp.server:main"

[project.urls]
Homepage = "https://poorpaul.dev"
Repository = "https://github.com/paulplee/ppb-mcp"
"Bug Tracker" = "https://github.com/paulplee/ppb-mcp/issues"
"Data Source" = "https://huggingface.co/datasets/paulplee/ppb-results"

[tool.hatch.build.targets.wheel]
packages = ["src/ppb_mcp"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
```

---

## STEP 4 — ENVIRONMENT VARIABLES

All configuration is via environment variables with sensible defaults. Document each in `.env.example`.

```bash
# .env.example

# HuggingFace dataset identifier (override for custom datasets or forks)
HF_DATASET=paulplee/ppb-results

# How often to refresh the in-memory dataset cache (hours)
REFRESH_INTERVAL_HOURS=1

# Server port (used in Streamable HTTP mode)
PORT=8000

# Server host
HOST=0.0.0.0

# Transport mode: "streamable-http" (remote/hosted) or "stdio" (local/uvx)
# When running via `uvx ppb-mcp`, this should be "stdio"
MCP_TRANSPORT=streamable-http

# Log level
LOG_LEVEL=INFO
```

---

## STEP 5 — DATA LAYER (`src/ppb_mcp/data.py`)

This is the most critical module. Implement with care.

### Requirements

1. **`PPBDataStore` class**: singleton that owns the DataFrame and the refresh loop.
2. **Startup load**: synchronously fetch the HuggingFace dataset via the **raw-JSONL path** (`load_dataset()` does not work — see Step 1):
   ```python
   from huggingface_hub import HfApi, hf_hub_download
   api = HfApi()
   files = [f for f in api.list_repo_files(HF_DATASET, repo_type="dataset") if f.endswith(".jsonl")]
   rows = []
   for f in files:
       p = hf_hub_download(HF_DATASET, f, repo_type="dataset")  # cached on subsequent calls
       with open(p) as fh:
           for line in fh:
               line = line.strip()
               if line:
                   rows.append(json.loads(line))
   df = pd.DataFrame(rows)
   ```
   Cache as a `pd.DataFrame` in memory. Log dataset shape and column names at `INFO` level on load.
3. **Background refresh**: use `anyio` to run a background task that re-runs the loader every `REFRESH_INTERVAL_HOURS`. To force a fresh download, clear the local repo cache directory (`huggingface_hub.scan_cache_dir().delete_revisions(...)` or simply unlink the cached snapshot directory) before re-fetching. On refresh failure, log the error and keep serving the stale cache — **never crash on refresh failure**.
4. **Thread safety**: use `asyncio.Lock` to guard DataFrame reads/writes during refresh.
5. **Schema validation**: after loading, assert that the **required columns** exist (`throughput_tok_s`, `gpu_name`, `gpu_vram_gb`, `model_base`, `quant`, `concurrent_users`). If any are missing, log a `WARNING` with the diff and continue with available columns. The reserved-but-null columns listed in Step 1 (`llm_engine_version`, `split_mode`, `tensor_split`, `quality_score`, `tags`) are explicitly whitelisted and must not trigger warnings.

### Key methods to expose

```python
class PPBDataStore:
    async def get_df(self) -> pd.DataFrame:
        """Returns the current cached DataFrame (thread-safe)."""

    def get_all_gpus(self) -> list[str]:
        """Sorted unique GPU names."""

    def get_all_models(self) -> list[str]:
        """Sorted unique model names."""

    def get_all_quantizations(self) -> list[str]:
        """Sorted unique quantization labels."""

    def get_last_refreshed(self) -> str:
        """ISO 8601 timestamp of last successful data load."""
```

---

## STEP 6 — PYDANTIC RESPONSE MODELS (`src/ppb_mcp/models.py`)

Define clean response models that FastMCP serializes to JSON for the LLM. Every tool must return a Pydantic model, never a raw dict.

Field names below are the **external (Pydantic) names** that the LLM sees; map from real columns in `data.py` (e.g. `throughput_tok_s` → `tokens_per_second`).

```python
class BenchmarkRow(BaseModel):
    """Single benchmark result row. ~12 useful fields curated from the 61 raw columns."""
    gpu_name: str                  # from gpu_name
    vram_gb: float                 # from gpu_total_vram_gb (canonical, multi-GPU-aware)
    model: str                     # from model_base, e.g. "Qwen3.5-9B"
    model_org: str                 # from model_org, e.g. "unsloth"
    model_full_path: str           # from model, e.g. "unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
    quantization: str              # from quant
    concurrent_users: int          # from concurrent_users
    tokens_per_second: float       # from throughput_tok_s
    avg_ttft_ms: float | None = None
    p50_itl_ms: float | None = None
    n_ctx: int | None = None
    backend: str | None = None     # from backends
    submitter: str | None = None
    timestamp: str | None = None

class QueryResult(BaseModel):
    rows: list[BenchmarkRow]
    total_count: int
    filtered_count: int

class QuantizationRecommendation(BaseModel):
    recommended_quantization: str
    model: str
    gpu_vram_gb: float
    concurrent_users: int
    estimated_vram_usage_gb: float
    estimated_vram_per_user_gb: float
    estimated_tokens_per_second: float
    headroom_gb: float
    confidence: Literal["high", "medium", "low"]
    reasoning: str           # Human-readable explanation for the LLM to relay
    alternatives: list[str]  # Other viable quantizations ranked by quality

class GPUHeadroom(BaseModel):
    gpu_name: str
    quantization: str
    model: str
    vram_required_gb: float
    vram_available_gb: float
    headroom_gb: float
    max_safe_concurrent_users: int
    is_viable: bool
    warning: str | None      # Non-None if headroom is tight (< 1 GB)

class TestedConfigs(BaseModel):
    gpus: list[str]
    models: list[str]
    quantizations: list[str]
    total_benchmark_rows: int
    last_updated: str
```

---

## STEP 7 — THE FOUR TOOLS

### Tool 1: `list_tested_configs`

**File**: `src/ppb_mcp/tools/list_configs.py`

**Purpose**: LLM orientation — called first to understand what data exists before issuing a targeted query.

**No input parameters.**

**Returns**: `TestedConfigs`

**Implementation**: calls `PPBDataStore.get_all_gpus()`, `get_all_models()`, `get_all_quantizations()` and returns the combined summary.

---

### Tool 2: `query_ppb_results`

**File**: `src/ppb_mcp/tools/query.py`

**Purpose**: Raw filtered data access. LLMs that want to reason over the data themselves use this.

**Input schema** (all optional, AND-filtered):

```python
class QueryInput(BaseModel):
    gpu_name: str | None = Field(None, description="Partial match on GPU name (case-insensitive)")
    vram_gb_min: float | None = Field(None, description="Minimum total VRAM in GB (filters on gpu_total_vram_gb)")
    vram_gb_max: float | None = Field(None, description="Maximum total VRAM in GB (filters on gpu_total_vram_gb)")
    model: str | None = Field(None, description="Partial match on model_base (case-insensitive), e.g. 'Qwen3.5-9B'")
    quantization: str | None = Field(None, description="Exact match on quantization label, e.g. Q4_K_M")
    backend: str | None = Field(None, description="Partial match on backend, e.g. 'CUDA' or 'Metal'")
    concurrent_users: int | None = Field(None, ge=1, le=32, description="Exact match on concurrent_users")
    limit: int = Field(50, ge=1, le=500, description="Max rows to return")
```

**Returns**: `QueryResult`

**Filtering behaviour**: use `str.contains(..., case=False, na=False)` for string partial matches. Numeric range filters are inclusive. **Default behaviour when no filters are provided**: return a stratified diverse sample — one representative row per `(gpu_name, model_base, quant)` combination, capped at `limit`. (Returning 50 random rows from 30k is useless for orientation; this gives the LLM a true cross-section.) Always populate `total_count` (full DataFrame size) and `filtered_count` (matching rows before `limit`).

---

### Tool 3: `recommend_quantization` ⭐ (primary tool)

**File**: `src/ppb_mcp/tools/recommend.py`

**Purpose**: The star tool. Given a GPU's VRAM and a desired concurrent user count, return the best quantization to run.

**Input schema**:

```python
class RecommendInput(BaseModel):
    gpu_vram_gb: float = Field(..., description="Total VRAM available on the GPU in GB, e.g. 31.8 for an RTX 5090")
    concurrent_users: int = Field(..., ge=1, le=32, description="Number of simultaneous inference requests to support")
    gpu_name: str | None = Field(None, description="Optional GPU name (partial, case-insensitive). If supplied, the algorithm prefers rows for this exact GPU; if omitted, any GPU at the matching VRAM tier is used.")
    model: str | None = Field(None, description="Specific model_base to filter on, e.g. 'Qwen3.5-9B'. If None, returns best across all tested models.")
    priority: Literal["quality", "speed", "balance"] = Field("balance", description="Optimization target: quality=highest quant that fits, speed=highest tokens/sec, balance=best tradeoff")
```

**Returns**: `QuantizationRecommendation`

**Recommendation algorithm — empirical-first, three-tier:**

The dataset contains real benchmark rows at specific `(gpu, model, quant, concurrent_users)` combinations. Any row that exists is **proof** that config fit on that GPU at that user count. Use this to ground the recommendation in measurement, not formula.

1. **Filter by model** if specified (partial match on `model_base`, case-insensitive).
2. **Tier 1 — empirical match (`confidence = "high"`):** find rows where `gpu_total_vram_gb ≤ gpu_vram_gb` (and matches `gpu_name` substring if supplied) AND `concurrent_users >= requested_concurrent_users`. Any such row is a hardware-confirmed fit. If ≥ 3 matching rows, set `confidence="high"`. If 1–2 rows, `confidence="medium"`.
3. **Tier 2 — empirical-near (`confidence = "medium"`):** if no rows match Tier 1, look for the same `(model_base, quant)` tested on a _different_ GPU at the requested user count. Use that throughput as the estimate; scale headroom against the user's `gpu_vram_gb` using the formula in Tier 3.
4. **Tier 3 — formula extrapolation (`confidence = "low"`):** when no benchmark row exists for the model+quant combo at all, fall back to:
   - `vram_per_user_gb ≈ (params_billions × bits_per_weight / 8) × 1.15`
   - Total at N concurrent: `vram_per_user_gb × concurrent_users` (overestimate; documented caveat)
   - Viable iff `total_vram ≤ gpu_vram_gb × 0.90` (10% OS/driver headroom)
   - Use shared helpers in `tools/_vram.py` (param-extraction regex over `model_base` + bits-per-weight lookup table covering all 32 quants in the dataset; see Step 1 schema notes for the table).
5. **Rank by priority** (across all viable rows from the chosen tier):
   - `"quality"` → sort by bits-per-weight descending (highest precision that fits)
   - `"speed"` → sort by `throughput_tok_s` descending
   - `"balance"` → composite: `0.5 × normalize(quant_bits) + 0.5 × normalize(throughput_tok_s)`
6. **Select top result**; derive `alternatives` as the next 2 viable quantizations (sorted by bits descending for clarity).
7. **Compute VRAM headroom from empirical row** when available: per-user VRAM ≈ `gpu_total_vram_gb / concurrent_users` of the matched row scaled by the user's request, then `headroom_gb = gpu_vram_gb - estimated_vram_usage_gb`. For Tier 3, use the formula values directly.
8. **Write `reasoning`** as a natural-language string the LLM can relay verbatim. Tier should be visible in the wording. Examples:
   - **Tier 1:** `"Q4_K_M is recommended for your RTX 5090 (32 GB) running 2 concurrent users of Qwen3.5-9B. Based on 5 measured benchmark runs, it uses ~6.2 GB per user (~12.4 GB total), leaving ~19.4 GB headroom and delivering ~187 tokens/sec per request."`
   - **Tier 2:** `"Q4_K_M is the likely best choice for your 24 GB GPU running 2 concurrent users of Qwen3.5-9B. We don't have direct measurements for this VRAM tier, but the same model+quant tested on RTX 5090 (32 GB) delivered 187 tokens/sec; headroom on your card should be ~12 GB."`
   - **Tier 3:** `"Q4_K_M is the estimated best choice for a 24 GB GPU running 2 concurrent users of LLaMA-3-8B. No direct benchmark exists for this combination — based on model size (8B params at ~4.5 bits/weight), it should require ~5.2 GB per user (~10.4 GB total). Treat the throughput estimate as approximate."`

---

### Tool 4: `get_gpu_headroom`

**File**: `src/ppb_mcp/tools/headroom.py`

**Purpose**: Sanity-check for users who already have a config in mind.

**Input schema**:

```python
class HeadroomInput(BaseModel):
    gpu_name: str = Field(..., description="GPU name, e.g. 'NVIDIA GeForce RTX 5090' (partial match)")
    quantization: str = Field(..., description="Quantization label, e.g. 'Q4_K_M'")
    model: str = Field(..., description="model_base, e.g. 'Qwen3.5-9B'")
    concurrent_users: int = Field(1, ge=1, le=32)
```

**Returns**: `GPUHeadroom`

**Logic**: look up matching benchmark row(s) by `(gpu_name, quant, model_base, concurrent_users)`. If found, use empirical `gpu_total_vram_gb` as `vram_available_gb`. If no exact row, fall back to `tools/_vram.py` formula. Set `is_viable = True` if headroom ≥ 0. Set `warning` if headroom < 1.0 GB. Set `max_safe_concurrent_users` by iterating from the requested count upward until VRAM is exhausted (or downward if not viable). Never raise on missing rows — return `is_viable=False` with an explanatory `warning`.

---

## STEP 8 — SERVER ENTRYPOINT (`src/ppb_mcp/server.py`)

```python
import os
from fastmcp import FastMCP
from ppb_mcp.data import PPBDataStore
from ppb_mcp.tools.list_configs import list_tested_configs
from ppb_mcp.tools.query import query_ppb_results
from ppb_mcp.tools.recommend import recommend_quantization
from ppb_mcp.tools.headroom import get_gpu_headroom

app = FastMCP(
    name="Poor Paul's MCP",
    description=(
        "Queryable GPU inference benchmarks from Poor Paul's Benchmark (PPB). "
        "Use recommend_quantization to find the best quantization for your GPU "
        "and concurrent user count. Data source: https://huggingface.co/datasets/paulplee/ppb-results"
    ),
    version="0.1.0",
)

# Register all four tools
app.tool(list_tested_configs)
app.tool(query_ppb_results)
app.tool(recommend_quantization)
app.tool(get_gpu_headroom)

def main():
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    if transport == "stdio":
        app.run(transport="stdio")
    else:
        app.run(transport="streamable-http", host=host, port=port)

if __name__ == "__main__":
    main()
```

The `main()` function is the `[project.scripts]` entrypoint, enabling `ppb-mcp` as a CLI command and `uvx ppb-mcp` for zero-install local use.

---

## STEP 9 — TESTS (`tests/`)

Write comprehensive tests using `pytest` and `pytest-asyncio`. Use a **static sample DataFrame fixture** in `conftest.py` for unit tests — do not call the live HuggingFace dataset in unit tests. The synthetic fixture must cover:

- At least 3 different GPUs (synthetic 8 GB, 16 GB, 24 GB VRAM — these don't exist in real data, but they're fine for exercising algorithm logic)
- At least 3 quantization levels (Q4_K_M, Q5_K_M, Q8_0)
- At least 2 model families
- Edge case rows: a GPU where only 1 quantization fits at 2 concurrent users
- Column names matching the real schema (`throughput_tok_s`, `gpu_name`, `gpu_vram_gb`, `gpu_total_vram_gb`, `model_base`, `model_org`, `model`, `quant`, `concurrent_users`, `backends`, ...)

**Add one real-data integration test** (marked `@pytest.mark.integration`, skipped by default unless `PPB_RUN_INTEGRATION=1`) that:

- Loads the real `paulplee/ppb-results` dataset via `PPBDataStore`
- Asserts `recommend_quantization(gpu_vram_gb=31.8, concurrent_users=2, gpu_name="RTX 5090", model="Qwen3.5-9B")` returns `confidence="high"` and a non-empty `reasoning`
- Asserts `list_tested_configs` reports the 3 known GPUs

### Required test coverage

**`test_tools.py`** — test each tool:

- `list_tested_configs`: returns correct counts from fixture data
- `query_ppb_results`: AND-filtering works correctly; limit is respected; empty result on impossible filter
- `recommend_quantization`:
  - correct quant selected for 8 GB GPU / 1 user
  - correct quant selected for 24 GB GPU / 2 users
  - `"low"` confidence when no matching rows in fixture
  - `priority="speed"` vs `priority="quality"` return different results when fixture supports it
- `get_gpu_headroom`: `is_viable=False` when VRAM is insufficient; warning set when headroom < 1 GB

**`test_data.py`**:

- Dataset loads and caches correctly (mock `HfApi.list_repo_files` and `hf_hub_download` to return tiny synthetic JSONL fixtures from a temp dir)
- Refresh on failure keeps stale data (mock the loader to raise; verify the previous DataFrame is still served)
- Schema validation warns on missing required columns without crashing
- Reserved-but-null columns (`tags`, `quality_score`, etc.) do **not** trigger a warning

**`test_server.py`**:

- All four tools are registered on the FastMCP app
- Tool schemas are valid JSON Schema (FastMCP exposes this)

---

## STEP 10 — DOCKERFILE AND DOCKER COMPOSE

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install build deps
RUN pip install --no-cache-dir hatchling

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

# Non-root user for security
RUN useradd -m ppbuser
USER ppbuser

EXPOSE 8000

ENV MCP_TRANSPORT=streamable-http
ENV PORT=8000
ENV HOST=0.0.0.0
ENV REFRESH_INTERVAL_HOURS=1
ENV HF_DATASET=paulplee/ppb-results

CMD ["ppb-mcp"]
```

```yaml
# docker-compose.yml
services:
  ppb-mcp:
    build: .
    image: paulplee/ppb-mcp:latest
    ports:
      - "8000:8000"
    environment:
      - HF_DATASET=${HF_DATASET:-paulplee/ppb-results}
      - REFRESH_INTERVAL_HOURS=${REFRESH_INTERVAL_HOURS:-1}
      - PORT=8000
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    restart: unless-stopped
    healthcheck:
      test:
        [
          "CMD",
          "python",
          "-c",
          "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()",
        ]
      interval: 30s
      timeout: 10s
      retries: 3
```

Add a `/health` HTTP endpoint to the FastMCP app that returns `{"status": "ok", "dataset_rows": N, "last_refreshed": "..."}`. This is used by Docker healthcheck and Lightsail monitoring.

---

## STEP 11 — DEPLOYMENT FILES (`deploy/`)

### `deploy/ppb-mcp.service` (systemd)

```ini
[Unit]
Description=Poor Paul's MCP Server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ppbuser
WorkingDirectory=/opt/ppb-mcp
ExecStart=/opt/ppb-mcp/.venv/bin/ppb-mcp
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ppb-mcp

Environment="MCP_TRANSPORT=streamable-http"
Environment="PORT=8000"
Environment="HOST=127.0.0.1"
Environment="REFRESH_INTERVAL_HOURS=1"
Environment="HF_DATASET=paulplee/ppb-results"

[Install]
WantedBy=multi-user.target
```

Note `HOST=127.0.0.1` — in production the process binds only to localhost; nginx handles external exposure.

### `deploy/nginx-ppb-mcp.conf` (nginx server block)

```nginx
server {
    listen 443 ssl http2;
    server_name mcp.poorpaul.dev;

    ssl_certificate /etc/letsencrypt/live/mcp.poorpaul.dev/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mcp.poorpaul.dev/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    # SSE requires buffering disabled
    proxy_buffering off;
    proxy_cache off;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 10s;
    }
}

server {
    listen 80;
    server_name mcp.poorpaul.dev;
    return 301 https://$host$request_uri;
}
```

### `deploy/deploy.sh`

Write a deploy script that:

1. Pulls the latest code from `main`
2. Installs/upgrades the package in a venv at `/opt/ppb-mcp/.venv`
3. Runs `systemctl restart ppb-mcp`
4. Tails the journal for 10 seconds to confirm startup
5. Runs the health check endpoint

---

## STEP 12 — GITHUB ACTIONS

### `.github/workflows/ci.yml`

Triggers on every PR to `main`. Steps:

1. `actions/checkout`
2. Set up Python 3.11
3. `pip install -e ".[dev]"` (add `pytest`, `pytest-asyncio`, `ruff` to dev extras in `pyproject.toml`)
4. `ruff check src/ tests/`
5. `pytest tests/ -v --tb=short`

### `.github/workflows/publish.yml`

Triggers on `push` to tags matching `v*.*.*`. Steps:

1. `actions/checkout`
2. Set up Python 3.11
3. `pip install build`
4. `python -m build`
5. `pypa/gh-action-pypi-publish` (uses `PYPI_API_TOKEN` secret)

---

## STEP 13 — README.md

The README is a brand asset. Write it with the following structure and tone — direct, technical, no marketing fluff.

```markdown
# Poor Paul's MCP

> GPU inference benchmark data, queryable by any LLM.

Poor Paul's MCP exposes the [Poor Paul's Benchmark](https://github.com/paulplee/poor-pauls-benchmark)
dataset as an MCP (Model Context Protocol) server. Connect it to Claude Desktop, Cursor, Windsurf,
or any MCP-compatible client and ask:

> "Which quantization should I run on my RTX 5090 for 2 concurrent users?"

(The current dataset covers RTX 5090, Apple M4 Pro, and NVIDIA GB10. For untested GPUs the server uses tier-2/tier-3 extrapolation — see the `recommend_quantization` confidence field.)

## Quick Start

### Option A — Use the hosted server (zero setup)

Add this to your MCP client config:

\`\`\`json
{
"mcpServers": {
"ppb": {
"url": "https://mcp.poorpaul.dev",
"transport": "streamable-http"
}
}
}
\`\`\`

### Option B — Self-host with uvx (no install)

\`\`\`bash
MCP_TRANSPORT=stdio uvx ppb-mcp
\`\`\`

### Option C — Self-host with Docker

\`\`\`bash
docker run -p 8000:8000 paulplee/ppb-mcp
\`\`\`

### Option D — Install and run

\`\`\`bash
pip install ppb-mcp
ppb-mcp
\`\`\`

## Available Tools

| Tool                     | Purpose                                           |
| ------------------------ | ------------------------------------------------- |
| `list_tested_configs`    | See all tested GPUs, models, and quantizations    |
| `query_ppb_results`      | Filter raw benchmark rows                         |
| `recommend_quantization` | Get the best quant for your GPU and user count ⭐ |
| `get_gpu_headroom`       | Check VRAM usage for a specific config           
### Option D — Install and run

\`\`\`bash
pip install ppb-mcp
ppb-mcp
\`\`\`

## Available Tools

| Tool | Purpose |
|------|---------|
| `list_tested_configs` | See all tested GPUs, models, and quantizations |
| `query_ppb_results` | Filter raw benchmark rows |
| `recommend_quantization` | Get the best quant for your GPU and user count ⭐ |
| `get_gpu_headroom` | Check VRAM usage for a specific config |

## Data Source

Benchmarks live at [`paulplee/ppb-results`](https://huggingface.co/datasets/paulplee/ppb-results)
on HuggingFace. The server refreshes its cache hourly. Want to contribute benchmarks?
See [poor-pauls-benchmark](https://github.com/paulplee/poor-pauls-benchmark).

## Self-Hosting with a Custom Dataset

Point the server at your own HuggingFace dataset (must match PPB schema):

\`\`\`bash
HF_DATASET=yourorg/your-dataset ppb-mcp
\`\`\`

## Analytics

Human-readable benchmark analytics: [poorpaul.dev/insights](https://poorpaul.dev/insights)

## License

MIT
```

---

## STEP 14 — DISCOVERY FILES

These two files live in the `poorpaul.dev` S3 bucket (not in this repo). Include them in the repo under `docs/` as reference and note in the README that they are served from the main site.

### `docs/llms.txt` (serves at `poorpaul.dev/llms.txt`)

```
# Poor Paul's Benchmark

> Empirical GPU inference benchmarks for consumer and prosumer hardware.
> Maintained by Paul Lee — https://poorpaul.dev

## MCP Server

The Poor Paul's MCP server provides structured, queryable access to all PPB benchmark data.
Connect any MCP client to answer questions like "which quantization fits on my GPU?"

- Endpoint: https://mcp.poorpaul.dev
- Transport: Streamable HTTP (MCP protocol)
- Source: https://github.com/paulplee/ppb-mcp

## Dataset

Raw benchmark data in Parquet format, versioned and public.

- HuggingFace: https://huggingface.co/datasets/paulplee/ppb-results

## Available MCP Tools

- list_tested_configs() — enumerate tested GPUs, models, and quantizations
- query_ppb_results(gpu_name, model, quantization, vram_gb_min, vram_gb_max) — filtered data access
- recommend_quantization(gpu_vram_gb, concurrent_users, model, priority) — primary recommendation
- get_gpu_headroom(gpu_name, quantization, model, concurrent_users) — VRAM headroom check

## Benchmark Code

Run your own benchmarks and contribute results:
https://github.com/paulplee/poor-pauls-benchmark
```

### `docs/mcp.json` (serves at `poorpaul.dev/.well-known/mcp.json`)

```json
{
  "mcp_version": "1.0",
  "name": "Poor Paul's MCP",
  "description": "Queryable GPU inference benchmarks — find the right quantization for your hardware and concurrent user count",
  "endpoint": "https://mcp.poorpaul.dev",
  "transport": "streamable-http",
  "tools": [
    "list_tested_configs",
    "query_ppb_results",
    "recommend_quantization",
    "get_gpu_headroom"
  ],
  "source": "https://github.com/paulplee/ppb-mcp",
  "data": "https://huggingface.co/datasets/paulplee/ppb-results",
  "homepage": "https://poorpaul.dev"
}
```

---

## STEP 15 — COMPLETION CHECKLIST

Before declaring done, verify each item:

- [ ] `python -m pytest tests/ -v` passes with no failures
- [ ] `ruff check src/ tests/` returns no errors
- [ ] `ppb-mcp --help` works (or `python -m ppb_mcp.server` runs cleanly)
- [ ] `uvx ppb-mcp` (stdio mode) starts and responds to a `list_tested_configs` call
- [ ] Docker image builds cleanly: `docker build -t ppb-mcp .`
- [ ] Docker container starts and `/health` returns 200: `docker run -p 8000:8000 ppb-mcp`
- [ ] All four tools are registered and visible in FastMCP tool listing
- [ ] `recommend_quantization` returns a `reasoning` string that is human-readable and relay-ready
- [ ] `query_ppb_results` returns empty `rows` (not an error) when no results match
- [ ] README Option A, B, C, D all produce working configs
- [ ] `docs/llms.txt` and `docs/mcp.json` are present and valid
- [ ] `pyproject.toml` has all required metadata for PyPI publish
- [ ] CI workflow triggers and passes on a test branch

---

## IMPORTANT NOTES FOR OPUS 4.7

1. **The schema in Step 1 is verified against live data.** Real columns are `throughput_tok_s` (not `tokens_per_second`), `model_base` (not `model`), `quant` (not `quantization`). Pydantic field names are renamed for LLM clarity; mapping happens in `data.py`.

2. **`load_dataset()` does not work for this dataset** — use the raw-JSONL `huggingface_hub` path documented in Steps 1 and 5. Do not add the `datasets` package as a dependency.

3. **The `recommend_quantization` tool is the most important.** It uses a three-tier empirical-first algorithm (see Step 7). The `reasoning` field is what users actually see — write it as a clear, confident sentence the LLM can relay verbatim. The tier should be implicit in the wording (e.g. "based on N measured benchmark runs" vs. "estimated").

4. **Fail gracefully everywhere**. If a tool receives a GPU name that doesn't exist in the dataset, return a clear message in the response model rather than raising. Never let a missing row crash a tool call. `query_ppb_results` returns `rows=[]`, never an error.

5. **The `PPBDataStore` is a singleton**. Instantiate once at startup; share across all tool calls. Do not reload on every tool invocation.

6. **Transport duality matters for adoption**. The `stdio` path (for `uvx ppb-mcp`) is how most individual users will first encounter the server. Make sure it works cleanly without any HTTP server startup noise polluting stdout — FastMCP handles this, but verify by running `MCP_TRANSPORT=stdio ppb-mcp` and confirming stdout contains only MCP protocol bytes.

7. **Test with the real dataset at least once** via the integration test in Step 9 before finalizing. The synthetic fixture validates algorithm logic; the integration test validates real column-name mapping and value ranges.

8. **Real GPU/quant/model values to use in examples** (instead of the spec's original RTX 3090 / llama-3-8b examples, which don't exist in current data): `RTX 5090` (31.8 GB), `Apple M4 Pro` (64 GB), `NVIDIA GB10` (119.6 GB); models like `Qwen3.5-9B`, `gpt-oss-20b`, `gemma-4-E4B-it`; quants like `Q4_K_M`, `Q5_K_M`, `Q8_0`, `IQ4_XS`, `Q4_K_XL`.
