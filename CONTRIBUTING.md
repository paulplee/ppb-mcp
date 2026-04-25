# Contributing to ppb-mcp

Thanks for your interest! This project ships small, but it's intended to be reliable infrastructure for anyone choosing GPU/quantization configs.

## Setup

```bash
git clone https://github.com/paulplee/ppb-mcp
cd ppb-mcp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quality gates

Before opening a PR, please run:

```bash
ruff check src tests
pytest -v
```

CI runs the same on Python 3.11 and 3.12.

## Adding a new tool

1. Implement it as `async def` in a new file under `src/ppb_mcp/tools/`.
2. Define a Pydantic response model in `src/ppb_mcp/models.py`.
3. Register it in `src/ppb_mcp/server.py` via `app.tool(your_func)`.
4. Add unit tests in `tests/test_tools.py` (one class per tool).

## Tool design rules

- **Never raise on missing data.** Return an empty result or a non-viable flag.
- **Use the singleton store**: `PPBDataStore.instance()` + `await store.ensure_loaded()`.
- **Empirical first, formula as fallback** — always prefer measured rows over extrapolation.
- **Tool docstrings are user-facing.** They become MCP tool descriptions.

## Releasing

Bump `__version__` in `src/ppb_mcp/__init__.py` and `version` in `pyproject.toml`, tag `vX.Y.Z`, push the tag. The publish workflow uploads to PyPI via Trusted Publishing.
