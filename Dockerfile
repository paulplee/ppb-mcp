# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/huggingface

WORKDIR /app

# Install build deps for hatchling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata first for better caching
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip hatchling \
    && pip install .

# Non-root user
RUN groupadd --system ppbuser && useradd --system --gid ppbuser --home-dir /home/ppbuser --create-home ppbuser \
    && mkdir -p /data/huggingface && chown -R ppbuser:ppbuser /data
USER ppbuser

ENV MCP_TRANSPORT=streamable-http \
    HOST=0.0.0.0 \
    PORT=9933 \
    LOG_LEVEL=INFO

EXPOSE 9933

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://0.0.0.0:9933/health || exit 1

CMD ["ppb-mcp"]
