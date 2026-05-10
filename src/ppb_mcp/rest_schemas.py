"""Pydantic v2 models for REST API query parameter validation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

_VALID_CONCURRENT_USERS = frozenset({1, 2, 4, 8, 16, 32})


class ResultsQueryParams(BaseModel):
    """Query parameters for GET /api/v1/results."""

    gpu: str | None = None
    model: str | None = None
    quant: str | None = None
    runner_type: str | None = None
    concurrent_users: int | None = None
    vram_min: float | None = None
    vram_max: float | None = None
    unified_memory: bool | None = None
    run_after: str | None = None
    run_before: str | None = None
    # Schema cap is 5000 (the endpoint enforces the tighter 500 / 5000 dynamic cap).
    limit: int = Field(default=100, ge=1, le=5000)

    @field_validator("unified_memory", mode="before")
    @classmethod
    def _parse_bool(cls, v: object) -> object:
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return v

    @field_validator("concurrent_users")
    @classmethod
    def _validate_concurrent_users(cls, v: int | None) -> int | None:
        if v is not None and v not in _VALID_CONCURRENT_USERS:
            raise ValueError(
                f"concurrent_users must be one of: {sorted(_VALID_CONCURRENT_USERS)}"
            )
        return v

    @field_validator("vram_min", "vram_max")
    @classmethod
    def _validate_vram(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("vram values must be non-negative")
        return v


class QualitativeQueryParams(BaseModel):
    """Query parameters for GET /api/v1/qualitative."""

    model: str | None = None
    quant: str | None = None
    gpu: str | None = None
    runner_type: str | None = None
    min_context_rot: float | None = None
    min_tool_accuracy: float | None = None
    min_mt_bench: float | None = None
    limit: int = Field(default=50, ge=1, le=200)

    @field_validator("min_context_rot", "min_tool_accuracy", "min_mt_bench")
    @classmethod
    def _validate_score(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("score must be between 0 and 1")
        return v


class CompareQueryParams(BaseModel):
    """Query parameters for GET /api/v1/compare/quants."""

    model: str = Field(..., min_length=1)
    gpu: str | None = None
    runner_type: str | None = None
    concurrent_users: int | None = None

    @field_validator("concurrent_users")
    @classmethod
    def _validate_concurrent_users(cls, v: int | None) -> int | None:
        if v is not None and v not in _VALID_CONCURRENT_USERS:
            raise ValueError(
                f"concurrent_users must be one of: {sorted(_VALID_CONCURRENT_USERS)}"
            )
        return v


class ContextRotQueryParams(BaseModel):
    """Query parameters for GET /api/v1/context-rot."""

    model: str = Field(..., min_length=1)
    quant: str = Field(..., min_length=1)
    gpu: str | None = None


class ToolAccuracyQueryParams(BaseModel):
    """Query parameters for GET /api/v1/tool-accuracy."""

    model: str = Field(..., min_length=1)
    quant: str = Field(..., min_length=1)
    gpu: str | None = None
