"""Shared filter helpers for ppb-mcp tools."""

from __future__ import annotations

_NULL_STRINGS = frozenset({"null", "none", "undefined", ""})


def is_blank(value: str | None) -> bool:
    """Return True if the value should be treated as 'no filter'.

    LLM agents sometimes pass the JSON string "null" instead of omitting a
    parameter or passing JSON null. This helper treats those sentinel strings
    as equivalent to an absent value so they do not break dataset queries.
    """
    if value is None:
        return True
    return str(value).strip().lower() in _NULL_STRINGS
