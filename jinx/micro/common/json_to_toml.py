from __future__ import annotations

from typing import Any

__all__ = ["json_to_toml"]


def json_to_toml(v: Any) -> Any:
    """Convert JSON-like Python value to TOML-serializable value.

    Mapping mirrors the Rust implementation semantics:
    - None -> "" (empty string)
    - bool -> bool
    - int  -> int
    - float -> float
    - str -> str
    - list/tuple -> list (recursively converted)
    - dict -> dict[str, Any] (recursively converted)
    - other -> str(value)
    """
    if v is None:
        return ""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return [json_to_toml(x) for x in v]
    if isinstance(v, dict):
        # Ensure keys are strings for TOML
        return {str(k): json_to_toml(val) for k, val in v.items()}
    # Fallback: stringify unknown types
    try:
        return str(v)
    except Exception:
        return ""
