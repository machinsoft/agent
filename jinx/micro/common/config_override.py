from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

try:  # Python 3.11+
    import tomllib as _toml  # type: ignore
except Exception:  # pragma: no cover - fallback
    try:
        import tomli as _toml  # type: ignore
    except Exception:  # pragma: no cover - final fallback
        _toml = None  # type: ignore

__all__ = [
    "parse_overrides",
    "apply_on_value",
]


def _parse_toml_like(raw: str) -> Any:
    s = raw.strip()
    # Prefer TOML if available
    if _toml is not None:
        try:
            data = _toml.loads(f"_x_ = {s}")
            return data.get("_x_")
        except Exception:
            pass
    # Try JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fallback scalars
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        pass
    # Strip optional quotes
    if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def parse_overrides(raw_overrides: Iterable[str]) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for item in raw_overrides or []:
        s = str(item)
        if "=" not in s:
            raise ValueError(f"Invalid override (missing '='): {s}")
        key, value_str = s.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in override: {s}")
        val = _parse_toml_like(value_str.strip())
        out.append((key, val))
    return out


def apply_on_value(target: Dict[str, Any], overrides: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    root = target
    for path, value in overrides:
        parts = [p for p in path.split(".") if p]
        cur: Dict[str, Any] = root
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            if is_last:
                cur[part] = value
                break
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
    return root
