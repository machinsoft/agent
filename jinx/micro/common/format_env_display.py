from __future__ import annotations

from typing import Dict, Iterable, Optional

__all__ = ["format_env_display"]


def format_env_display(env: Optional[Dict[str, str]], env_vars: Iterable[str]) -> str:
    parts: list[str] = []
    if env:
        for k in sorted(env.keys()):
            parts.append(f"{k}=*****")
    for var in env_vars or []:
        parts.append(f"{var}=*****")
    return "-" if not parts else ", ".join(parts)
