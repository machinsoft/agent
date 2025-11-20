from __future__ import annotations

import json as _json
import logging as _log
import os as _os
import random as _rand
from typing import Any

__all__ = [
    "backoff",
    "error_or_panic",
    "try_parse_error_message",
]

_INITIAL_DELAY_MS: float = 200.0
_BACKOFF_FACTOR: float = 2.0


def backoff(attempt: int) -> float:
    """Exponential backoff with jitter. Returns delay in seconds.

    attempt >= 1
    """
    a = max(1, int(attempt))
    base_ms = _INITIAL_DELAY_MS * (_BACKOFF_FACTOR ** (a - 1))
    jitter = _rand.uniform(0.9, 1.1)
    return (base_ms * jitter) / 1000.0


def error_or_panic(message: str) -> None:
    """Log error or raise in debug/alpha modes.

    Controlled via env:
      JINX_PANIC_ON_ERROR=1 -> raise RuntimeError
    """
    if str(_os.getenv("JINX_PANIC_ON_ERROR", "")).strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(message)
    _log.error(message)


def try_parse_error_message(text: str) -> str:
    """Extract error.message from JSON string; otherwise return input or a default."""
    try:
        obj = _json.loads(text or "{}")
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg:
                return msg
    except Exception:
        pass
    if not text:
        return "Unknown error"
    return text
