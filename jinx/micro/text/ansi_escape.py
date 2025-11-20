from __future__ import annotations

import re
from typing import Any

__all__ = [
    "expand_tabs",
    "ansi_escape",
    "ansi_escape_line",
]

# Simple ANSI escape sequence pattern (CSI sequences)
_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def expand_tabs(s: str) -> str:
    """Replace tabs with four spaces.

    Intentional simplification to avoid gutter collisions in transcript-like views.
    """
    if "\t" not in s:
        return s
    return s.replace("\t", "    ")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def ansi_escape(s: str) -> Any:
    """Convert ANSI-bearing string into PTK-friendly text; fallback to plain text.

    Returns an object acceptable by prompt_toolkit.print_formatted_text.
    - Preferred: prompt_toolkit.formatted_text.ANSI(s)
    - Fallback: raw string with ANSI sequences stripped
    """
    text = expand_tabs(s)
    try:
        # Late import to keep module light and optional
        ANSI = __import__("prompt_toolkit.formatted_text", fromlist=["ANSI"]).ANSI  # type: ignore[attr-defined]
        return ANSI(text)
    except Exception:
        return _strip_ansi(text)


def ansi_escape_line(s: str) -> str:
    """Return first line after ANSI normalization; log if multiple lines."""
    text = expand_tabs(s)
    # Avoid ANSI for single-line extraction; strip for consistency
    cleaned = _strip_ansi(text)
    if "\n" in cleaned:
        try:
            from jinx.micro.logger.debug_logger import debug_log_sync

            debug_log_sync(
                "ansi_escape_line: expected a single line; truncating to first line",
                category="WARN",
            )
        except Exception:
            pass
    first, *_ = cleaned.splitlines() or [""]
    return first
