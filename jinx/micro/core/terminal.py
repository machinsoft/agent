from __future__ import annotations

import os
import re
from functools import lru_cache

__all__ = ["user_agent"]

_VALID_RE = re.compile(r"[A-Za-z0-9\-_./]")


def _sanitize_header_value(value: str) -> str:
    return "".join(ch if _VALID_RE.match(ch) else "_" for ch in value)


@lru_cache(maxsize=1)
def user_agent() -> str:
    tp = os.getenv("TERM_PROGRAM", "").strip()
    if tp:
        ver = os.getenv("TERM_PROGRAM_VERSION", "").strip()
        return _sanitize_header_value(f"{tp}/{ver}" if ver else tp)

    v = os.getenv("WEZTERM_VERSION", "").strip()
    if v:
        return _sanitize_header_value(f"WezTerm/{v}")

    if os.getenv("KITTY_WINDOW_ID") or (os.getenv("TERM", "").find("kitty") != -1):
        return _sanitize_header_value("kitty")

    if os.getenv("ALACRITTY_SOCKET") or (os.getenv("TERM", "") == "alacritty"):
        return _sanitize_header_value("Alacritty")

    v = os.getenv("KONSOLE_VERSION", "").strip()
    if v:
        return _sanitize_header_value(f"Konsole/{v}")

    if os.getenv("GNOME_TERMINAL_SCREEN"):
        return _sanitize_header_value("gnome-terminal")

    v = os.getenv("VTE_VERSION", "").strip()
    if v:
        return _sanitize_header_value(f"VTE/{v}")

    if os.getenv("WT_SESSION"):
        return _sanitize_header_value("WindowsTerminal")

    return _sanitize_header_value(os.getenv("TERM", "unknown"))
