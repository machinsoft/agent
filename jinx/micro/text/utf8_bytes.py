from __future__ import annotations

__all__ = [
    "take_bytes_at_char_boundary",
    "take_last_bytes_at_char_boundary",
]


def take_bytes_at_char_boundary(s: str, max_bytes: int) -> str:
    """Return prefix of ``s`` fitting into ``max_bytes`` UTF-8 bytes.

    The cut is made only at Unicode code point boundaries (no partial bytes).
    """
    if max_bytes <= 0:
        return ""
    if not s:
        return s

    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    return b[:max_bytes].decode("utf-8", errors="ignore")


def take_last_bytes_at_char_boundary(s: str, max_bytes: int) -> str:
    """Return suffix of ``s`` fitting into ``max_bytes`` UTF-8 bytes.

    The cut is made only at Unicode code point boundaries (no partial bytes).
    """
    if max_bytes <= 0:
        return ""
    if not s:
        return s

    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s

    tail = b[-max_bytes:]
    return tail.decode("utf-8", errors="ignore")
