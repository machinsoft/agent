from __future__ import annotations

from .token import is_admin


def is_elevated() -> bool:
    return bool(is_admin())


def ensure_elevated() -> bool:
    return is_elevated()
