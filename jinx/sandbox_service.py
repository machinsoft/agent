"""Sandbox service facade.

Thin wrapper delegating to the micro-module implementation under
``jinx.micro.sandbox.service`` to keep the public API stable.
"""

from __future__ import annotations

__all__ = [
    "blast_zone",
    "arcane_sandbox",
]


def blast_zone(*args, **kwargs):
    from jinx.micro.sandbox.service import blast_zone as _blast_zone
    return _blast_zone(*args, **kwargs)


async def arcane_sandbox(*args, **kwargs):
    from jinx.micro.sandbox.service import arcane_sandbox as _arcane_sandbox
    return await _arcane_sandbox(*args, **kwargs)
