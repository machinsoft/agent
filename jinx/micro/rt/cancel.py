from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Generic, Optional, TypeVar

__all__ = [
    "CancelErr",
    "CancellationToken",
    "or_cancel",
]

T = TypeVar("T")


class CancelErr(Exception):
    """Raised when a wait was cancelled via token."""

    def __str__(self) -> str:  # minimal debug text
        return "Cancelled"


class CancellationToken:
    """Lightweight cancellation token.

    - call `cancel()` to broadcast cancellation
    - await `wait_cancelled()` to await cancellation
    - check `is_cancelled()` for a non-blocking probe
    """

    __slots__ = ("_ev",)

    def __init__(self) -> None:
        self._ev = asyncio.Event()

    def cancel(self) -> None:
        self._ev.set()

    def is_cancelled(self) -> bool:
        return self._ev.is_set()

    async def wait_cancelled(self) -> None:
        await self._ev.wait()


async def or_cancel(awaitable: Awaitable[T], token: CancellationToken) -> T:
    """Await `awaitable` unless `token` is cancelled first.

    Returns:
      - result of `awaitable` when it completes first
    Raises:
      - CancelErr when token cancels first
    """
    task = asyncio.create_task(awaitable)
    cancel_task = asyncio.create_task(token.wait_cancelled())
    try:
        done, pending = await asyncio.wait(
            {task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancel_task in done and not task.done():
            task.cancel()
            raise CancelErr()
        # Else, main task completed
        return await task
    finally:
        # Cleanup whichever remains
        for p in (task, cancel_task):
            if not p.done():
                p.cancel()
