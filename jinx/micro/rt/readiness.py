from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Set

__all__ = [
    "ReadinessError",
    "ReadinessFlag",
]


class ReadinessError(Exception):
    """Errors for readiness subscription workflow."""

    TOKEN_LOCK_FAILED = "TokenLockFailed"
    FLAG_ALREADY_READY = "FlagAlreadyReady"

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


@dataclass(frozen=True)
class _Token:
    value: int


class ReadinessFlag:
    """Token-authorized readiness flag with async waiting.

    - subscribe() returns an authorization token; fails if already ready.
    - mark_ready(token) flips flag if token is valid (one-shot).
    - wait_ready() awaits readiness with broadcast semantics.

    Lock hold times are bounded by LOCK_TIMEOUT for real-time safety.
    """

    LOCK_TIMEOUT: float = 1.0  # seconds

    def __init__(self) -> None:
        self._ready: bool = False
        self._next_id: int = 1  # 0 is reserved
        self._tokens: Set[int] = set()
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()

    def __repr__(self) -> str:  # lightweight debug
        return f"ReadinessFlag(ready={self._ready})"

    def is_ready(self) -> bool:
        # Fast-path read; non-reversible once true
        if self._ready:
            return True
        # Opportunistic fast-path: no subscribers implies ready
        # (mirrors Rust try_lock + empty set path; best-effort without lock)
        if not self._tokens:
            self._ready = True
            self._event.set()
        return self._ready

    async def _acquire(self) -> None:
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self.LOCK_TIMEOUT)
        except asyncio.TimeoutError as _:
            raise ReadinessError(ReadinessError.TOKEN_LOCK_FAILED)

    def _release(self) -> None:
        try:
            if self._lock.locked():
                self._lock.release()
        except RuntimeError:
            pass

    async def subscribe(self) -> int:
        if self.is_ready():
            raise ReadinessError(ReadinessError.FLAG_ALREADY_READY)

        await self._acquire()
        try:
            if self._ready:
                raise ReadinessError(ReadinessError.FLAG_ALREADY_READY)
            token = self._next_id
            self._next_id += 1
            self._tokens.add(token)
            return token
        finally:
            self._release()

    async def mark_ready(self, token: int) -> bool:
        if self._ready:
            return False
        if token <= 0:
            return False

        await self._acquire()
        try:
            if token not in self._tokens:
                return False
            self._tokens.discard(token)
            self._tokens.clear()  # one token is enough; invalidate others
            self._ready = True
            self._event.set()
            return True
        finally:
            self._release()

    async def wait_ready(self) -> None:
        if self.is_ready():
            return
        await self._event.wait()
