from __future__ import annotations

import time
from typing import Dict, Set


class DedupRegistry:
    """Time-bounded de-duplication registry.

    Tracks ids in three states:
    - pending: admitted to queues but not yet scheduled
    - inflight: scheduled/running
    - recent_done: finished within TTL

    A new id is admitted only if it is in none of these sets.
    """

    def __init__(self, ttl_seconds: float = 600.0):
        self._ttl = float(max(1.0, ttl_seconds))
        self._pending: Set[str] = set()
        self._inflight: Set[str] = set()
        self._done_at: Dict[str, float] = {}

    def _purge(self, now: float | None = None) -> None:
        t = now if now is not None else time.monotonic()
        ttl = self._ttl
        if not self._done_at:
            return
        # Fast path: periodic thinning; avoid full scan too frequently
        # Still safe to do simple scan given typical sizes
        expired = [k for k, ts in self._done_at.items() if (t - ts) > ttl]
        for k in expired:
            self._done_at.pop(k, None)

    def try_admit(self, msg_id: str, *, now: float | None = None) -> bool:
        if not msg_id:
            return True  # treat missing id as admissible; caller should ensure id
        self._purge(now)
        if msg_id in self._pending or msg_id in self._inflight or msg_id in self._done_at:
            return False
        self._pending.add(msg_id)
        return True

    def on_scheduled(self, msg_id: str) -> None:
        if not msg_id:
            return
        self._pending.discard(msg_id)
        self._inflight.add(msg_id)

    def on_finished(self, msg_id: str, *, now: float | None = None) -> None:
        if not msg_id:
            return
        self._inflight.discard(msg_id)
        self._pending.discard(msg_id)
        t = now if now is not None else time.monotonic()
        self._done_at[msg_id] = t
        self._purge(t)

    def clear(self) -> None:
        self._pending.clear()
        self._inflight.clear()
        self._done_at.clear()
