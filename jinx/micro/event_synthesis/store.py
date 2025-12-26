from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
import time
from typing import Any, Deque, Dict, List, Optional


@dataclass(frozen=True)
class SynthEvent:
    seq: int
    t: float
    topic: str
    group: str
    weight: float
    payload: Any


class EventStore:
    def __init__(self, *, per_group: int = 256, global_max: int = 512) -> None:
        self._lock = Lock()
        self._seq = 0
        self._per_group = int(per_group)
        self._groups: Dict[str, Deque[SynthEvent]] = {}
        self._global: Deque[SynthEvent] = deque(maxlen=int(global_max))

    def record(self, topic: str, payload: Any, *, group: str, weight: float) -> SynthEvent:
        g = (group or "main").strip() or "main"
        t = time.monotonic()
        with self._lock:
            self._seq += 1
            ev = SynthEvent(
                seq=self._seq,
                t=t,
                topic=str(topic or ""),
                group=g,
                weight=float(weight),
                payload=payload,
            )
            dq = self._groups.get(g)
            if dq is None:
                dq = deque(maxlen=self._per_group)
                self._groups[g] = dq
            dq.append(ev)
            self._global.append(ev)
            return ev

    def snapshot(self, *, group: Optional[str] = None) -> List[SynthEvent]:
        g = (group or "").strip()
        with self._lock:
            if g:
                dq = self._groups.get(g)
                return list(dq) if dq is not None else []
            return list(self._global)

    def clear(self, *, group: Optional[str] = None) -> None:
        g = (group or "").strip()
        with self._lock:
            if g:
                self._groups.pop(g, None)
                return
            self._groups.clear()
            self._global.clear()
