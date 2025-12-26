from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .store import EventStore, SynthEvent


_STORE = EventStore()


def record_event(topic: str, payload: Any, *, group: str = "main", weight: float = 1.0) -> SynthEvent:
    return _STORE.record(topic, payload, group=group, weight=weight)


def snapshot_events(*, group: Optional[str] = None) -> List[SynthEvent]:
    return _STORE.snapshot(group=group)


def clear_events(*, group: Optional[str] = None) -> None:
    _STORE.clear(group=group)


def _compact_ws(s: str) -> str:
    return " ".join((s or "").split())


def _clip(s: str, n: int) -> str:
    if n <= 0:
        return ""
    if len(s) <= n:
        return s
    return s[: max(0, n - 3)] + "..."


def _render_payload(payload: Any, *, max_chars: int) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return _clip(_compact_ws(payload), max_chars)
    if isinstance(payload, dict):
        parts: List[str] = []
        for k in sorted(payload.keys(), key=lambda x: str(x)):
            ks = str(k)
            if ks in {"tb", "traceback", "stack", "diff", "unified_diff"}:
                continue
            v = payload.get(k)
            if v is None:
                continue
            vs = _compact_ws(str(v))
            if not vs:
                continue
            parts.append(f"{ks}={_clip(vs, 96)}")
        return _clip(" ".join(parts), max_chars)
    return _clip(_compact_ws(str(payload)), max_chars)


def _score_event(ev: SynthEvent, now: float) -> float:
    age = now - float(ev.t)
    if age <= 0:
        return float(ev.weight)
    rec = 1.0 / (1.0 + (age / 6.0))
    return float(ev.weight) * rec


def build_event_stream_block(
    group: str,
    *,
    max_events: int = 48,
    max_chars: int = 1800,
    include_global_fallback: bool = True,
) -> str:
    g = (group or "main").strip() or "main"
    now = time.monotonic()

    evs = snapshot_events(group=g)
    if (not evs) and include_global_fallback:
        evs = snapshot_events(group=None)

    if not evs:
        return ""

    scored: List[Tuple[float, SynthEvent]] = [(_score_event(ev, now), ev) for ev in evs]
    scored.sort(key=lambda t: (t[0], t[1].seq), reverse=True)

    head = "<event_stream>\n" + f"group: {g}\n"
    tail = "</event_stream>"

    buf: List[str] = [head]
    budget = max(0, int(max_chars) - len(head) - len(tail) - 2)

    used = 0
    added = 0
    for _, ev in scored:
        if added >= int(max_events):
            break
        age = now - float(ev.t)
        pay = _render_payload(ev.payload, max_chars=240)
        if ev.group != g and include_global_fallback:
            line = f"- w={ev.weight:.2f} age={age:.1f}s g={ev.group} {ev.topic} {pay}\n"
        else:
            line = f"- w={ev.weight:.2f} age={age:.1f}s {ev.topic} {pay}\n"
        if used + len(line) > budget:
            break
        buf.append(line)
        used += len(line)
        added += 1

    buf.append(tail)
    return "".join(buf)


def _safe_get(d: Any, key: str) -> str:
    if not isinstance(d, dict):
        return ""
    try:
        v = d.get(key)
    except Exception:
        return ""
    if v is None:
        return ""
    try:
        s = str(v)
    except Exception:
        return ""
    return _compact_ws(s)


def build_event_state_block(
    group: str,
    *,
    max_chars: int = 900,
    include_global_fallback: bool = True,
) -> str:
    g = (group or "main").strip() or "main"
    evs = snapshot_events(group=g)
    if (not evs) and include_global_fallback:
        evs = snapshot_events(group=None)
    if not evs:
        return ""

    last_user = ""
    last_turn = ""
    last_llm = ""
    last_exec_err = ""
    last_task = ""

    for ev in reversed(evs):
        if not last_user and ev.topic == "user.input":
            last_user = _safe_get(ev.payload, "text")
            last_turn = _safe_get(ev.payload, "turn_id")
        elif not last_llm and ev.topic == "llm.output":
            last_llm = _safe_get(ev.payload, "preview")
            if not last_turn:
                last_turn = _safe_get(ev.payload, "turn_id")
        elif not last_exec_err and ev.topic == "exec.error":
            last_exec_err = _safe_get(ev.payload, "error")
            if not last_turn:
                last_turn = _safe_get(ev.payload, "turn_id")
        elif not last_task and ev.topic in {"task.result", "task.progress", "task.request"}:
            tid = _safe_get(ev.payload, "id")
            name = _safe_get(ev.payload, "name")
            ok = _safe_get(ev.payload, "ok")
            msg = _safe_get(ev.payload, "msg")
            pct = _safe_get(ev.payload, "pct")
            if ev.topic == "task.result":
                last_task = _compact_ws(f"result id={tid} name={name} ok={ok}")
            elif ev.topic == "task.progress":
                last_task = _compact_ws(f"progress id={tid} name={name} pct={pct} msg={msg}")
            else:
                last_task = _compact_ws(f"request id={tid} name={name}")
        if last_user and last_llm and last_exec_err and last_task:
            break

    head = "<event_state>\n" + f"group: {g}\n"
    tail = "</event_state>"
    budget = max(0, int(max_chars) - len(head) - len(tail) - 2)
    lines: List[str] = []
    if last_turn:
        lines.append(f"turn_id: {_clip(last_turn, 64)}\n")
    if last_user:
        lines.append(f"last_user: {_clip(last_user, 320)}\n")
    if last_exec_err:
        lines.append(f"last_error: {_clip(last_exec_err, 360)}\n")
    if last_task:
        lines.append(f"last_task: {_clip(last_task, 240)}\n")
    if last_llm:
        lines.append(f"last_llm_preview: {_clip(last_llm, 360)}\n")

    used = 0
    out: List[str] = [head]
    for ln in lines:
        if used + len(ln) > budget:
            break
        out.append(ln)
        used += len(ln)
    out.append(tail)
    return "".join(out)


__all__ = [
    "SynthEvent",
    "record_event",
    "snapshot_events",
    "clear_events",
    "build_event_stream_block",
    "build_event_state_block",
]
