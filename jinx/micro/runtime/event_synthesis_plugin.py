from __future__ import annotations

from typing import Any, Dict, Tuple

from jinx.micro.event_synthesis.api import record_event
from jinx.micro.runtime.plugins import register_plugin, subscribe_event
from jinx.micro.runtime.task_ctx import get_current_group
from jinx.micro.runtime.api import on
from jinx.micro.runtime.contracts import (
    TASK_REQUEST,
    TASK_PROGRESS,
    TASK_RESULT,
    PROGRAM_SPAWN,
    PROGRAM_EXIT,
    PROGRAM_LOG,
)


def _infer_group(payload: Any) -> str:
    try:
        g = str((payload or {}).get("group") or "").strip()
        if g:
            return g
    except Exception:
        pass
    return get_current_group()


def _w(topic: str) -> float:
    if topic == "queue.intake":
        return 5.0
    if topic == "locator.results":
        return 2.5
    if topic == "turn.error":
        return 4.5
    if topic == "turn.metrics":
        return 1.2
    if topic == "turn.scheduled":
        return 2.0
    if topic == "turn.finished":
        return 1.0
    if topic == TASK_REQUEST:
        return 1.8
    if topic == TASK_PROGRESS:
        return 0.8
    if topic == TASK_RESULT:
        return 3.0
    if topic in {PROGRAM_SPAWN, PROGRAM_EXIT}:
        return 1.0
    if topic == PROGRAM_LOG:
        return 0.6
    return 1.0


async def _start(ctx) -> None:  # type: ignore[no-redef]
    task_meta: Dict[str, Tuple[str, str]] = {}

    async def _on_plugin_event(topic: str, payload: Any) -> None:
        g = _infer_group(payload)
        record_event(topic, payload, group=g, weight=_w(topic))

    subscribe_event("queue.intake", plugin="event_synthesis", callback=_on_plugin_event)
    subscribe_event("locator.results", plugin="event_synthesis", callback=_on_plugin_event)
    subscribe_event("turn.scheduled", plugin="event_synthesis", callback=_on_plugin_event)
    subscribe_event("turn.finished", plugin="event_synthesis", callback=_on_plugin_event)
    subscribe_event("turn.metrics", plugin="event_synthesis", callback=_on_plugin_event)
    subscribe_event("turn.error", plugin="event_synthesis", callback=_on_plugin_event)

    async def _on_task_request(topic: str, payload: Any) -> None:
        tid = str((payload or {}).get("id") or "")
        name = str((payload or {}).get("name") or "")
        g = get_current_group()
        if tid:
            task_meta[tid] = (name, g)
        record_event(topic, {"id": tid, "name": name}, group=g, weight=_w(topic))

    async def _on_task_progress(topic: str, payload: Any) -> None:
        tid = str((payload or {}).get("id") or "")
        pct = payload.get("pct")
        msg = str((payload or {}).get("msg") or "")
        name, g = task_meta.get(tid, ("", get_current_group()))
        record_event(topic, {"id": tid, "name": name, "pct": pct, "msg": msg}, group=g, weight=_w(topic))

    async def _on_task_result(topic: str, payload: Any) -> None:
        tid = str((payload or {}).get("id") or "")
        ok = bool((payload or {}).get("ok"))
        res = (payload or {}).get("result")
        err = str((payload or {}).get("error") or "")
        name, g = task_meta.pop(tid, ("", get_current_group()))
        record_event(topic, {"id": tid, "name": name, "ok": ok, "result": res, "error": err}, group=g, weight=_w(topic))

    async def _on_program_event(topic: str, payload: Any) -> None:
        pid = str((payload or {}).get("id") or "")
        name = str((payload or {}).get("name") or "")
        level = str((payload or {}).get("level") or "")
        msg = str((payload or {}).get("msg") or "")
        record_event(topic, {"id": pid, "name": name, "level": level, "msg": msg}, group=get_current_group(), weight=_w(topic))

    await on(TASK_REQUEST, _on_task_request)
    await on(TASK_PROGRESS, _on_task_progress)
    await on(TASK_RESULT, _on_task_result)
    await on(PROGRAM_SPAWN, _on_program_event)
    await on(PROGRAM_EXIT, _on_program_event)
    await on(PROGRAM_LOG, _on_program_event)


async def _stop(ctx) -> None:  # type: ignore[no-redef]
    return None


def register_event_synthesis_plugin() -> None:
    register_plugin(
        "event_synthesis",
        start=_start,
        stop=_stop,
        enabled=True,
        priority=25,
        version="1.0.0",
        deps=[],
        features={"event_synthesis"},
    )


__all__ = [
    "register_event_synthesis_plugin",
]
