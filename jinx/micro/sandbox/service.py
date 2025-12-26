from __future__ import annotations

from typing import Callable, Awaitable, Optional, Tuple
import asyncio as _asyncio
import time as _time
from threading import Event as _TEvent, Lock as _TLock, BoundedSemaphore as _TBoundedSem

from jinx.sandbox.executor import blast_zone
from jinx.sandbox.async_runner import run_sandbox
from jinx.micro.sandbox.normalizer import code_key as _code_key
from jinx.micro.runtime.autobrain_config import (
    get_int as _brain_int,
    record_outcome as _brain_record,
    TaskContext as _TaskCtx,
)


__all__ = ["blast_zone", "arcane_sandbox"]

# Cross-loop coalescing for identical code runs
_SBX_LOCK: _TLock = _TLock()
_SBX_STATE: dict[str, Tuple[_TEvent, Optional[str], list[Tuple[_asyncio.AbstractEventLoop, Callable[[Optional[str]], Awaitable[None]]]]]] = {}

# Adaptive concurrency from AutoBrain
def _get_sbx_conc() -> int:
    return _brain_int("sandbox_conc")

# Initial semaphore (will be recreated dynamically if needed)
_SBX_CONC_CURRENT = 2
_SBX_TSEM: _TBoundedSem = _TBoundedSem(_SBX_CONC_CURRENT)

# Short TTL cache to avoid immediate re-runs of identical code
_SBX_TTL_MS = 1000
_SBX_CACHE: dict[str, Tuple[int, Optional[str]]] = {}


def _now_ms() -> int:
    try:
        return int(_time.monotonic_ns() // 1_000_000)
    except Exception:
        return int(_time.time() * 1000)


async def arcane_sandbox(c: str, call: Callable[[str | None], Awaitable[None]] | None = None) -> None:
    """Run code in a separate process and surface results asynchronously.

    Coalesces concurrent requests for identical code across threads/event-loops,
    ensuring only one sandbox process runs and all requesters receive the same
    completion signal and callback.
    """
    # Stable structural key (AST-based when possible)
    key = _code_key(c or "")
    loop = _asyncio.get_running_loop()

    # TTL cache: if recent identical run finished, deliver cached result immediately
    if _SBX_TTL_MS > 0:
        with _SBX_LOCK:
            ent = _SBX_CACHE.get(key)
        if ent is not None and ent[0] >= _now_ms():
            if call is not None:
                try:
                    await call(ent[1])
                except Exception:
                    pass
            return

    # Register waiter and possibly start the run
    first: bool = False
    with _SBX_LOCK:
        st = _SBX_STATE.get(key)
        if st is None:
            evt = _TEvent()
            callbacks: list[Tuple[_asyncio.AbstractEventLoop, Callable[[Optional[str]], Awaitable[None]]]] = []
            if call is not None:
                callbacks.append((loop, call))
            _SBX_STATE[key] = (evt, None, callbacks)
            first = True
            evt_to_wait = evt
        else:
            evt, err, callbacks = st
            if call is not None:
                callbacks.append((loop, call))
            _SBX_STATE[key] = (evt, err, callbacks)
            evt_to_wait = evt

    async def _runner() -> None:
        # Local holder to capture error from run_sandbox
        holder: dict[str, Optional[str]] = {"err": None}
        t0 = _time.perf_counter()

        async def _cb(e: Optional[str]) -> None:
            holder["err"] = e

        # Execute sandbox exactly once under cross-loop concurrency limiter
        acquired = False
        try:
            await _asyncio.to_thread(_SBX_TSEM.acquire)
            acquired = True
            await run_sandbox(c, _cb)
        finally:
            if acquired:
                try:
                    _SBX_TSEM.release()
                except Exception:
                    pass
            # Report outcome to AutoBrain for learning
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            success = holder["err"] is None
            try:
                _brain_record("sandbox_conc", success, elapsed_ms)
            except Exception:
                pass

        # Fan out callbacks and signal waiters
        with _SBX_LOCK:
            evt, _prev_err, callbacks = _SBX_STATE.get(key, (_TEvent(), None, []))
            # Update stored error for late readers
            _SBX_STATE[key] = (evt, holder["err"], callbacks)
            evt.set()
            # Snapshot callbacks and delete state to allow future identical runs
            cb_snapshot = list(callbacks)
            del _SBX_STATE[key]
            # Update TTL cache
            if _SBX_TTL_MS > 0:
                _SBX_CACHE[key] = (_now_ms() + _SBX_TTL_MS, holder["err"])

        # Schedule callbacks in their original loops
        for lp, cb in cb_snapshot:
            try:
                _asyncio.run_coroutine_threadsafe(cb(holder["err"]), lp)
            except Exception:
                pass

    if first:
        # Start the single runner task in this loop
        _asyncio.create_task(_runner())
    # Await completion (cross-loop safe via thread Event)
    await _asyncio.to_thread(evt_to_wait.wait)
    return
