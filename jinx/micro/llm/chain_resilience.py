from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict

from jinx.log_paths import CHAIN_STATE

_DEFAULT_DISABLE_MS = 60_000  # 1 minute
_FAIL_THRESHOLD = 3
_DISABLE_MS = _DEFAULT_DISABLE_MS


async def _read_state() -> Dict[str, Any]:
    try:
        def _load() -> Dict[str, Any]:
            if not os.path.exists(CHAIN_STATE):
                return {}
            with open(CHAIN_STATE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        return await asyncio.to_thread(_load)
    except Exception:
        return {}


async def _write_state(st: Dict[str, Any]) -> None:
    try:
        def _dump() -> None:
            d = os.path.dirname(CHAIN_STATE) or "."
            os.makedirs(d, exist_ok=True)
            with open(CHAIN_STATE, "w", encoding="utf-8") as f:
                json.dump(st, f)
        await asyncio.to_thread(_dump)
    except Exception:
        return


async def allow_execution(now_ms: int | None = None) -> bool:
    """Return whether the chained planner is allowed to run at this moment.

    Considers a persistent disable window set after repeated failures.
    """
    st = await _read_state()
    if not isinstance(st, dict):
        return True
    try:
        until = int(st.get("disable_until_ms") or 0)
    except Exception:
        until = 0
    now = int(now_ms or time.time() * 1000)
    return now >= until


async def record_success() -> None:
    st = await _read_state()
    if not isinstance(st, dict):
        st = {}
    # Decay the failure count on success
    try:
        fc = int(st.get("fail_count") or 0)
    except Exception:
        fc = 0
    st["fail_count"] = max(0, fc - 1)
    # Clear disable window if present and expired
    st.pop("disable_until_ms", None)
    await _write_state(st)


async def record_failure(kind: str, *, now_ms: int | None = None) -> None:
    """Record a failure and possibly set a temporary disable window."""
    st = await _read_state()
    if not isinstance(st, dict):
        st = {}
    try:
        fc = int(st.get("fail_count") or 0)
    except Exception:
        fc = 0
    fc += 1
    st["fail_count"] = fc
    # Optional rolling stats by kind
    kinds = st.get("kinds") or {}
    kinds[kind] = int(kinds.get(kind) or 0) + 1
    st["kinds"] = kinds
    if fc >= _FAIL_THRESHOLD:
        now = int(now_ms or time.time() * 1000)
        st["disable_until_ms"] = now + _DISABLE_MS
        # Reset counter after disabling to allow recovery on next window
        st["fail_count"] = 0
    await _write_state(st)


async def save_last_plan(plan: Dict[str, Any]) -> None:
    """Persist the last successful planner output for fallback use."""
    if not isinstance(plan, dict):
        return
    st = await _read_state()
    if not isinstance(st, dict):
        st = {}
    # Keep a compact copy
    keep = {
        "goal": plan.get("goal"),
        "plan": plan.get("plan"),
        "sub_queries": plan.get("sub_queries"),
        "risks": plan.get("risks"),
        "note": plan.get("note"),
        "cortex": plan.get("cortex", {}),
        "ts": int(time.time() * 1000),
    }
    st["last_plan"] = keep
    await _write_state(st)


async def load_last_plan() -> Dict[str, Any] | None:
    """Load the last successful planner output if present."""
    st = await _read_state()
    if not isinstance(st, dict):
        return None
    lp = st.get("last_plan")
    return lp if isinstance(lp, dict) else None
