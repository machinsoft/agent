from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional

from jinx.micro.llm.llm_cache import call_openai_cached

_JSON_MULTI_SAMPLES = 2
_JSON_MULTI_HEDGE_MS = 0


async def call_json_multi_validated(
    instructions: str,
    model: str,
    input_text: str,
    parse: Callable[[str], Optional[Dict[str, Any]]],
    validate: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    *,
    base_extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run multiple cached/coalesced LLM calls in parallel and return the first VALIDATED JSON dict.

    - Uses small temperature variations; only first request registers the family, others opt-out to avoid collapsing.
    - If none validate, returns the first successfully parsed dict (even if not validated), else None.
    - Respects global TTL/timeout/concurrency from llm_cache.
    """
    n = max(1, int(_JSON_MULTI_SAMPLES))
    temps_base = [0.2, 0.5, 0.8, 0.3]
    temps = temps_base[:max(1, n)]
    hedge_ms = int(_JSON_MULTI_HEDGE_MS)

    extra = dict(base_extra_kwargs or {})

    async def _one(t: float, register_family: bool) -> Optional[Dict[str, Any]]:
        kw = dict(extra)
        kw["temperature"] = t
        if not register_family:
            kw["__no_family__"] = True
        out = await call_openai_cached(instructions, model, input_text, extra_kwargs=kw)
        obj = parse(out) if out else None
        if not obj:
            return None
        good = validate(obj)
        return good or obj

    tasks: list[asyncio.Task] = []
    if not temps:
        temps = [0.2]
    # Start first immediately
    t0 = asyncio.create_task(_one(temps[0], True))
    tasks.append(t0)
    # Optional hedged start of second
    if len(temps) > 1 and hedge_ms > 0:
        try:
            await asyncio.wait_for(asyncio.sleep(max(0.0, hedge_ms) / 1000.0), timeout=max(0.05, hedge_ms / 1000.0))
        except Exception:
            pass
        if not t0.done():
            t1 = asyncio.create_task(_one(temps[1], False))
            tasks.append(t1)

    best: Optional[Dict[str, Any]] = None
    for fut in asyncio.as_completed(tasks):
        try:
            obj = await fut
        except Exception:
            continue
        if obj is None:
            continue
        # If already validated, validate() returned a dict with the normalized schema
        v = validate(obj) if obj is not None else None
        if v:
            return v
        if best is None:
            best = obj
    return best
