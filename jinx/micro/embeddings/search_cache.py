from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Tuple, Optional

from .project_search_api import search_project as _search_project

# AutoBrain adaptive configuration
try:
    from jinx.micro.runtime.autobrain_config import (
        get_int as _ab_int,
        record_outcome as _ab_record,
    )
    _AB_OK = True
except Exception:
    _AB_OK = False
    def _ab_int(n, c=None): return 4
    def _ab_record(n, s, l=0, c=None): pass

# Lightweight in-memory TTL cache + request coalescing for project search
# Goal: avoid duplicate concurrent searches and reduce repeated work bursts

_TTL_SEC = 5.0
_MAX_CONC = 4
_DUMP = False

_mem: Dict[Tuple[str, int, Optional[int]], Tuple[float, List[Dict[str, Any]]]] = {}
_inflight: Dict[Tuple[str, int, Optional[int]], asyncio.Future] = {}
_sem = asyncio.Semaphore(max(1, _MAX_CONC))


def _now() -> float:
    return time.time()


async def _dump_line(line: str) -> None:
    if not _DUMP:
        return
    try:
        from jinx.logger.file_logger import append_line as _append
        from jinx.log_paths import BLUE_WHISPERS
        await _append(BLUE_WHISPERS, f"[search_cache] {line}")
    except Exception:
        pass


def _key(query: str, k: int, max_time_ms: Optional[int]) -> Tuple[str, int, Optional[int]]:
    q = (query or "").strip()
    return (q, int(max(1, k)), int(max_time_ms) if (max_time_ms is not None) else None)


def _cache_get(key: Tuple[str, int, Optional[int]]) -> Optional[List[Dict[str, Any]]]:
    v = _mem.get(key)
    if not v:
        return None
    exp, data = v
    if exp < _now():
        _mem.pop(key, None)
        return None
    return data


def _cache_put(key: Tuple[str, int, Optional[int]], data: List[Dict[str, Any]]) -> None:
    _mem[key] = (_now() + max(0.5, _TTL_SEC), data)


async def search_project_cached(query: str, *, k: int | None = None, max_time_ms: int | None = 300) -> List[Dict[str, Any]]:
    """Cached/coalesced wrapper over project_search_api.search_project.

    - Coalesces identical concurrent (query, k, max_time_ms) calls.
    - TTL cache for a short period to avoid re-running heavy stages repeatedly.
    - Concurrency limited via semaphore.
    """
    k_eff = int(k) if (k is not None) else None
    if k_eff is None:
        # Defer to underlying default later; but for cache key we need a number; assume 20 default in config
        try:
            k_eff = int(os.getenv("EMBED_PROJECT_TOP_K", "20"))
        except Exception:
            k_eff = 20
    key = _key(query, k_eff, max_time_ms)
    c = _cache_get(key)
    if c is not None:
        return c

    fut = _inflight.get(key)
    if fut is not None:
        try:
            return await fut
        except Exception:
            return []

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _inflight[key] = fut
    try:
        async with _sem:
            await _dump_line(f"call qlen={len((query or '').strip())} k={k_eff} t={max_time_ms}")
            # Delegate; it is already async and internally time-bounded by stages
            res = await _search_project(query, k=k_eff, max_time_ms=max_time_ms)
        _cache_put(key, res)
        fut.set_result(res)
        return res
    except Exception as e:
        try:
            fut.set_result([])
        except Exception:
            pass
        return []
    finally:
        _inflight.pop(key, None)
