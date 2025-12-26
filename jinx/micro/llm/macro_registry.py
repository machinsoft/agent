from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

try:
    from jinx.logger.file_logger import append_line as _append
    from jinx.log_paths import BLUE_WHISPERS
except Exception:  # pragma: no cover
    _append = None
    BLUE_WHISPERS = ""


@dataclass
class MacroContext:
    key: str
    anchors: Dict[str, List[str]]
    programs: List[str]
    os_name: str
    py_ver: str
    cwd: str
    now_iso: str
    now_epoch: str
    input_text: str = ""

    def env(self, name: str, default: str = "") -> str:
        return default


_Handler = Callable[[List[str], MacroContext], Awaitable[str]]

_REGISTRY: Dict[str, _Handler] = {}
_LOCK = asyncio.Lock()
_GEN_RE = re.compile(r"\{\{m:([a-zA-Z0-9_]+)((?::[^{}:\s]+)*)\}\}")


async def register_macro(namespace: str, handler: _Handler) -> None:
    ns = (namespace or "").strip().lower()
    if not ns:
        return
    async with _LOCK:
        _REGISTRY[ns] = handler


async def list_namespaces() -> List[str]:
    async with _LOCK:
        return sorted(_REGISTRY.keys())


async def expand_dynamic_macros(text: str, ctx: MacroContext, *, max_expansions: int = 50) -> str:
    """Expand dynamic macros concurrently with per-call deduplication.

    - Collects all macro occurrences first (up to max_expansions) to avoid repeated scans.
    - Deduplicates identical (namespace, args) pairs and runs handlers concurrently under a
      small semaphore to keep latency low while respecting RT constraints.
    - Assembles the final text in a single pass preserving original order.
    """
    if not text or not isinstance(text, str):
        return text
    if "{{m:" not in text:
        return text

    # 1) Find occurrences once
    occ: List[Tuple[int, int, str, Tuple[str, ...]]] = []  # (start, end, ns, args)
    pos = 0
    while True:
        m = _GEN_RE.search(text, pos)
        if not m:
            break
        ns = (m.group(1) or "").strip().lower()
        args_blob = m.group(2) or ""
        args = tuple(a for a in args_blob.split(":") if a)
        occ.append((m.start(), m.end(), ns, args))
        pos = m.end()
        if len(occ) >= max_expansions:
            break
    if not occ:
        return text

    # 2) Snapshot registry once
    async with _LOCK:
        reg = dict(_REGISTRY)

    # 3) Deduplicate macro calls for this pass
    uniq_keys: List[Tuple[str, Tuple[str, ...]]] = []
    seen: set[Tuple[str, Tuple[str, ...]]] = set()
    for _, _, ns, args in occ:
        key = (ns, args)
        if key not in seen:
            seen.add(key)
            uniq_keys.append(key)

    # 4) Run handlers concurrently under a small semaphore
    sem = asyncio.Semaphore(4)

    results: Dict[Tuple[str, Tuple[str, ...]], str] = {}

    async def _run_one(ns: str, args: Tuple[str, ...]) -> None:
        h = reg.get(ns)
        if not h:
            results[(ns, args)] = ""
            return
        try:
            async with sem:
                val = await h(list(args), ctx)
            results[(ns, args)] = val or ""
        except Exception as e:
            results[(ns, args)] = ""
            if _append:
                return

    await asyncio.gather(*[asyncio.create_task(_run_one(ns, args)) for ns, args in uniq_keys])

    # 5) Assemble final text preserving order
    out_parts: List[str] = []
    last = 0
    for start, end, ns, args in occ:
        out_parts.append(text[last:start])
        out_parts.append(results.get((ns, args), ""))
        last = end
    out_parts.append(text[last:])
    return "".join(out_parts)
