from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, Optional

from jinx.micro.runtime.program import MicroProgram
from jinx.micro.runtime.contracts import TASK_REQUEST
from jinx.micro.runtime.patch import AutoPatchArgs
from jinx.micro.runtime.patch.autopatch import autopatch as _autopatch

_AUTOFIX_TRIES = 3


def _derive_query(code: str) -> str:
    s = (code or "").strip()
    # Prefer function/class name or first callable identifier
    m = re.search(r"\b(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", s)
    if m:
        return m.group(2)
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", s)
    if m:
        return m.group(1)
    # fallback: first identifier
    m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", s)
    return m.group(1) if m else ""


class AutoFixProgram(MicroProgram):
    """Reactive auto-fix program for failed patches.

    Handles TASK_REQUEST "autofix.retry" with payload kwargs:
      { path?: str, code?: str, symbol?: str, query?: str, tries?: int }

    Strategy: attempt a small ladder of autopatch_ex variants under strict RT budgets.
    """

    def __init__(self) -> None:
        super().__init__(name="AutoFixProgram")

    async def run(self) -> None:
        from jinx.micro.runtime.api import on as _on
        await _on(TASK_REQUEST, self._on_task)
        await self.log("autofix online")
        while True:
            await asyncio.sleep(1.0)

    async def _on_task(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            name = str(payload.get("name") or "")
            tid = str(payload.get("id") or "")
            if name != "autofix.retry" or not tid:
                return
            kw = payload.get("kwargs") or {}
            await self._handle_retry(tid, kw)
        except Exception:
            pass

    async def _handle_retry(self, tid: str, kw: Dict[str, Any]) -> None:
        path = str(kw.get("path") or "") or None
        code = str(kw.get("code") or "") or None
        symbol = str(kw.get("symbol") or "") or None
        query = str(kw.get("query") or "") or None
        tries = 0
        try:
            tries = int(kw.get("tries") or _AUTOFIX_TRIES)
        except Exception:
            tries = _AUTOFIX_TRIES
        tries = max(1, min(6, tries))
        from jinx.micro.runtime.api import report_progress as _report_progress, report_result as _report_result
        await _report_progress(tid, 10.0, "autofix start")
        q = query or _derive_query(code or "")
        attempts = []
        # assemble ladder (prefer in-file semantic, then symbol path, then global symbol)
        attempts.append(AutoPatchArgs(path=path, code=code, query=q or None, preview=False, force=True))
        attempts.append(AutoPatchArgs(path=path, code=code, symbol=symbol or q or None, preview=False, force=True))
        attempts.append(AutoPatchArgs(path=None, code=code, symbol=symbol or q or None, query=q or None, preview=False, force=True))
        # cap attempts by tries
        attempts = attempts[:tries]
        for i, a in enumerate(attempts, start=1):
            await _report_progress(tid, 20.0 + i * (60.0 / max(1, len(attempts))), f"autofix try {i}/{len(attempts)}")
            ok, strat, detail = await _autopatch(a)
            if ok:
                await _report_result(tid, True, {"strategy": strat, "diff": detail, "attempt": i})
                return
        await _report_result(tid, False, error="autofix exhausted")


async def spawn_autofix() -> str:
    from jinx.micro.runtime.api import spawn as _spawn
    return await _spawn(AutoFixProgram())
