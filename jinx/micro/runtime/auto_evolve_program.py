from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, Optional

from jinx.micro.runtime.program import MicroProgram
from jinx.micro.runtime.contracts import TASK_REQUEST

_AUTOEVOLVE_CONC = 3
_AUTOEVOLVE_BUDGET_MS = 1600
_AUTOFIX_TRIES = 3


class AutoEvolveProgram(MicroProgram):
    """Autonomous self-repair and self-improvement orchestrator.

    Triggers:
    - plugin event "queue.intake": attempt auto action/patch for code-mod tasks
    - plugin event "turn.error": parse traceback, run autofix ladder
    - plugin event "auto.patch.report": escalate failed patches via autofix ladder
    - task "autoevolve.request": external request to attempt auto action

    All operations are strictly time-bounded and concurrency-limited.
    """

    def __init__(self) -> None:
        super().__init__(name="AutoEvolveProgram")
        self._sem = asyncio.Semaphore(_AUTOEVOLVE_CONC)
        self._budget_ms = _AUTOEVOLVE_BUDGET_MS
        # Track limited retries for a given query when acquiring skills
        self._skill_retry: dict[str, int] = {}

    async def run(self) -> None:
        # Subscribe to TASK bus
        from jinx.micro.runtime.api import on as _on
        await _on(TASK_REQUEST, self._on_task)
        # Subscribe to plugin events
        try:
            from jinx.micro.runtime.plugins import subscribe_event as _sub
            _sub("queue.intake", plugin="auto_evolve", callback=self._on_queue_intake)
            _sub("turn.error", plugin="auto_evolve", callback=self._on_turn_error)
            _sub("auto.patch.report", plugin="auto_evolve", callback=self._on_patch_report)
            _sub("auto.skill_acquired", plugin="auto_evolve", callback=self._on_skill_acquired)
        except Exception:
            pass
        await self.log("auto-evolve online")
        # Idle loop
        while True:
            await asyncio.sleep(1.0)

    async def _on_task(self, topic: str, payload: Dict[str, Any]) -> None:
        name = str(payload.get("name") or "").strip()
        tid = str(payload.get("id") or "").strip()
        if not tid:
            return
        if name != "autoevolve.request":
            return
        kw = payload.get("kwargs") or {}
        q = str(kw.get("query") or "").strip()
        await self._attempt_auto_action(q)

    # === Event handlers ===
    async def _on_queue_intake(self, _topic: str, payload: Dict[str, Any]) -> None:
        try:
            txt = str((payload or {}).get("text") or "").strip()
            if not txt:
                return
            # Do not block intake; try in background
            asyncio.create_task(self._attempt_auto_action(txt))
        except Exception:
            pass

    async def _on_turn_error(self, _topic: str, payload: Dict[str, Any]) -> None:
        # Parse traceback to get candidate file; then schedule autofix ladder
        tb = str((payload or {}).get("tb") or "")
        q = str((payload or {}).get("error") or "")
        path = self._extract_path_from_traceback(tb)
        try:
            from jinx.micro.runtime.api import submit_task as _submit
            await _submit(
                "autofix.retry",
                path=path,
                query=q or (f"fix error: {path}" if path else "fix error"),
                tries=_AUTOFIX_TRIES,
            )
        except Exception:
            pass

    async def _on_patch_report(self, _topic: str, payload: Dict[str, Any]) -> None:
        # Escalate failed auto patches via autofix ladder
        if not isinstance(payload, dict):
            return
        ok = bool(payload.get("success"))
        if ok:
            return
        file_path = str(payload.get("file") or "") or None
        msg = str(payload.get("message") or "")
        try:
            from jinx.micro.runtime.api import submit_task as _submit
            await _submit(
                "autofix.retry",
                path=file_path,
                query=msg or "repair failed patch",
                tries=_AUTOFIX_TRIES,
            )
        except Exception:
            pass

    async def _on_skill_acquired(self, _topic: str, payload: Dict[str, Any]) -> None:
        # Retry the original query once skill acquired
        if not isinstance(payload, dict):
            return
        q = str(payload.get("query") or "").strip()
        if not q:
            return
        # At most 2 retries to avoid loops
        n = int(self._skill_retry.get(q, 0))
        if n >= 2:
            return
        self._skill_retry[q] = n + 1
        # Retry with reduced budget to maintain RT
        try:
            await self._attempt_auto_action(q)
        except Exception:
            pass

    # === Core actions ===
    async def _attempt_auto_action(self, query: str) -> None:
        if not query:
            return
        async with self._sem:
            try:
                from jinx.micro.runtime.action_router import auto_route_and_execute as _auto
                async def _call():
                    return await _auto(query, budget_ms=max(300, self._budget_ms))
                rep = await asyncio.wait_for(_call(), timeout=(self._budget_ms + 400) / 1000.0)
                # Escalate capability gap: non-modifying/low-conf or no file
                reason = str((rep or {}).get("reason") or "")
                executed = bool((rep or {}).get("executed"))
                if (not executed) and reason in {"non_modifying_or_low_conf", "no_file_resolved"}:
                    try:
                        from jinx.micro.runtime.api import submit_task as _submit
                        await _submit("skill.acquire", query=query)
                        # track for a future retry when skill arrives
                        self._skill_retry[query] = int(self._skill_retry.get(query, 0))
                    except Exception:
                        pass
            except Exception:
                pass

    # === Utilities ===
    def _extract_path_from_traceback(self, tb: str) -> Optional[str]:
        if not tb:
            return None
        # Match: File "E:\\agent\\path\\to\\file.py", line N
        m = re.search(r'File \"([^\"]+\.py)\", line \d+', tb)
        if not m:
            return None
        p = m.group(1)
        # Only consider files under current working dir to avoid system paths
        try:
            cwd = os.getcwd()
            if p.lower().startswith(cwd.lower()):
                return p
        except Exception:
            pass
        return None


async def spawn_auto_evolve() -> str:
    from jinx.micro.runtime.api import spawn as _spawn
    return await _spawn(AutoEvolveProgram())
