"""Action Router - fully automated decision and execution layer.

Goals:
- Zero user toggles: defaults ON (see autoconfig)
- Language-agnostic classification (via ml_orchestrator)
- Resource resolution (via resource_locator)
- Safe, time-bounded auto-execution of code-modifying intents
- Publish outcome and expose compact report for UI/LLM consumption
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional


_CODE_MOD_TASKS = {
    "implementation",
    "refactoring",
    "debugging",
}

_ARCH_TASKS = {
    "architecture",
    "api_architecture",
    "design_api",
    "api_design",
}


async def _synth_arch_spec(query: str, *, budget_ms: int = 900) -> Optional[Dict[str, Any]]:
    """Synthesize a compact API spec from free-form query using LLM.

    Returns a dict like {"name": str, "resources": [{"name": str, "fields": {...}, "endpoints": [...]}, ...]}
    """
    try:
        from jinx.micro.llm.service import spark_openai as _spark
    except Exception:
        return None
    try:
        from jinx.prompts import render_prompt as _render_prompt
        shape = (
            '{\n  "name": "string",\n  "resources": [\n    {\n      "name": "string",\n      "fields": {"id": "int|str|float|bool", ...},\n      "endpoints": ["list", "get", "create", "update", "delete"]\n    }\n  ]\n}'
        )
        # Minimal project context
        from jinx.micro.llm.prompting import derive_basic_context as _derive
        pname, resources = _derive()
        import json as _json
        prompt = _render_prompt(
            "architect_api",
            shape=shape,
            project_name=pname,
            candidate_resources_json=_json.dumps(resources),
            request=query,
        )
    except Exception:
        prompt = f"Request: {query}"
    try:
        out, _ = await asyncio.wait_for(_spark(prompt), timeout=max(0.3, budget_ms) / 1000.0)
    except Exception:
        return None
    if not out or not isinstance(out, str):
        return None
    # Extract JSON substring
    try:
        import re as _re
        m = _re.search(r"\{[\s\S]*\}", out)
        s = m.group(0) if m else out.strip()
        obj = json.loads(s)
        if isinstance(obj, dict) and obj.get("resources"):
            return obj  # type: ignore[return-value]
    except Exception:
        return None
    return None


async def auto_route_and_execute(query: str, *, budget_ms: int = 1500) -> Dict[str, Any]:
    """Route an arbitrary query and auto-execute if safe and confident.

    Returns a dict report:
      {
        'executed': bool,
        'reason': str,
        'task_type': str | None,
        'confidence': float,
        'file': str | None,
        'rel': str | None,
        'patch_success': bool | None,
        'message': str,
      }
    """
    query = (query or "").strip()
    if not query:
        return {
            "executed": False,
            "reason": "empty_query",
            "task_type": None,
            "confidence": 0.0,
            "file": None,
            "rel": None,
            "patch_success": None,
            "message": "",
        }

    # Auto action is always enabled for autonomous operation

    # Classification via ML orchestrator
    try:
        from jinx.micro.runtime.ml_orchestrator import get_ml_orchestrator
        ml = await get_ml_orchestrator()
        pred = await ml.predict_task(query, explain=False)
        task_type = str(pred.get("task_type") or "").strip().lower()
        confidence = float(pred.get("confidence") or 0.0)
    except Exception:
        task_type = ""
        confidence = 0.0

    # Architecture/design tasks: route to API architect micro-program
    arch_conf_min = 0.55
    if task_type in _ARCH_TASKS and confidence >= arch_conf_min:
        # Try to synthesize a compact spec; if fail, submit with None (program will default)
        try:
            spec = await _synth_arch_spec(query, budget_ms=900)
        except Exception:
            spec = None
        # Submit task to API architect
        try:
            from jinx.micro.runtime.api import submit_task as _submit
            await _submit("architect.api", spec=spec, framework="fastapi", budget_ms=1600)
        except Exception:
            pass
        try:
            from jinx.micro.runtime.plugins import publish_event
            publish_event("auto.arch.requested", {"task_type": task_type, "confidence": confidence})
        except Exception:
            pass
        return {
            "executed": True,
            "reason": "arch_submitted",
            "task_type": task_type or None,
            "confidence": confidence,
            "file": None,
            "rel": None,
            "patch_success": None,
            "message": "api architecture generation submitted",
        }

    # Decision: only code-modifying tasks with reasonable confidence
    if task_type not in _CODE_MOD_TASKS or confidence < 0.55:
        # Skill fallback: try to execute a lightweight skill to satisfy the query
        try:
            from jinx.micro.runtime.skills import try_execute as _skill_try
            s_out = await _skill_try(query, budget_ms=600)
        except Exception:
            s_out = None
        if s_out and s_out.strip():
            try:
                from jinx.micro.runtime.plugins import publish_event
                publish_event("skill.executed", {"query": query})
            except Exception:
                pass
            return {
                "executed": True,
                "reason": "skill_executed",
                "task_type": task_type or None,
                "confidence": confidence,
                "file": None,
                "rel": None,
                "patch_success": None,
                "message": s_out.strip(),
            }
        return {
            "executed": False,
            "reason": "non_modifying_or_low_conf",
            "task_type": task_type or None,
            "confidence": confidence,
            "file": None,
            "rel": None,
            "patch_success": None,
            "message": "",
        }

    # Resolve primary resource
    rel = None
    file_path = None
    try:
        # Prefer orchestrator-resolved files if present
        primary = (pred.get("primary_resource") or {}) if isinstance(pred, dict) else {}
        rel = str(primary.get("rel") or "")
        file_path = str(primary.get("path") or "")
        if not file_path:
            # Fallback: use resource locator directly
            from jinx.micro.runtime.resource_locator import get_resource_locator
            loc = await get_resource_locator()
            located = await loc.locate(query, k=1, budget_ms=180)
            if located:
                rel = located[0].rel
                file_path = located[0].path
    except Exception:
        file_path = None

    if not file_path:
        # Skill fallback on unresolved file
        try:
            from jinx.micro.runtime.skills import try_execute as _skill_try
            s_out = await _skill_try(query, budget_ms=600)
        except Exception:
            s_out = None
        if s_out and s_out.strip():
            try:
                from jinx.micro.runtime.plugins import publish_event
                publish_event("skill.executed", {"query": query})
            except Exception:
                pass
            return {
                "executed": True,
                "reason": "skill_executed",
                "task_type": task_type or None,
                "confidence": confidence,
                "file": None,
                "rel": None,
                "patch_success": None,
                "message": s_out.strip(),
            }
        return {
            "executed": False,
            "reason": "no_file_resolved",
            "task_type": task_type or None,
            "confidence": confidence,
            "file": None,
            "rel": None,
            "patch_success": None,
            "message": "",
        }

    # Guard: avoid restricted/internal paths
    try:
        from jinx.micro.common.internal_paths import is_restricted_path as _is_restricted
        if _is_restricted(str(rel or "")):
            return {
                "executed": False,
                "reason": "restricted_path",
                "task_type": task_type or None,
                "confidence": confidence,
                "file": file_path,
                "rel": rel,
                "patch_success": None,
                "message": "",
            }
    except Exception:
        pass

    # Execute patch with time budget
    try:
        from jinx.micro.code.patch_orchestrator import PatchOrchestrator
        po = PatchOrchestrator()
        # Run under timeout to respect RT constraints
        async def _run():
            return await po.apply_from_description(description=query, file_path=file_path)
        report = await asyncio.wait_for(_run(), timeout=max(0.2, budget_ms) / 1000.0)
        ok = bool(report.success)
        msg = str(report.message or "")
    except asyncio.TimeoutError:
        ok = False
        msg = "auto patch timeout"
    except Exception as e:
        ok = False
        msg = f"auto patch error: {e}"

    # Publish and store
    try:
        from jinx.micro.runtime.plugins import publish_event
        publish_event("auto.patch.report", {
            "query": query,
            "task_type": task_type,
            "confidence": confidence,
            "file": file_path,
            "rel": rel,
            "success": ok,
            "message": msg,
        })
    except Exception:
        pass

    try:
        import jinx.state as jx_state
        setattr(jx_state, "auto_last_patch_report", {
            "task_type": task_type,
            "confidence": confidence,
            "file": file_path,
            "rel": rel,
            "success": ok,
            "message": msg,
        })
    except Exception:
        pass

    return {
        "executed": True,
        "reason": "executed",
        "task_type": task_type or None,
        "confidence": confidence,
        "file": file_path,
        "rel": rel,
        "patch_success": ok,
        "message": msg,
    }


__all__ = [
    "auto_route_and_execute",
]
