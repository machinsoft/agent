from __future__ import annotations

import os
import time
from typing import Any, Dict

from jinx.async_utils.fs import write_text
from jinx.micro.embeddings.project_config import ROOT as PROJECT_ROOT

_BRAIN_DIR = os.path.join(".jinx", "brain")
_MEMORY_DIR = os.path.join(".jinx", "memory")


def _brain_dir() -> str:
    return _BRAIN_DIR


def _memory_dir() -> str:
    return _MEMORY_DIR


def _safe_name(s: str, n: int = 40) -> str:
    s = (s or "").strip().lower()
    for ch in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return (s[:n] + ("" if len(s) <= n else "..")) or "untitled"


async def persist_brain(user_text: str, plan: Dict[str, Any], merged_context: str) -> None:
    """Persist the current brain (plan + merged context) under project root.

    Files are written to `${PROJECT_ROOT}/${JINX_BRAIN_DIR}/<ts>_<slug>.md`.
    Project embeddings will naturally ingest these `.md` files (since ROOT includes `.md`).
    """
    try:
        root = PROJECT_ROOT
        rel_dir = _brain_dir()
        out_dir = os.path.join(root, rel_dir)
        ts_ms = int(time.time() * 1000)
        goal = str(plan.get("goal") or "").strip()
        slug = _safe_name(goal or (user_text or ""))
        fname = f"{ts_ms}_{slug}.md"
        path = os.path.join(out_dir, fname)
        # Compose content (markdown)
        lines = []
        lines.append(f"---")
        lines.append(f"ts_ms: {ts_ms}")
        if goal:
            lines.append(f"goal: {goal}")
        steps = plan.get("plan") or []
        try:
            if isinstance(steps, list) and steps:
                lines.append("steps:")
                for i, st in enumerate(steps, start=1):
                    if not isinstance(st, dict):
                        continue
                    s = str(st.get("step") or "").strip()
                    w = str(st.get("why") or "").strip()
                    c = str(st.get("criteria") or "").strip()
                    if s:
                        lines.append(f"  - step: {s}")
                    if w:
                        lines.append(f"    why: {w}")
                    if c:
                        lines.append(f"    criteria: {c}")
        except Exception:
            pass
        subs = plan.get("sub_queries") or []
        if isinstance(subs, list) and subs:
            lines.append("subs:")
            for sq in subs:
                sqs = str(sq or "").strip()
                if sqs:
                    lines.append(f"  - {sqs}")
        risks = plan.get("risks") or []
        if isinstance(risks, list) and risks:
            lines.append("risks:")
            for rk in risks:
                rks = str(rk or "").strip()
                if rks:
                    lines.append(f"  - {rks}")
        note = str(plan.get("note") or "").strip()
        if note:
            lines.append(f"note: {note}")
        cortex = plan.get("cortex") or {}
        if isinstance(cortex, dict) and cortex:
            lines.append("cortex:")
            for ck, cv in cortex.items():
                cvs = str(cv or "").strip()
                if cvs:
                    lines.append(f"  {ck}: {cvs}")
        utxt = (user_text or "").strip()
        if utxt:
            lines.append(f"user: |\n  {utxt}")
        lines.append(f"---\n")
        if merged_context and merged_context.strip():
            lines.append(merged_context.strip())
        body = "\n".join(lines) + "\n"
        await write_text(path, body)
    except Exception:
        # Non-fatal persistence
        return


async def persist_memory(mem_text: str, evergreen_text: str, *, user_text: str = "", plan_goal: str = "") -> None:
    """Persist memory snapshot (sanitized transcript + evergreen) as a markdown file.

    Files are written to `${PROJECT_ROOT}/${JINX_MEMORY_DIR}/<ts>_<slug>.md`.
    Project embeddings (emb/) will ingest these and allow future queries to recall
    forgotten plans or user intents without polluting primary goals.
    """
    try:
        root = PROJECT_ROOT
        rel_dir = _memory_dir()
        out_dir = os.path.join(root, rel_dir)
        ts_ms = int(time.time() * 1000)
        slug = _safe_name(plan_goal or user_text or (mem_text[:40] if mem_text else ""))
        fname = f"{ts_ms}_{slug}.md"
        path = os.path.join(out_dir, fname)
        import re as _re

        def _strip_wrapped(tag: str, text: str) -> str:
            t = (text or "").strip()
            if not t:
                return ""
            # Extract inner if wrapped by <tag>...</tag>
            m = _re.search(rf"<\s*{tag}[^>]*>([\s\S]*?)</\s*{tag}\s*>", t, _re.IGNORECASE)
            return (m.group(1).strip() if m else t)

        mem_raw = _strip_wrapped("memory", mem_text)
        evg_raw = _strip_wrapped("evergreen", evergreen_text)

        # Skip persisting pure inactivity events to avoid memory spam
        if (user_text or "").strip() == "<no_response>" and not mem_raw and not evg_raw:
            return

        lines = []
        lines.append("---")
        lines.append(f"ts_ms: {ts_ms}")
        if plan_goal:
            lines.append(f"goal: {plan_goal}")
        # Avoid recording '<no_response>' as user text in memory snapshots
        if user_text and (user_text.strip() != "<no_response>"):
            lines.append(f"user: |\n  {user_text}")
        if mem_raw:
            lines.append("memory: |")
            for ln in (mem_raw.splitlines() or [mem_raw]):
                lines.append(f"  {ln}")
        if evg_raw:
            lines.append("evergreen: |")
            for ln in (evg_raw.splitlines() or [evg_raw]):
                lines.append(f"  {ln}")
        lines.append("---\n")
        body = "\n".join(lines) + "\n"
        await write_text(path, body)
    except Exception:
        return
