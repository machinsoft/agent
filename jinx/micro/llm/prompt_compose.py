from __future__ import annotations

import platform
import re
import sys
from datetime import datetime
from typing import List
import asyncio

from jinx.micro.conversation.cont import load_last_anchors
from jinx.micro.runtime.exports import collect_export, get_program_export

_VAR_RE = re.compile(r"\{\{var:([a-zA-Z0-9_]+)\}\}")
_ENV_RE = re.compile(r"\{\{env:([A-Z0-9_]+)\}\}")
_ANCHORS_RE = re.compile(r"\{\{anchors:(questions|symbols|paths)(?::(\d+))?\}\}")
_SYS_RE = re.compile(r"\{\{sys:(os|py|cwd)\}\}")
_TIME_RE = re.compile(r"\{\{time:(iso|epoch)\}\}")
_RUN_RE = re.compile(r"\{\{runtime:(programs|programs_count)\}\}")
_EXP_RE = re.compile(r"\{\{export:([a-zA-Z0-9_]+)(?::(\d+))?\}\}")
_PEXP_RE = re.compile(r"\{\{program:([a-zA-Z0-9]+):([a-zA-Z0-9_]+)\}\}")


async def compose_dynamic_prompt(text: str, *, key: str) -> str:
    """Expand lightweight runtime macros inside the prompt text.

    Supported forms:
    - {{var:key}} -> code tag id
    - {{anchors:questions[:N]}} / {{anchors:symbols[:N]}} / {{anchors:paths[:N]}} -> comma-separated
    - {{sys:os|py|cwd}} -> platform system / python version / current working dir
    - {{time:iso|epoch}} -> current time ISO8601 or seconds since epoch
    - {{runtime:programs|programs_count}} -> active program IDs list (comma) or count
    """
    if not text:
        return text

    out = text

    # 1) Vars
    def _var_sub(m: re.Match) -> str:
        name = m.group(1).strip().lower()
        if name == "key":
            return key
        return ""

    out = _VAR_RE.sub(_var_sub, out)

    # 2) Env
    def _env_sub(m: re.Match) -> str:
        return ""

    out = _ENV_RE.sub(_env_sub, out)

    # 3) Anchors
    try:
        anc = await load_last_anchors()
    except Exception:
        anc = {}

    def _anch_sub(m: re.Match) -> str:
        kind = m.group(1)
        n_s = m.group(2)
        try:
            n = int(n_s) if n_s else None
        except Exception:
            n = None
        arr: List[str] = [str(x).strip() for x in (anc.get(kind) or []) if str(x).strip()]
        if n is not None:
            arr = arr[:max(0, n)]
        return ", ".join(arr)

    out = _ANCHORS_RE.sub(_anch_sub, out)

    # 4) System
    def _sys_sub(m: re.Match) -> str:
        what = m.group(1)
        if what == "os":
            return platform.system()
        if what == "py":
            return sys.version.split(" ")[0]
        if what == "cwd":
            try:
                import os
                return os.getcwd()
            except Exception:
                return ""
        return ""

    out = _SYS_RE.sub(_sys_sub, out)

    # 5) Time
    def _time_sub(m: re.Match) -> str:
        kind = m.group(1)
        if kind == "iso":
            try:
                return datetime.now().isoformat(timespec="seconds")
            except Exception:
                return ""
        if kind == "epoch":
            try:
                return str(int(datetime.now().timestamp()))
            except Exception:
                return ""
        return ""

    out = _TIME_RE.sub(_time_sub, out)

    # 6) Runtime
    async def _runtime_expand(s: str) -> str:
        # Expand runtime macros one by one to avoid multiple list calls
        has_prog = bool(_RUN_RE.search(s))
        if not has_prog:
            return s
        try:
            from jinx.micro.runtime.api import list_programs as _list_programs
            pids = await _list_programs()
        except Exception:
            pids = []
        def _run_sub(m: re.Match) -> str:
            tok = m.group(1)
            if tok == "programs_count":
                return str(len(pids))
            if tok == "programs":
                return ", ".join(pids)
            return ""
        return _RUN_RE.sub(_run_sub, s)

    out = await _runtime_expand(out)

    # 7) Program exports (aggregated)
    async def _export_expand(s: str) -> str:
        has_exp = bool(_EXP_RE.search(s)) or bool(_PEXP_RE.search(s))
        if not has_exp:
            return s
        # aggregate exports first — manual streaming replacement (no nested event loop)
        # Replace sequentially
        text = s
        pos = 0
        buf: List[str] = []
        while True:
            m = _EXP_RE.search(text, pos)
            if not m:
                buf.append(text[pos:])
                break
            buf.append(text[pos:m.start()])
            key_name = m.group(1)
            n_s = m.group(2)
            try:
                n = int(n_s) if n_s else None
            except Exception:
                n = None
            try:
                vals = await collect_export(key_name, limit=n)
            except Exception:
                vals = []
            buf.append(", ".join(vals))
            pos = m.end()
        s2 = "".join(buf)

        # program-specific export: {{program:PID:key}}
        pos = 0
        buf = []
        while True:
            m = _PEXP_RE.search(s2, pos)
            if not m:
                buf.append(s2[pos:])
                break
            buf.append(s2[pos:m.start()])
            pid = m.group(1)
            exk = m.group(2)
            try:
                val = await get_program_export(pid, exk)
            except Exception:
                val = ""
            buf.append(val)
            pos = m.end()
        return "".join(buf)

    out = await _export_expand(out)
    return out
