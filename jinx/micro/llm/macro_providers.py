from __future__ import annotations

import os
import re
import asyncio
from typing import List

from jinx.micro.llm.macro_registry import register_macro, MacroContext
from jinx.micro.embeddings.retrieval import retrieve_top_k as _dlg_topk
from jinx.micro.embeddings.project_retrieval import retrieve_project_top_k as _proj_topk
from jinx.micro.memory.storage import read_compact as _read_compact, read_evergreen as _read_evergreen, read_channel as _read_channel, read_topic as _read_topic
from jinx.micro.memory.search import rank_memory as _rank_memory
from jinx.micro.memory.graph import query_graph as _query_graph
from jinx.micro.memory.pin_store import load_pins as _pins_load, save_pins as _pins_save
from jinx.micro.memory.router import assemble_memroute as _memroute
from jinx.micro.memory.unified import assemble_unified_memory_lines as _mem_unified
from jinx.micro.llm.macro_cache import memoized_call
from jinx.micro.memory.turns import parse_active_turns as _parse_turns, get_user_message as _turn_user, get_jinx_reply_to as _turn_jinx
from jinx.micro.embeddings.symbol_index import query_symbol_index as _sym_query

_EMB_TOPK_DEFAULT = 3
_EMB_MS_DEFAULT = 180
_EMB_PREVIEW_CHARS_DEFAULT = 160
_MEM_TOPK_DEFAULT = 8
_MEM_PREVIEW_CHARS_DEFAULT = 160
_MACRO_PROVIDER_TTL_MS_DEFAULT = 1500
_TURNS_PREVIEW_CHARS_DEFAULT = 160
_RUN_EXPORT_TTL_MS_DEFAULT = 120000
_CODE_TOPK_DEFAULT = 8
_CODE_MS_DEFAULT = 280
_CODE_PREVIEW_CHARS_DEFAULT = 160

_registered = False


def _norm_preview(x: str, lim: int) -> str:
    s = " ".join((x or "").split())
    return s[:lim]


async def _emb_handler(args: List[str], ctx: MacroContext) -> str:
    from jinx.micro.logger.debug_logger import debug_log
    await debug_log(f"Called with args={args}", "MACRO:emb")
    
    try:
        scope = (args[0] if args else "dialogue").strip().lower()
    except Exception:
        scope = "dialogue"
    
    await debug_log(f"Scope={scope}", "MACRO:emb")
    
    n = 0
    q = ""
    # parse args like [scope, N, q=...]
    for a in (args[1:] if len(args) > 1 else []):
        aa = a.strip()
        if aa.startswith("q="):
            q = aa[2:]
            continue
        try:
            n = int(aa)
        except Exception:
            pass
    if n <= 0:
        n = _EMB_TOPK_DEFAULT
    if not q:
        q = (ctx.input_text or "").strip()
    if not q:
        # fallback: last question anchor
        try:
            q = (ctx.anchors.get("questions") or [""])[-1].strip()
        except Exception:
            q = ""
    if not q:
        await debug_log("✗ No query text available", "MACRO:emb")
        return ""
    
    await debug_log(f"Query='{q[:50]}...' k={n}", "MACRO:emb")
    ms = _EMB_MS_DEFAULT
    lim = _EMB_PREVIEW_CHARS_DEFAULT

    out: List[str] = []
    if scope in ("dialogue", "dlg"):
        await debug_log("Retrieving from dialogue embeddings...", "MACRO:emb")
        hits = await _dlg_topk(q, k=n, max_time_ms=ms)
        await debug_log(f"Got {len(hits)} hits from dialogue", "MACRO:emb")
        for _score, _src, obj in hits:
            meta = obj.get("meta", {})
            pv = (meta.get("text_preview") or "").strip()
            if not pv:
                continue
            out.append(_norm_preview(pv, lim))
    elif scope in ("project", "proj"):
        await debug_log("Retrieving from project embeddings...", "MACRO:emb")
        hits = await _proj_topk(q, k=n, max_time_ms=ms)
        await debug_log(f"Got {len(hits)} hits from project", "MACRO:emb")
        for _score, file_rel, obj in hits:
            meta = obj.get("meta", {})
            pv = (meta.get("text_preview") or "").strip()
            if pv:
                out.append(_norm_preview(pv, lim))
                continue
            ls = int(meta.get("line_start") or 0)
            le = int(meta.get("line_end") or 0)
            if file_rel:
                if ls or le:
                    out.append(f"[{file_rel}:{ls}-{le}]")
                else:
                    out.append(f"[{file_rel}]")
    else:
        await debug_log(f"✗ Unknown scope: {scope}", "MACRO:emb")
        return ""

    # Compact single-line result for inline prompt usage
    out = [s for s in out if s]
    result = " | ".join(out[:n])
    await debug_log(f"✓ Returning {len(out)} results ({len(result)} chars)", "MACRO:emb")
    return result


async def _memfacts_handler(args: List[str], ctx: MacroContext) -> str:
    """Facts provider: {{m:memfacts:kind[:N]}}

    kind: paths|symbols|prefs|decisions
    N: number of lines (default 8)
    """
    kind = (args[0] if args else "").strip().lower()
    if kind not in ("paths","symbols","prefs","decisions"):
        return ""
    n = 0
    if len(args) > 1:
        try:
            n = int(args[1])
        except Exception:
            n = 0
    if n <= 0:
        n = _MEM_TOPK_DEFAULT
    lim = _MEM_PREVIEW_CHARS_DEFAULT
    try:
        txt = await _read_channel(kind)
    except Exception:
        txt = ""
    if not txt:
        return ""
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    out = [ln[:lim] for ln in lines[:n]]
    return " | ".join(out)


async def _memgraph_handler(args: List[str], ctx: MacroContext) -> str:
    """Knowledge graph neighbors: {{m:memgraph:term[:K]}}

    term: substring to match node keys (e.g., 'symbol: my_func' or 'path: utils.py') or any token
    K: number of neighbors (default 8)
    """
    term = (args[0] if args else "").strip()
    if not term:
        # fallback to input_text
        term = (ctx.input_text or "").strip()
    n = 0
    if len(args) > 1:
        try:
            n = int(args[1])
        except Exception:
            n = 0
    if n <= 0:
        n = _MEM_TOPK_DEFAULT
    if not term:
        return ""
    try:
        items = await _query_graph(term, k=n)
    except Exception:
        items = []
    if not items:
        return ""
    # already formatted as 'key (score)'
    return " | ".join(items[:n])


async def _memtopic_handler(args: List[str], ctx: MacroContext) -> str:
    """Topic memory: {{m:memtopic:name[:N]}}

    Reads from .jinx/memory/topics/<name>.md
    """
    name = (args[0] if args else "").strip().lower()
    if not name:
        return ""
    n = 0
    if len(args) > 1:
        try:
            n = int(args[1])
        except Exception:
            n = 0
    if n <= 0:
        n = _MEM_TOPK_DEFAULT
    lim = _MEM_PREVIEW_CHARS_DEFAULT
    try:
        txt = await _read_topic(name)
    except Exception:
        txt = ""
    if not txt:
        return ""
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return " | ".join([ln[:lim] for ln in lines[:n]])


async def _memroute_handler(args: List[str], ctx: MacroContext) -> str:
    """Assemble routed memory: {{m:memroute[:K]}} using pins+graph+ranker.

    K default 12, preview via JINX_MACRO_MEM_PREVIEW_CHARS.
    """
    n = 0
    if len(args) > 0:
        try:
            n = int(args[0])
        except Exception:
            n = 0
    if n <= 0:
        n = 12
    lim = _MEM_PREVIEW_CHARS_DEFAULT
    q = (ctx.input_text or "").strip()
    if not q:
        try:
            q = (ctx.anchors.get("questions") or [""])[-1].strip()
        except Exception:
            q = ""
    # TTL memoization to avoid recomputation within a short window
    ttl_ms = _MACRO_PROVIDER_TTL_MS_DEFAULT
    key = f"memroute|{n}|{lim}|{q}"
    async def _call() -> str:
        try:
            # Unified path (pins+graph+vector+kb+ranker), deduped & clamped
            lines = await _mem_unified(q, k=n, preview_chars=lim)
        except Exception:
            try:
                # Fallback to legacy memroute
                lines = await _memroute(q, k=n, preview_chars=lim)
            except Exception:
                lines = []
        return " | ".join(lines[:n])
    return await memoized_call(key, ttl_ms, _call)


async def _turns_handler(args: List[str], ctx: MacroContext) -> str:
    """Turns provider: {{m:turns:kind:n[:chars=lim]}}

    kind: user|jinx|pair (default user)
    n: 1-based turn index
    chars: optional clamp (defaults to JINX_MACRO_TURNS_PREVIEW_CHARS or JINX_MACRO_MEM_PREVIEW_CHARS)
    """
    try:
        kind = (args[0] if args else "user").strip().lower()
    except Exception:
        kind = "user"
    n = 0
    clamp = None
    for a in (args[1:] if len(args) > 1 else []):
        aa = (a or "").strip()
        if not aa:
            continue
        if aa.startswith("chars="):
            try:
                clamp = int(aa.split("=",1)[1])
            except Exception:
                pass
            continue
        try:
            n = int(aa)
        except Exception:
            pass
    if n <= 0:
        return ""
    try:
        lim = _TURNS_PREVIEW_CHARS_DEFAULT
    except Exception:
        lim = _MEM_PREVIEW_CHARS_DEFAULT
    if clamp is not None:
        try:
            lim = max(24, int(clamp))
        except Exception:
            pass
    if kind == "user":
        s = await _turn_user(n)
        return (s or "")[:lim]
    if kind == "jinx":
        s = await _turn_jinx(n)
        return (s or "")[:lim]
    if kind == "pair":
        turns = await _parse_turns()
        if n <= 0 or n > len(turns):
            return ""
        t = turns[n-1]
        u = (t.get("user") or "").strip()
        a = (t.get("jinx") or "").strip()
        out = (f"User: {u}\nJinx: {a}").strip()
        return out[:lim]
    return ""


async def _run_handler(args: List[str], ctx: MacroContext) -> str:
    """Last run artifacts: {{m:run:kind[:N][:ttl=ms][:chars=lim]}}

    kind: stdout|stderr|status (default stdout)
    N: number of tail lines for stdout/stderr (default 3)
    ttl: freshness window in ms (default JINX_RUN_EXPORT_TTL_MS or 120000)
    chars: preview clamp (default JINX_MACRO_MEM_PREVIEW_CHARS or 160)
    """
    kind = (args[0] if args else "stdout").strip().lower()
    n = 0
    ttl_ms = None
    lim = None
    # parse optional numeric and kv args
    for a in (args[1:] if len(args) > 1 else []):
        aa = a.strip()
        if not aa:
            continue
        if aa.startswith("ttl="):
            try:
                ttl_ms = int(aa.split("=",1)[1])
            except Exception:
                pass
            continue
        if aa.startswith("chars="):
            try:
                lim = int(aa.split("=",1)[1])
            except Exception:
                pass
            continue
        try:
            n = int(aa)
        except Exception:
            pass
    if n <= 0:
        n = 3 if kind in ("stdout","stderr") else 1
    if ttl_ms is None:
        ttl_ms = _RUN_EXPORT_TTL_MS_DEFAULT
    if lim is None:
        lim = _MEM_PREVIEW_CHARS_DEFAULT
    # TTL memoization across identical macro invocations
    pttl = _MACRO_PROVIDER_TTL_MS_DEFAULT
    key = f"run|{kind}|{n}|{ttl_ms}|{lim}"
    async def _call() -> str:
        # Local import to avoid import-time cycles
        from jinx.micro.exec.run_exports import (
            read_last_stdout as _run_stdout,
            read_last_stderr as _run_stderr,
            read_last_status as _run_status,
        )
        if kind == "stdout":
            return _run_stdout(n, lim, ttl_ms)
        if kind == "stderr":
            return _run_stderr(n, lim, ttl_ms)
        if kind == "status":
            return _run_status(ttl_ms)
        return ""
    return await memoized_call(key, pttl, _call)


async def _codegraph_handler(args: List[str], ctx: MacroContext) -> str:
    """Symbol graph from index: {{m:codegraph:token[:mode][:K]}}

    mode: summary (default) | edges
      - summary: "defs:N calls:M"
      - edges: "def file:line | call file:line | ..." (clamped to K)
    """
    token = (args[0] if args else "").strip()
    mode = (args[1] if len(args) > 1 else "summary").strip().lower()
    K = 0
    if len(args) > 2:
        try:
            K = int(args[2])
        except Exception:
            K = 0
    if not token:
        # try from input_text callable
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", ctx.input_text or "")
        if m:
            token = m.group(1)
    if not token:
        return ""
    try:
        idx = await _sym_query(token)
    except Exception:
        idx = {"defs": [], "calls": []}
    defs = idx.get("defs") or []
    calls = idx.get("calls") or []
    if mode == "edges":
        out: List[str] = []
        for rel, ln in (defs[:K] if K > 0 else defs):
            out.append(f"def {rel}:{ln}")
        for rel, ln in (calls[:K] if K > 0 else calls):
            out.append(f"call {rel}:{ln}")
        return " | ".join(out)
    # summary
    return f"defs:{len(defs)} calls:{len(calls)}"


# ----------------------- Code intelligence providers -----------------------

_EXCLUDE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "node_modules", "emb", "build", "dist"}


def _is_excluded_dir(name: str) -> bool:
    n = name.lower()
    return n in _EXCLUDE_DIRS or n.startswith(".")


def _iter_py_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if not _is_excluded_dir(d)]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            yield os.path.join(dirpath, fn)


async def _code_handler(args: List[str], ctx: MacroContext) -> str:
    """Code provider: {{m:code:usage|def|class|import:token[:K][:ms=budget][:chars=lim]}}

    - usage: find call sites for a symbol (regex: \btoken\s*\()
    - def: find function definitions (regex: ^\s*def\s+token\s*\()
    - class: find class definitions (regex: ^\s*class\s+token\s*[(:])
    - import: find import lines referencing token (import X or from Y import X)
    Returns: "path:line: preview | path:line: preview ..."
    """
    mode = (args[0] if args else "usage").strip().lower()
    token = ""
    n = 0
    budget_ms = None
    lim = None
    # parse rest of args
    for a in (args[1:] if len(args) > 1 else []):
        aa = (a or "").strip()
        if not aa:
            continue
        if aa.startswith("ms="):
            try:
                budget_ms = int(aa.split("=",1)[1])
            except Exception:
                pass
            continue
        if aa.startswith("chars="):
            try:
                lim = int(aa.split("=",1)[1])
            except Exception:
                pass
            continue
        if aa.isdigit():
            try:
                n = int(aa)
            except Exception:
                pass
            continue
        # first non-kv arg after mode is token
        if not token:
            token = aa
    if not token:
        # fallback: extract first callable name from input_text
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", ctx.input_text or "")
        if m:
            token = m.group(1)
    if not token:
        return ""
    if n <= 0:
        n = _CODE_TOPK_DEFAULT
    if budget_ms is None:
        budget_ms = _CODE_MS_DEFAULT
    if lim is None:
        lim = _CODE_PREVIEW_CHARS_DEFAULT

    # Attempt symbol index first (exact token) for defs/calls
    try:
        idx = await _sym_query(token)
    except Exception:
        idx = {"defs": [], "calls": []}
    defs = idx.get("defs") or []
    calls = idx.get("calls") or []
    if mode in ("def", "class") and idx.get("defs"):
        pairs = idx.get("defs")[:n]
        return " | ".join([f"{rel}:{ln}" for rel, ln in pairs])
    if mode == "usage" and idx.get("calls"):
        pairs = idx.get("calls")[:n]
        return " | ".join([f"{rel}:{ln}" for rel, ln in pairs])

    # Compile regex once per mode (fallback scan)
    if mode == "def":
        pat = re.compile(rf"(?mi)^\s*def\s+{re.escape(token)}\s*\(")
    elif mode == "class":
        pat = re.compile(rf"(?mi)^\s*class\s+{re.escape(token)}\s*[(:]")
    elif mode == "import":
        pat = re.compile(rf"(?mi)^(?:\s*from\s+[^\n]+\s+import\s+[^\n]*\b{re.escape(token)}\b|\s*import\s+[^\n]*\b{re.escape(token)}\b)")
    else:
        pat = re.compile(rf"(?m)\b{re.escape(token)}\s*\(")

    root = ctx.cwd or os.getcwd()

    async def _scan() -> str:
        def _work() -> str:
            out: List[str] = []
            count = 0
            import time as _t
            t0 = _t.perf_counter()
            for fp in _iter_py_files(root):
                if budget_ms and (_t.perf_counter() - t0) * 1000.0 > budget_ms:
                    break
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, start=1):
                            if pat.search(line):
                                rel = os.path.relpath(fp, root)
                                prev = " ".join(line.strip().split())[:lim]
                                out.append(f"{rel}:{i}: {prev}")
                                count += 1
                                if count >= n:
                                    return " | ".join(out[:n])
                except Exception:
                    continue
            return " | ".join(out[:n])
        res = await asyncio.to_thread(_work)
        if res:
            return res
        # Fallback: embeddings-based project search by token if regex scan found nothing
        try:
            hits = await _proj_topk(token, k=n, max_time_ms=min(400, max(120, budget_ms or 200)))
        except Exception:
            hits = []
        out: List[str] = []
        for _score, file_rel, obj in hits:
            meta = obj.get("meta", {})
            pv = (meta.get("text_preview") or "").strip()
            if pv:
                out.append(_norm_preview(pv, lim))
                continue
            ls = int(meta.get("line_start") or 0)
            le = int(meta.get("line_end") or 0)
            tag = f"[{file_rel}:{ls}-{le}]" if (file_rel and (ls or le)) else (f"[{file_rel}]" if file_rel else "")
            if tag:
                out.append(tag)
        return " | ".join(out[:n])

    # TTL memoization to avoid repeated scans on identical queries
    pttl = _MACRO_PROVIDER_TTL_MS_DEFAULT
    key = f"code|{mode}|{token}|{n}|{budget_ms}|{lim}"
    return await memoized_call(key, pttl, _scan)


# ---------------------------- Policy rails macro ----------------------------

async def _policy_handler(args: List[str], ctx: MacroContext) -> str:
    """Policy provider: {{m:policy:jinx_rails}}

    Returns a small ASCII-only instruction tail to reinforce Jinx rails uniformly.
    """
    variant = (args[0] if args else "").strip().lower()
    if variant != "jinx_rails":
        return ""
    lines = [
        "- ASCII-only; no code fences; avoid angle brackets in values.",
        "- Respect Risk Policies; avoid denied paths and globs.",
        "- Atomic diffs; tiny patches; preserve existing style and naming.",
        "- Async-first; never block the event loop; offload CPU via asyncio.to_thread.",
        "- Avoid non-stdlib deps unless already present; prefer internal runtime APIs.",
        "- No triple quotes; do NOT use try/except.",
        "- Deterministic outputs; JSON-only where required; keep outputs concise.",
    ]
    return "\n" + "\n".join(lines) + "\n"


def _pins_enabled() -> bool:
    return True


async def _pins_handler(args: List[str], ctx: MacroContext) -> str:
    """List pinned lines: {{m:pins[:N]}}"""
    n = 0
    if len(args) > 0:
        try:
            n = int(args[0])
        except Exception:
            n = 0
    if n <= 0:
        n = _MEM_TOPK_DEFAULT
    try:
        pins = _pins_load()
    except Exception:
        pins = []
    out = [p for p in pins[:n] if p]
    return " | ".join(out)


async def _pinadd_handler(args: List[str], ctx: MacroContext) -> str:
    """Add a pinned line: {{m:pinadd:line...}} (uses input_text if empty)."""
    line = " ".join(args).strip()
    if not line:
        line = (ctx.input_text or "").strip()
    if not line:
        return ""
    try:
        pins = _pins_load()
    except Exception:
        pins = []
    if line not in pins:
        pins.insert(0, line)
        try:
            _pins_save(pins)
        except Exception:
            pass
    return line


async def _pindel_handler(args: List[str], ctx: MacroContext) -> str:
    """Delete a pinned line by exact match: {{m:pindel:line...}}"""
    line = " ".join(args).strip()
    if not line:
        return ""
    try:
        pins = _pins_load()
    except Exception:
        pins = []
    pins = [p for p in pins if p != line]
    try:
        _pins_save(pins)
    except Exception:
        pass
    return ""


def _tokens(s: str) -> List[str]:
    out: List[str] = []
    for m in re.finditer(r"(?u)[\w\.]+", s or ""):
        t = (m.group(0) or "").strip().lower()
        if t and len(t) >= 3:
            out.append(t)
    return out


async def _mem_handler(args: List[str], ctx: MacroContext) -> str:
    """Memory provider: {{m:mem:scope[:N][:q=...]}}

    scope: compact|evergreen|any (default: compact)
    N: number of snippets (default: JINX_MACRO_MEM_TOPK or 6)
    q=: optional query to filter/select relevant lines
    """
    scope = (args[0] if args else "compact").strip().lower()
    n = 0
    q = ""
    for a in (args[1:] if len(args) > 1 else []):
        aa = a.strip()
        if aa.startswith("q="):
            q = aa[2:]
            continue
        try:
            n = int(aa)
        except Exception:
            pass
    if n <= 0:
        n = 6
    lim = _MEM_PREVIEW_CHARS_DEFAULT

    # Load memory texts
    try:
        comp = await _read_compact()
    except Exception:
        comp = ""
    try:
        ever = await _read_evergreen()
    except Exception:
        ever = ""

    def _lines_of(txt: str) -> List[str]:
        return [ln.strip() for ln in (txt or "").splitlines() if ln.strip()]

    c_lines = _lines_of(comp)
    e_lines = _lines_of(ever)
    # Build candidate pool
    if scope == "evergreen":
        pool = e_lines
    elif scope == "any":
        pool = e_lines + c_lines
    else:
        pool = c_lines

    if not pool:
        return ""

    # Selection
    out: List[str] = []
    if q:
        # Use async ranker for better relevance
        ranked = await _rank_memory(q, scope=scope if scope in ("compact","evergreen","any") else "compact", k=n, preview_chars=lim)
        out.extend(ranked)
    else:
        # No query: take most recent for compact, otherwise head for evergreen
        if scope == "evergreen":
            for ln in e_lines[-n:]:
                out.append(ln[:lim])
        elif scope == "any":
            # interleave last few from both (favor compact recency)
            tail_c = c_lines[-(n*2):]
            tail_e = e_lines[-n:]
            merged = (tail_c + tail_e)[-n:]
            out.extend([ln[:lim] for ln in merged])
        else:
            for ln in c_lines[-n:]:
                out.append(ln[:lim])

    out = [s for s in out if s]
    return " | ".join(out[:n])


async def register_builtin_macros() -> None:
    global _registered
    if _registered:
        return
    await register_macro("emb", _emb_handler)
    await register_macro("mem", _mem_handler)
    await register_macro("memfacts", _memfacts_handler)
    await register_macro("memgraph", _memgraph_handler)
    await register_macro("memtopic", _memtopic_handler)
    await register_macro("memroute", _memroute_handler)
    await register_macro("turns", _turns_handler)
    await register_macro("run", _run_handler)
    await register_macro("pins", _pins_handler)
    await register_macro("pinadd", _pinadd_handler)
    await register_macro("pindel", _pindel_handler)
    await register_macro("code", _code_handler)
    await register_macro("codegraph", _codegraph_handler)
    await register_macro("policy", _policy_handler)
    _registered = True
