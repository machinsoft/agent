from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Awaitable, Dict
import time

from jinx.micro.embeddings.search_cache import search_project_cached
from .write_patch import patch_write
from .line_patch import patch_line_range
from .anchor_patch import patch_anchor_insert_after
from .symbol_patch import patch_symbol_python
from .symbol_body_patch import patch_symbol_body_python
from .context_patch import patch_context_replace
from .semantic_patch import patch_semantic_in_file
from .utils import diff_stats as _diff_stats, should_autocommit as _should_autocommit
from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root
from jinx.micro.common.internal_paths import is_restricted_path
from jinx.micro.brain.concepts import activate_concepts as _brain_activate
from jinx.micro.brain.attention import record_attention as _att_rec
import re as _re
from jinx.micro.memory.graph import apply_feedback as _kg_feedback
from .strategy_bandit import bandit_order_for_context as _bandit_order, bandit_update as _bandit_update
from .symbol_patch_generic import patch_symbol_generic as _patch_symbol_generic
from jinx.micro.brain.concepts import record_reinforcement as _brain_reinf
from jinx.micro.embeddings.symbol_index import query_symbol_index as _sym_query
from jinx.micro.common.env import truthy as _truthy
from jinx.micro.runtime.autobrain_config import (
    get_int as _ab_int,
    get as _ab_get,
    record_outcome as _ab_record,
    TaskContext as _AbTaskCtx,
)
try:
    from jinx.observability.metrics import record_patch_event as _rec_patch  # type: ignore
except Exception:
    _rec_patch = None  # type: ignore
try:
    from jinx.observability.otel import span as _span  # type: ignore
except Exception:
    _span = None  # type: ignore
try:
    from jinx.micro.embeddings.project_callgraph import windows_for_symbol as _cg_windows  # type: ignore
except Exception:
    _cg_windows = None  # type: ignore
try:
    # Optional CodeGraph tree-sitter backend
    from jinx.codegraph.ts_backend import defs_for_token as _ts_defs  # type: ignore
except Exception:
    _ts_defs = None  # type: ignore
try:
    from jinx.rt.threadpool import run_cpu as _run_cpu  # type: ignore
except Exception:
    _run_cpu = None  # type: ignore
try:
    from jinx.rt.admission import guard as _guard  # type: ignore
except Exception:
    _guard = None  # type: ignore
try:
    from jinx.micro.embeddings.project_pipeline import embed_file as _embed_file  # type: ignore
    from jinx.micro.embeddings.project_util import sha256_path as _sha_path  # type: ignore
except Exception:
    _embed_file = None  # type: ignore
    _sha_path = None  # type: ignore
try:
    from jinx.micro.embeddings.refresh_queue import enqueue_refresh as _enqueue_embed  # type: ignore
except Exception:
    _enqueue_embed = None  # type: ignore
try:
    from jinx.micro.runtime.risk_policies import is_allowed_path as _risk_allow  # type: ignore
except Exception:
    def _risk_allow(_p: str) -> bool:  # type: ignore
        return True


@dataclass
class AutoPatchArgs:
    path: Optional[str] = None
    code: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    symbol: Optional[str] = None
    anchor: Optional[str] = None
    query: Optional[str] = None
    preview: bool = False
    max_span: Optional[int] = None
    force: bool = False
    context_before: Optional[str] = None
    context_tolerance: Optional[float] = None


async def _evaluate_candidate(name: str, coro) -> Tuple[str, bool, str, int]:
    """Run candidate in preview mode, return (name, ok, diff_or_detail, total_changes)."""
    ok, detail = await coro
    # detail is usually a diff for preview=True; compute size as risk proxy
    add, rem = _diff_stats(detail or "")
    total = add + rem
    return name, ok, detail, total


async def autopatch(args: AutoPatchArgs) -> Tuple[bool, str, str]:
    """Choose best patch strategy using a smart, candidate-based selector.

    - Builds an ordered list of viable strategies from the provided args.
    - Evaluates candidates in preview mode with timeboxing; scores by autocommit suitability and diff size.
    - Commits the best candidate (or returns its preview when args.preview is True).

    Returns (ok, strategy, detail_or_diff).
    """
    path = args.path or ""
    code = args.code or ""
    start_ts = time.monotonic()
    # Adaptive configuration from AutoBrain
    max_ms = _ab_int("autopatch_max_ms")
    # Exhaustive mode: disable timeboxing and search caps if enabled
    no_budgets = _truthy("JINX_AUTOPATCH_NO_BUDGETS", "1")
    max_ms_local = None if no_budgets else max_ms
    search_k = _ab_int("autopatch_search_topk")

    # Resolve project root once
    ROOT = _resolve_root()

    # Brain-driven query expansion (future-grade retrieval): expand terms/symbols
    exp_query = (args.query or "").strip()
    brain_pairs: List[Tuple[str, float]] = []
    if exp_query:
        try:
            if _truthy("EMBED_BRAIN_ENABLE", "1"):
                btop = 10
                bmax = 6
                brain_pairs = await _brain_activate(exp_query, top_k=btop)
                seen_bt: set[str] = set()
                btoks: List[str] = []
                for key, _sc in brain_pairs:
                    low = (key or "").lower()
                    tok = ""
                    if low.startswith("term: "):
                        tok = low.split(": ", 1)[1]
                    elif low.startswith("symbol: "):
                        tok = low.split(": ", 1)[1]
                    if tok and tok not in seen_bt:
                        btoks.append(tok)
                        seen_bt.add(tok)
                    if len(btoks) >= bmax:
                        break
                if btoks:
                    exp_query = (exp_query + " " + " ".join(btoks)).strip()
        except Exception:
            brain_pairs = []

    # Gather candidates (as tuples of (name, preview_coro_factory, commit_coro_factory))
    candidates: List[Tuple[str, Callable[[], Awaitable[Tuple[bool, str]]], Callable[[], Awaitable[Tuple[bool, str]]]]] = []
    # Metadata per candidate name (embedding score, cg proximity)
    emb_meta: Dict[str, float] = {}
    cg_meta: Dict[str, float] = {}

    # Guard: skip restricted paths
    if path and is_restricted_path(path):
        return False, "restricted", f"path is restricted: {path}"
    # Guard: risk policy deny
    if path:
        try:
            rel_for_risk = os.path.relpath(path, start=_resolve_root())
        except Exception:
            rel_for_risk = path
        if not _risk_allow(rel_for_risk):
            return False, "risk_deny", f"path denied by risk policy: {rel_for_risk}"

    # 1) explicit line range
    if (args.line_start or 0) > 0 and (args.line_end or 0) > 0 and path:
        ls = int(args.line_start)
        le = int(args.line_end)
        candidates.append((
            "line",
            lambda: patch_line_range(path, ls, le, code, preview=True, max_span=args.max_span),
            lambda: patch_line_range(path, ls, le, code, preview=False, max_span=args.max_span),
        ))

    # 3c) tree-sitter assisted replacement for non-Python files when symbol+code provided
    if path and (args.symbol or "") and (args.code or ""):
        try:
            ext = os.path.splitext(path)[1].lower()
        except Exception:
            ext = ""
        if ext and ext not in (".py",):
            if _ts_defs is not None:
                # Build a candidate that replaces the definition span
                async def _ts_preview_commit(preview: bool):
                    try:
                        # Admission guard for heavy TS
                        if _guard is not None:
                            async with _guard("graph", timeout_ms=150) as admitted:
                                if not admitted:
                                    return False, "no ts token"
                                spans = await asyncio.to_thread(_ts_defs, path, args.symbol or "", max_items=1)
                        else:
                            spans = await asyncio.to_thread(_ts_defs, path, args.symbol or "", max_items=1)
                    except Exception:
                        spans = []
                    if not spans:
                        return False, "no ts span"
                    ls, le = spans[0]
                    return await patch_line_range(path, int(ls), int(le), args.code or "", preview=preview, max_span=args.max_span)
                candidates.append((
                    "ts_line",
                    lambda: _ts_preview_commit(True),
                    lambda: _ts_preview_commit(False),
                ))

    # 6b) tree-sitter based locator when path unknown and symbol provided
    if not path and (args.symbol or "") and _ts_defs is not None:
        try:
            from jinx.micro.embeddings.project_config import ROOT, INCLUDE_EXTS, EXCLUDE_DIRS, MAX_FILE_BYTES  # type: ignore
            from jinx.micro.embeddings.project_iter import iter_candidate_files  # type: ignore
        except Exception:
            ROOT = os.getcwd()
            iter_candidate_files = None  # type: ignore
        async def _locate_via_ts(max_scan: int = 200) -> list[str]:
            outs: list[str] = []
            if iter_candidate_files is None:
                return outs
            cnt = 0
            for abs_p, rel_p in iter_candidate_files(ROOT, include_exts=INCLUDE_EXTS, exclude_dirs=EXCLUDE_DIRS, max_file_bytes=MAX_FILE_BYTES):
                if rel_p.endswith('.py'):
                    continue
                cnt += 1
                if cnt > max_scan:
                    break
                try:
                    spans = await asyncio.to_thread(_ts_defs, abs_p, args.symbol or "", max_items=1)
                except Exception:
                    spans = []
                if spans:
                    outs.append(abs_p)
                    break
            return outs
        paths = await _locate_via_ts()
        for fpath in paths[:1]:
            try:
                rel_for_risk = os.path.relpath(fpath, start=ROOT)
            except Exception:
                rel_for_risk = fpath
            if not _risk_allow(rel_for_risk):
                continue
            async def _ts_preview_commit2(preview: bool, fpath=fpath):
                try:
                    if _guard is not None:
                        async with _guard("graph", timeout_ms=150) as admitted:
                            if not admitted:
                                return False, "no ts token"
                            if _run_cpu is not None:
                                spans = await _run_cpu(_ts_defs, fpath, args.symbol or "", max_items=1)
                            else:
                                spans = await asyncio.to_thread(_ts_defs, fpath, args.symbol or "", max_items=1)
                    else:
                        spans = await asyncio.to_thread(_ts_defs, fpath, args.symbol or "", max_items=1)
                except Exception:
                    spans = []
                if not spans:
                    return False, "no ts span"
                ls, le = spans[0]
                return await patch_line_range(fpath, int(ls), int(le), args.code or "", preview=preview, max_span=args.max_span)
            candidates.append((
                "ts_line_search",
                lambda fpath=fpath: _ts_preview_commit2(True, fpath=fpath),
                lambda fpath=fpath: _ts_preview_commit2(False, fpath=fpath),
            ))
    # 3a) symbol index suggestions when symbol provided but path unknown
    if (args.symbol or "") and not path:
        try:
            idx = await _sym_query(args.symbol or "")
        except Exception:
            idx = {"defs": [], "calls": []}
        defs = idx.get("defs") or []
        # Optionally include callgraph windows rooted at preferred def
        if (args.code or "") and _cg_windows:
            try:
                ROOT = _resolve_root()
                pref_rel = defs[0][0] if defs else None
                if _guard is not None:
                    async with _guard("graph", timeout_ms=150) as admitted:
                        if not admitted:
                            wins = []
                        elif _run_cpu is not None:
                            wins = await _run_cpu(
                                _cg_windows,
                                args.symbol or "",
                                prefer_rel=pref_rel,
                                callers_limit=2,
                                callees_limit=2,
                                around=10,
                                scan_cap_files=160,
                                time_budget_ms=450,
                            )
                        else:
                            wins = await asyncio.to_thread(
                                _cg_windows,
                                args.symbol or "",
                                prefer_rel=pref_rel,
                                callers_limit=2,
                                callees_limit=2,
                                around=10,
                                scan_cap_files=160,
                                time_budget_ms=450,
                            )
                else:
                    wins = await asyncio.to_thread(
                        _cg_windows,
                        args.symbol or "",
                        prefer_rel=pref_rel,
                        callers_limit=2,
                        callees_limit=2,
                        around=10,
                        scan_cap_files=160,
                        time_budget_ms=450,
                    )
            except Exception:
                wins = []
            cap = 3
            for rel, a, b, kind in wins[:cap]:
                fpath = os.path.join(ROOT, rel)
                if is_restricted_path(fpath):
                    continue
                if not _risk_allow(rel):
                    continue
                nm = f"cg_window_search_{kind.lower()}@{rel}"
                candidates.append((
                    nm,
                    (lambda fpath=fpath, a=a, b=b: patch_line_range(fpath, int(a), int(b), code, preview=True, max_span=args.max_span)),
                    (lambda fpath=fpath, a=a, b=b: patch_line_range(fpath, int(a), int(b), code, preview=False, max_span=args.max_span)),
                ))
                cg_meta[nm] = 1.0
        # Prefer definitions; use body vs header depending on code
        def _looks_like_header(snippet: str) -> bool:
            sn = (snippet or "").lstrip()
            if sn.startswith(("def ", "class ", "async def ")):
                return True
            try:
                import ast as _ast
                m = _ast.parse(snippet or "")
                for n in getattr(m, "body", []) or []:
                    if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                        return True
            except Exception:
                pass
            return False
        for rel, _ln in defs[:6]:  # cap to keep RT budgets
            fpath = os.path.join(ROOT, rel)
            if is_restricted_path(fpath):
                continue
            if _looks_like_header(code or ""):
                candidates.append((
                    "symbol",
                    (lambda fpath=fpath: patch_symbol_python(fpath, args.symbol or "", code, preview=True)),
                    (lambda fpath=fpath: patch_symbol_python(fpath, args.symbol or "", code, preview=False)),
                ))
            else:
                candidates.append((
                    "symbol_body",
                    (lambda fpath=fpath: patch_symbol_body_python(fpath, args.symbol or "", code, preview=True)),
                    (lambda fpath=fpath: patch_symbol_body_python(fpath, args.symbol or "", code, preview=False)),
                ))

    # 2) python symbol (requires path) with smarter header detection
    if (args.symbol or "") and (str(path).endswith(".py") if path else False):
        # CodeGraph-based scope patch: replace exact def/class scope for symbol if resolvable
        try:
            from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root
            from jinx.micro.embeddings.project_py_scope import find_python_scope as _py_scope
        except Exception:
            _resolve_root = None  # type: ignore
            _py_scope = None  # type: ignore
        if _resolve_root and _py_scope:
            try:
                ROOT = _resolve_root()
                rel_here = os.path.relpath(path, ROOT)
            except Exception:
                rel_here = None
            async def _cg_scope_preview_commit(preview: bool):
                try:
                    # Use symbol index to find definition line in this file
                    idx = await _sym_query(args.symbol or "")
                except Exception:
                    idx = {"defs": [], "calls": []}
                defs = idx.get("defs") or []
                line_for_this: int | None = None
                for rel, ln in defs:
                    if rel_here and rel == rel_here:
                        line_for_this = int(ln)
                        break
                if line_for_this is None:
                    return False, "no_def_in_file"
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        src = f.read()
                except Exception:
                    return False, "read_fail"
                if _run_cpu is not None:
                    a, b = await _run_cpu(_py_scope, src, line_for_this)
                else:
                    a, b = _py_scope(src, line_for_this)
                if a <= 0 or b <= 0 or b < a:
                    return False, "scope_not_found"
                return await patch_line_range(path, int(a), int(b), args.code or "", preview=preview, max_span=args.max_span)
            candidates.append((
                "cg_scope",
                lambda: _cg_scope_preview_commit(True),
                lambda: _cg_scope_preview_commit(False),
            ))
        # Callgraph windows (DEF/CALLER/CALLEE_DEF) for current symbol
        if _cg_windows and (args.code or ""):
            try:
                ROOT = _resolve_root()
                rel_here = os.path.relpath(path, ROOT)
                if _guard is not None:
                    async with _guard("graph", timeout_ms=150) as admitted:
                        if not admitted:
                            wins = []
                        elif _run_cpu is not None:
                            wins = await _run_cpu(
                                _cg_windows,
                                args.symbol or "",
                                prefer_rel=rel_here,
                                callers_limit=2,
                                callees_limit=2,
                                around=10,
                                scan_cap_files=120,
                                time_budget_ms=400,
                            )
                        else:
                            wins = await asyncio.to_thread(
                                _cg_windows,
                                args.symbol or "",
                                prefer_rel=rel_here,
                                callers_limit=2,
                                callees_limit=2,
                                around=10,
                                scan_cap_files=120,
                                time_budget_ms=400,
                            )
                else:
                    wins = await asyncio.to_thread(
                        _cg_windows,
                        args.symbol or "",
                        prefer_rel=rel_here,
                        callers_limit=2,
                        callees_limit=2,
                        around=10,
                        scan_cap_files=120,
                        time_budget_ms=400,
                    )
            except Exception:
                wins = []
            cap = 3
            for rel, a, b, kind in wins[:cap]:
                fpath = os.path.join(ROOT, rel)
                if is_restricted_path(fpath):
                    continue
                if not _risk_allow(rel):
                    continue
                nm = f"cg_window_{kind.lower()}@{rel}"
                candidates.append((
                    nm,
                    (lambda fpath=fpath, a=a, b=b: patch_line_range(fpath, int(a), int(b), code, preview=True, max_span=args.max_span)),
                    (lambda fpath=fpath, a=a, b=b: patch_line_range(fpath, int(a), int(b), code, preview=False, max_span=args.max_span)),
                ))
                cg_meta[nm] = 1.0
        # Offer a libcst codemod rename candidate when code=new_name and symbol=old_name
        if (args.code or ""):
            async def _rename_preview_commit(preview: bool):
                try:
                    from jinx.codemods.rename_symbol import preview_rename_text as _prev, rename_symbol_file as _apply
                except Exception:
                    return False, "codemod unavailable"
                try:
                    # preview: compute diff by applying in-memory vs file content
                    if preview:
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                src0 = f.read()
                        except Exception:
                            src0 = ""
                        if _run_cpu is not None:
                            dst0 = await _run_cpu(_prev, src0, args.symbol or "", args.code or "")
                        else:
                            dst0 = await asyncio.to_thread(_prev, src0, args.symbol or "", args.code or "")
                        if dst0 == src0:
                            return False, "no changes"
                        # Render a lightweight unified-like summary
                        return True, f"codemod_rename {args.symbol}->{args.code}"
                    else:
                        ok = await _apply(path, old_name=args.symbol or "", new_name=args.code or "")
                        return (ok, f"codemod_rename {args.symbol}->{args.code}")
                except Exception as e:
                    return False, f"codemod error: {e}"
            candidates.append((
                "codemod_rename_py",
                lambda: _rename_preview_commit(True),
                lambda: _rename_preview_commit(False),
            ))
        # Optional: project-wide rename via Rope (env-gated)
        if (args.code or "") and _truthy("JINX_CODEMOD_ROPE", "0"):
            async def _rope_rename(preview: bool):
                try:
                    from jinx.codemods.rope_rename import project_rename_symbol as _proj_rename
                    from jinx.micro.embeddings.project_config import resolve_project_root as _root
                except Exception:
                    return False, "rope unavailable"
                if preview:
                    # Non-destructive preview summary
                    return True, f"rope_rename {args.symbol}->{args.code}"
                # Compute module rel path
                try:
                    ROOT = _root()
                    rel = os.path.relpath(path, ROOT)
                except Exception:
                    rel = path or ""
                ok = await _proj_rename(ROOT, rel, old_name=args.symbol or "", new_name=args.code or "")
                return ok, f"rope_rename {args.symbol}->{args.code}"
            candidates.append((
                "rope_rename_project",
                lambda: _rope_rename(True),
                lambda: _rope_rename(False),
            ))
        def _looks_like_header(snippet: str) -> bool:
            sn = (snippet or "").lstrip()
            if sn.startswith(("def ", "class ", "async def ")):
                return True
            # Try AST parse to see if snippet declares a def/class at top-level
            try:
                import ast as _ast
                m = _ast.parse(snippet or "")
                for n in getattr(m, "body", []) or []:
                    if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                        return True
            except Exception:
                pass
            return False

        if _looks_like_header(code):
            candidates.append((
                "symbol",
                lambda: patch_symbol_python(path, args.symbol or "", code, preview=True),
                lambda: patch_symbol_python(path, args.symbol or "", code, preview=False),
            ))
        else:
            candidates.append((
                "symbol_body",
                lambda: patch_symbol_body_python(path, args.symbol or "", code, preview=True),
                lambda: patch_symbol_body_python(path, args.symbol or "", code, preview=False),
            ))

    # 3) anchor insert
    if (args.anchor or "") and path:
        candidates.append((
            "anchor",
            lambda: patch_anchor_insert_after(path, args.anchor or "", code, preview=True),
            lambda: patch_anchor_insert_after(path, args.anchor or "", code, preview=False),
        ))

    # 3b) context replace (multi-variant tolerances)
    if (args.context_before or "") and path:
        tol = float(args.context_tolerance) if (args.context_tolerance is not None) else 0.72
        candidates.append((
            "context",
            lambda: patch_context_replace(path, args.context_before or "", code, preview=True, tolerance=tol),
            lambda: patch_context_replace(path, args.context_before or "", code, preview=False, tolerance=tol),
        ))
        # add laxer variants to increase hit probability
        for tol_v in (0.64, 0.55):
            candidates.append((
                f"context_{tol_v}",
                (lambda tol_v=tol_v: patch_context_replace(path, args.context_before or "", code, preview=True, tolerance=tol_v)),
                (lambda tol_v=tol_v: patch_context_replace(path, args.context_before or "", code, preview=False, tolerance=tol_v)),
            ))

    # 4) semantic in-file when we know the path and have a query (add wide variant)
    if path and (args.query or ""):
        candidates.append((
            "semantic",
            lambda: patch_semantic_in_file(path, args.query or "", code, preview=True),
            lambda: patch_semantic_in_file(path, args.query or "", code, preview=False),
        ))
        # wider window variant
        candidates.append((
            "semantic_wide",
            lambda: patch_semantic_in_file(path, args.query or "", code, preview=True, topk=8, margin=10, tol=0.5),
            lambda: patch_semantic_in_file(path, args.query or "", code, preview=False, topk=8, margin=10, tol=0.5),
        ))
    # 4b) Node (TS/JS) direct symbol patch if we know the path and symbol
    if path and (args.symbol or ""):
        try:
            ext = os.path.splitext(path)[1].lower()
        except Exception:
            ext = ""
        if ext in (".js", ".jsx"):
            candidates.append((
                "node_symbol",
                lambda: _patch_symbol_generic(path, "js", args.symbol or "", code, preview=True),
                lambda: _patch_symbol_generic(path, "js", args.symbol or "", code, preview=False),
            ))

    # 5) write new file or overwrite
    if path:
        candidates.append((
            "write",
            lambda: patch_write(path, code, preview=True),
            lambda: patch_write(path, code, preview=False),
        ))

    # 6) search-based if query provided (multi-hit across expanded query; add wide semantic variants and Node symbol patchers)
    if not path and (args.query or ""):
        try:
            limit_ms = None if no_budgets else min(max_ms, 600)
            hits_base = await search_project_cached(args.query or "", k=max(1, search_k), max_time_ms=limit_ms)
        except Exception:
            hits_base = []
        hits_exp: List[Dict] = []
        if exp_query and exp_query != (args.query or ""):
            try:
                limit_ms2 = None if no_budgets else min(max_ms, 600)
                hits_exp = await search_project_cached(exp_query, k=max(1, search_k), max_time_ms=limit_ms2)
            except Exception:
                hits_exp = []
        # Merge by file and range
        seen_keys: set[Tuple[str, int, int]] = set()
        merged: List[Dict] = []
        for lst in (hits_base or []), (hits_exp or []):
            for h in (lst or []):
                f = str(h.get("file") or "").strip()
                if not f:
                    continue
                ls_h = int(h.get("line_start") or 1)
                le_h = int(h.get("line_end") or 1)
                kx = (f, ls_h, le_h)
                if kx in seen_keys:
                    continue
                seen_keys.add(kx)
                merged.append(h)
        # Prefer higher embedding score first (stable within same strategy name further on)
        try:
            merged.sort(key=lambda hh: float(hh.get("score", 0.0) or 0.0), reverse=True)
        except Exception:
            pass
        for h in merged:
            f = str(h.get("file") or "").strip()
            if not f:
                continue
            fpath = os.path.join(ROOT, f)
            if is_restricted_path(fpath):
                continue
            if not _risk_allow(f):
                continue
            ls_h = int(h.get("line_start") or 1)
            le_h = int(h.get("line_end") or 1)
            # If symbol provided and Node file, add generic symbol patch candidates
            if (args.symbol or ""):
                try:
                    ext = os.path.splitext(fpath)[1].lower()
                except Exception:
                    ext = ""
                if ext in (".js", ".jsx"):
                    candidates.append((
                        "node_symbol",
                        (lambda fpath=fpath: _patch_symbol_generic(fpath, "js", args.symbol or "", code, preview=True)),
                        (lambda fpath=fpath: _patch_symbol_generic(fpath, "js", args.symbol or "", code, preview=False)),
                    ))
            # Prefer semantic first for each hit, then fallback to line
            nm1 = f"search_semantic@{f}"
            candidates.append((
                nm1,
                (lambda fpath=fpath, q=exp_query or (args.query or ""): patch_semantic_in_file(fpath, q, code, preview=True)),
                (lambda fpath=fpath, q=exp_query or (args.query or ""): patch_semantic_in_file(fpath, q, code, preview=False)),
            ))
            emb_meta[nm1] = float(h.get("score", 0.0) or 0.0)
            nm2 = f"search_semantic_wide@{f}"
            candidates.append((
                nm2,
                (lambda fpath=fpath, q=exp_query or (args.query or ""): patch_semantic_in_file(fpath, q, code, preview=True, topk=8, margin=10, tol=0.5)),
                (lambda fpath=fpath, q=exp_query or (args.query or ""): patch_semantic_in_file(fpath, q, code, preview=False, topk=8, margin=10, tol=0.5)),
            ))
            emb_meta[nm2] = float(h.get("score", 0.0) or 0.0)
            nm3 = f"search_line@{f}"
            candidates.append((
                nm3,
                (lambda fpath=fpath, ls_h=ls_h, le_h=le_h: patch_line_range(fpath, ls_h, le_h, code, preview=True, max_span=args.max_span)),
                (lambda fpath=fpath, ls_h=ls_h, le_h=le_h: patch_line_range(fpath, ls_h, le_h, code, preview=False, max_span=args.max_span)),
            ))
            emb_meta[nm3] = float(h.get("score", 0.0) or 0.0)

    # Reorder candidates by bandit per context (language, symbol/anchor/query flags + extras)
    try:
        ext = os.path.splitext(path)[1].lower() if path else ""
    except Exception:
        ext = ""
    # Extra features: ts, non-py, filesize bucket
    feats: list[str] = []
    try:
        if _ts_defs is not None:
            feats.append("ts")
    except Exception:
        pass
    if ext and ext != ".py":
        feats.append("np")
    try:
        if path and os.path.isfile(path):
            sz = os.path.getsize(path)
            if sz >= 512*1024:
                feats.append("fs_big")
            elif sz >= 128*1024:
                feats.append("fs_med")
            else:
                feats.append("fs_small")
    except Exception:
        pass
    ctx = "|".join([
        (ext or ""),
        ("sym" if (args.symbol or "") else ""),
        ("anc" if (args.anchor or "") else ""),
        ("qry" if (args.query or "") else ""),
        *feats,
    ])
    cand_names = [name for name, _p, _c in candidates]
    try:
        order = _bandit_order(ctx, cand_names)
    except Exception:
        order = cand_names
    rank: Dict[str, int] = {nm: i for i, nm in enumerate(order)}
    candidates.sort(key=lambda t: rank.get(t[0], 1_000_000))

    # Timeboxed concurrent evaluation and selection (batched)
    best: Dict[str, object] | None = None
    PREV_CONC = 4
    # Helper scorer
    def _score(name: str, diff: str, total: int) -> Tuple[int, int, int, int, int]:
        base_for_ok = name.split("@", 1)[0]
        okc, _reason = _should_autocommit(base_for_ok.replace("search_", "").replace("symbol_body", "symbol"), diff)
        pref = {
            "symbol": 9, "symbol_body": 8, "cg_scope": 10,
            "cg_window_def": 8, "cg_window_caller": 7, "cg_window_callee_def": 7,
            "cg_window_search_def": 7, "cg_window_search_caller": 6, "cg_window_search_callee_def": 6,
            "semantic": 7, "search_semantic": 6,
            "context": 5, "anchor": 4, "ts_line": 3, "ts_line_search": 3,
            "line": 3, "search_line": 2, "write": 1,
        }
        base = base_for_ok
        for key in ("_wide", "_0.64", "_0.55"):
            base = base.replace(key, "")
        # Risk score (lower is safer)
        def _risk(ext_l: str, base_l: str, total_l: int) -> float:
            r = max(0.0, total_l / 50.0)
            if ext_l and ext_l != ".py":
                r += 0.5
            if base_l in ("write", "search_line"):
                r += 0.5
            return min(5.0, r)
        risk = _risk(ext, base, total)
        emb_sc = float(emb_meta.get(name, 0.0) or 0.0)
        cg_sc = float(cg_meta.get(name, 0.0) or 0.0)
        # Combine: prefer ok_to_commit, then strategy preference, then higher embedding/cg score, then lower risk, then smaller diff
        return (1 if okc else 0, pref.get(base, 0), int(emb_sc * 1000 + cg_sc * 100), int(-risk * 100), -total)

    # Evaluate in windows to respect timebox while still exploring
    pos = 0
    n = len(candidates)
    while pos < n:
        # timebox check
        if (max_ms_local is not None) and ((time.monotonic() - start_ts) * 1000.0 > max_ms_local):
            break
        batch = candidates[pos: pos + PREV_CONC]
        pos += PREV_CONC
        # launch previews concurrently
        tasks = []
        for name, prev_factory, commit_factory in batch:
            tasks.append((name, commit_factory, asyncio.create_task(_evaluate_candidate(name, prev_factory()))))
        # gather
        done_list = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
        for (name, commit_factory, _), res in zip(tasks, done_list):
            if isinstance(res, Exception):
                continue
            cname, ok, diff, total = res
            if not ok:
                continue
            sc = _score(name, diff, total)
            if best is None or (sc > best["score"]):
                best = {"name": name, "diff": diff, "score": sc, "commit": commit_factory}

    # Fallback sequential evaluation only if none found yet
    if best is None:
        pref_weight = {
            "symbol": 9,
            "symbol_body": 8,
            "semantic": 7,
            "search_semantic": 6,
            "context": 5,
            "anchor": 4,
            "line": 3,
            "search_line": 2,
            "write": 1,
        }
        for name, prev_factory, commit_factory in candidates:
            # timebox unless disabled
            if (max_ms_local is not None) and ((time.monotonic() - start_ts) * 1000.0 > max_ms_local):
                break
            cname, ok, diff, total = await _evaluate_candidate(name, prev_factory())
            if not ok:
                continue
            okc, reason = _should_autocommit(name.replace("search_", "").replace("symbol_body", "symbol"), diff)
            score = (1 if okc else 0, pref_weight.get(name, 0), -total)
            if best is None or (score > best["score"]):
                best = {"name": cname, "diff": diff, "score": score, "commit": commit_factory}

    # If nothing succeeded during preview, attempt last resort paths (old behavior fallbacks)
    if best is None:
        # Try the original simple flow as a final fallback
        if path and code:
            ok, detail = await patch_write(path, code, preview=bool(args.preview))
            return ok, "write", detail
        if args.query:
            limit_ms2 = None if no_budgets else min(max_ms, 300)
            hits = await search_project_cached(exp_query or (args.query or ""), k=1, max_time_ms=limit_ms2)
            if hits:
                h = hits[0]
                fpath = os.path.join(ROOT, h.get("file") or "")
                if fpath:
                    ok, detail = await patch_semantic_in_file(fpath, exp_query or (args.query or ""), code, preview=bool(args.preview))
                    if ok:
                        return True, "search_semantic", detail
                    ok2, detail2 = await patch_line_range(fpath, int(h.get("line_start") or 1), int(h.get("line_end") or 1), code, preview=bool(args.preview), max_span=args.max_span)
                    return ok2, "search_line", detail2
            return False, "search", "no hits"
        return False, "auto", "insufficient arguments for autopatch"

    # We have a best candidate selected by preview. If preview requested, return its diff.
    if args.preview:
        return True, str(best["name"]), str(best["diff"])

    # Commit the chosen candidate
    # Capture original content before commit for rollback (best-effort)
    orig_src: Optional[str] = None
    if path:
        try:
            with open(path, "r", encoding="utf-8") as _f:
                orig_src = _f.read()
        except Exception:
            orig_src = None
    # Admission guard for patch commit
    if _guard is not None:
        async with _guard("patch", timeout_ms=200) as admitted:
            if not admitted:
                if _span is not None:
                    with _span('autopatch.guard_denied', attrs={'op': 'commit', 'ctx': ctx}):
                        pass
                return False, "admission_denied", "patch commit not admitted"
            okc, detailc = await best["commit"]()
    else:
        okc, detailc = await best["commit"]()
    # Background embedding refresh for the affected file (always enabled)
    try:
        if okc and _sha_path is not None:
            # Determine rel path
            relp = None
            if path:
                try:
                    relp = os.path.relpath(path, start=ROOT)
                except Exception:
                    relp = path
            if not relp:
                # Parse from diff +++ b/<file>
                try:
                    for ln in str(best.get("diff") or "").splitlines():
                        if ln.startswith("+++ "):
                            parts = ln.split()
                            if len(parts) >= 2:
                                pp = parts[1]
                                if pp.startswith("b/"):
                                    pp = pp[2:]
                                relp = pp
                                break
                except Exception:
                    relp = None
            if relp:
                abs_p = os.path.join(ROOT, relp)
                if os.path.isfile(abs_p):
                    # Compute sha off thread
                    try:
                        if _run_cpu is not None:
                            sha = await _run_cpu(_sha_path, abs_p)
                        else:
                            sha = await asyncio.to_thread(_sha_path, abs_p)
                    except Exception:
                        sha = None
                    if _enqueue_embed is not None:
                        try:
                            await _enqueue_embed(abs_p, relp, str(sha) if sha else None)
                        except Exception:
                            pass
    except Exception:
        pass
    # Quick verify for Python when path provided (always enabled)
    try:
        if okc and path and str(path).endswith('.py'):
            import py_compile as _pyc
            # Admission guard for verify
            if _guard is not None:
                async with _guard('patch', timeout_ms=150) as admitted:
                    if not admitted:
                        if _span is not None:
                            with _span('autopatch.guard_denied', attrs={'op': 'verify', 'path': path}):
                                pass
                        verify_ok = True  # skip verify when not admitted
                    else:
                        try:
                            _pyc.compile(path, doraise=True)
                            verify_ok = True
                        except Exception:
                            verify_ok = False
            else:
                try:
                    _pyc.compile(path, doraise=True)
                    verify_ok = True
                except Exception:
                    verify_ok = False
            # emit span for verify
            if _span is not None:
                with _span('autopatch.verify_py', attrs={'path': path, 'ok': bool(verify_ok)}):
                    pass
            # rollback on failed quick verify (best-effort)
            if not verify_ok and orig_src is not None:
                try:
                    with open(path, 'w', encoding='utf-8') as _f:
                        _f.write(orig_src)
                except Exception:
                    pass
    except Exception:
        pass
    # Bandit update
    try:
        _bandit_update(ctx, str(best["name"]), bool(okc))
    except Exception:
        pass
    # Observability metrics
    try:
        if _rec_patch is not None:
            add, rem = _diff_stats(str(best.get("diff") or ""))
            _rec_patch(ctx, str(best["name"]), bool(okc), int(add + rem))
    except Exception:
        pass
    # OTEL span on commit
    try:
        if _span is not None:
            add, rem = _diff_stats(str(best.get("diff") or ""))
            attrs = {
                "ctx": ctx,
                "strategy": str(best["name"]),
                "ok": bool(okc),
                "diff_total": int(add + rem),
                "path_ext": (os.path.splitext(path)[1].lower() if path else ""),
                "has_symbol": bool(args.symbol),
                "has_anchor": bool(args.anchor),
                "has_query": bool(args.query),
            }
            with _span("autopatch.commit", attrs=attrs):
                pass
    except Exception:
        pass
    # Record attention on success to reinforce short-term working memory
    try:
        if okc:
            keys: List[str] = []
            # Path attention (relative)
            if path:
                try:
                    relp = os.path.relpath(path, start=ROOT)
                except Exception:
                    relp = path
                keys.append(f"path: {relp}")
            # Symbol attention
            if args.symbol:
                keys.append(f"symbol: {args.symbol}")
            # Query term attention
            qtxt = exp_query or (args.query or "")
            for m in _re.finditer(r"(?u)[\w\.]{3,}", qtxt):
                t = (m.group(0) or "").strip().lower()
                if t and len(t) >= 3:
                    keys.append(f"term: {t}")
            await _att_rec(keys, weight=1.0)
            # Apply KG feedback: nodes and edges among co-activated concepts
            try:
                fb_nodes = [(k, 0.6) for k in keys]
                # build sparse pairwise edges (cap to first 6 to limit work)
                cap = min(6, len(keys))
                fb_edges: List[tuple[str, str, float]] = []
                for i in range(cap):
                    for j in range(i + 1, cap):
                        fb_edges.append((keys[i], keys[j], 0.4))
                _kg_feedback(fb_nodes, fb_edges)
            except Exception:
                pass
            # Persist reinforcement for brain concepts (with decay applied later)
            try:
                await _brain_reinf(keys, weight=1.0)
            except Exception:
                pass
    except Exception:
        pass
    # Report outcome to AutoBrain for learning
    try:
        elapsed_ms = (time.monotonic() - start_ts) * 1000
        _ab_record("autopatch_max_ms", okc, elapsed_ms)
        _ab_record("autopatch_search_topk", okc, elapsed_ms)
    except Exception:
        pass
    return okc, str(best["name"]), detailc
