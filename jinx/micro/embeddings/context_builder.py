from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Tuple
from pathlib import Path

from .project_rerank import rerank_hits_unified
from .project_config import ROOT
from jinx.micro.common.internal_paths import is_restricted_path
from .project_retrieval_config import (
    PROJ_DEFAULT_TOP_K,
    PROJ_SNIPPET_AROUND,
    PROJ_SNIPPET_PER_HIT_CHARS,
    PROJ_TOTAL_CODE_BUDGET,
    PROJ_ALWAYS_FULL_PY_SCOPE,
    PROJ_FULL_SCOPE_TOP_N,
    PROJ_NO_CODE_BUDGET,
    PROJ_CALLGRAPH_ENABLED,
    PROJ_CALLGRAPH_TOP_HITS,
    PROJ_CALLGRAPH_CALLERS_LIMIT,
    PROJ_CALLGRAPH_CALLEES_LIMIT,
    PROJ_CALLGRAPH_TIME_MS,
    PROJ_MAX_FILES,
    PROJ_CONSOLIDATE_PER_FILE,
    PROJ_USAGE_REFS_LIMIT,
)
import jinx.state as jx_state
try:
    from .prefetch_cache import get_project as _pref_get_project
except Exception:
    _pref_get_project = None  # type: ignore[assignment]
try:
    from jinx.micro.runtime.seeds import get_seeds as _get_seeds
except Exception:
    _get_seeds = None  # type: ignore[assignment]
from .project_snippet import build_snippet
from .snippet_cache import make_snippet_cache_key, get_cached_snippet, put_cached_snippet
from .graph_cache import get_symbol_graph_cached, find_usages_cached
from .project_py_scope import get_python_symbol_at_line
from .project_stage_literal import stage_literal_hits
from .api_lens import extract_api_edges as _api_edges
from .project_lang import lang_for_file
from .refs_format import format_usage_ref, format_literal_ref
from jinx.micro.text.heuristics import is_code_like as _is_code_like

from .retrieval_core import (
    retrieve_project_top_k,
    retrieve_project_multi_top_k,
)
from jinx.micro.rt.activity import set_activity_detail as _actdet, clear_activity_detail as _actdet_clear
from jinx.async_utils.fs import read_text_abs_thread
from jinx.micro.memory.unified import assemble_unified_memory_lines as _mem_unified
import jinx.state as jx_state
from jinx.micro.brain.concepts import activate_concepts as _brain_activate
from jinx.micro.brain.attention import record_attention as _att_rec
from jinx.micro.brain.attention import get_attention_weights as _atten_get
from jinx.log_paths import BLUE_WHISPERS
from jinx.logger.file_logger import append_line as _log_append
import re as _re
from jinx.micro.embeddings.context_blocks import (
    build_code_block as _build_code_block,
    build_brain_block as _build_brain_block,
    build_refs_block as _build_refs_block,
    build_graph_block as _build_graph_block,
    build_memory_block as _build_memory_block,
    join_blocks as _join_blocks,
)

# Serialize heavy CPU under throttle
THROTTLE_LOCK: asyncio.Lock = asyncio.Lock()


async def build_project_context_for(query: str, *, k: int | None = None, max_chars: int | None = None, max_time_ms: int | None = 300) -> str:
    """Build project context with adaptive ML-driven parameters."""
    t0 = time.perf_counter()
    
    # Fast-path: if prefetch cache has a recent context for this query, return it.
    try:
        if _pref_get_project is not None:
            pref = _pref_get_project(query)
            if pref:
                # Record cache hit outcome
                try:
                    from jinx.micro.brain.outcome_tracker import record_outcome
                    asyncio.create_task(record_outcome('context_build', True, {'cache_hit': True, 'query_len': len(query)}))
                except Exception:
                    pass
                return pref
    except Exception:
        pass
    
    # Use adaptive retrieval parameters if k not specified
    if k is None:
        try:
            from jinx.micro.brain.adaptive_retrieval import select_retrieval_params
            k_adaptive, timeout_adaptive = await select_retrieval_params(query)
            k = k_adaptive
            if max_time_ms is None:
                max_time_ms = timeout_adaptive
        except Exception:
            k = PROJ_DEFAULT_TOP_K
    else:
        k = k
    # Brain activation: derive unified concepts from project+memory+KG to expand query
    _brain_enable = True
    _brain_topk = 12
    _brain_expand_max = 6

    brain_pairs: list[tuple[str, float]] = []
    brain_terms: list[str] = []
    if _brain_enable:
        try:
            brain_pairs = await _brain_activate(query, top_k=_brain_topk)
            # Build lightweight expansion token list from top concept terms/symbols
            seen_bt: set[str] = set()
            for key, sc in brain_pairs:
                low = (key or "").lower()
                tok = ""
                if low.startswith("term: "):
                    tok = low.split(": ", 1)[1]
                elif low.startswith("symbol: "):
                    tok = low.split(": ", 1)[1]
                # skip path tokens in expansion to avoid over-constraining query
                if tok and tok not in seen_bt:
                    brain_terms.append(tok)
                    seen_bt.add(tok)
                if len(brain_terms) >= _brain_expand_max:
                    break
        except Exception:
            brain_pairs = []
            brain_terms = []

    exp_query = (query or "").strip()
    if brain_terms:
        exp_query = (exp_query + " " + " ".join(brain_terms)).strip()
    # Centralized seeds (cog/foresight/oracle/hypersigil) with TTL + dedupe
    try:
        if _get_seeds is not None:
            seeds_terms = list(_get_seeds(top_n=12) or [])
        else:
            seeds_terms = []
    except Exception:
        seeds_terms = []
    if seeds_terms:
        exp_query = (exp_query + " " + " ".join(seeds_terms)).strip()

    # In parallel, kick off memory retrieval when enabled (biased with brain tokens)
    _mem_enable = True
    _mem_k = 8
    mem_task: asyncio.Task | None = None
    if _mem_enable:
        try:
            _mem_preview = 160
            mem_task = asyncio.create_task(_mem_unified(exp_query, k=_mem_k, preview_chars=_mem_preview))
        except Exception:
            mem_task = None

    # Retrieve hits for original, brain-expanded, and opportunistically memory-expanded queries, then merge
    hits_base = await retrieve_project_top_k(query, k=k, max_time_ms=max_time_ms)
    
    # Debug logging
    try:
        from jinx.micro.logger.debug_logger import debug_log
        await debug_log(f"retrieve_project_top_k('{query[:50]}') returned {len(hits_base)} hits", "EMBEDDINGS")
    except Exception:
        pass
    
    hits_exp: list[tuple[float, str, dict]] = []
    if exp_query != (query or ""):
        try:
            hits_exp = await retrieve_project_top_k(exp_query, k=k, max_time_ms=max_time_ms)
            try:
                await debug_log(f"retrieve_project_top_k(expanded) returned {len(hits_exp)} hits", "EMBEDDINGS")
            except Exception:
                pass
        except Exception:
            hits_exp = []
    # Opportunistic memory-derived expansion (without blocking): extract quick tokens from mem_task if ready
    hits_mem: list[tuple[float, str, dict]] = []
    mem_query = (query or "").strip()
    if mem_task is not None:
        try:
            mem_terms: list[str] = []
            if mem_task.done():
                mem_lines = await mem_task
                try:
                    # Quick token pass over a few lines to avoid latency; prefer alnum terms >=3 chars
                    seen_mt: set[str] = set()
                    for ln in (mem_lines or [])[:8]:
                        for m in _re.finditer(r"(?u)[\w\.]{3,}", ln or ""):
                            tok = (m.group(0) or "").strip().lower()
                            if tok and tok not in seen_mt:
                                mem_terms.append(tok)
                                seen_mt.add(tok)
                            if len(mem_terms) >= 8:
                                break
                        if len(mem_terms) >= 8:
                            break
                except Exception:
                    mem_terms = []
            if mem_terms:
                mem_query = (mem_query + " " + " ".join(mem_terms)).strip()
                if mem_query != (query or "") and mem_query != exp_query:
                    try:
                        # Use a smaller time budget for the memory-augmented pass
                        mm = int((max_time_ms or 300) * 0.6)
                    except Exception:
                        mm = max_time_ms or 300
                    try:
                        hits_mem = await retrieve_project_top_k(mem_query, k=k, max_time_ms=mm)
                    except Exception:
                        hits_mem = []
        except Exception:
            hits_mem = []
    # Merge with dedupe by (file_rel, ls, le)
    def _kof(h: tuple[float, str, dict]) -> tuple:
        sc, rel, obj = h
        m = (obj.get("meta") or {})
        return (str(m.get("file_rel") or rel), int(m.get("line_start") or 0), int(m.get("line_end") or 0))
    seen_k: set[tuple] = set()
    hits: list[tuple[float, str, dict]] = []
    for lst in (hits_base or []), (hits_exp or []), (hits_mem or []):
        for h in lst:
            kx = _kof(h)
            if kx in seen_k:
                continue
            seen_k.add(kx)
            hits.append(h)
    if not hits:
        return ""
    # Unified rerank with source-aware + KG-aware boosts
    hits_sorted = rerank_hits_unified(hits, query)
    parts: List[str] = []
    refs_parts: List[str] = []
    graph_parts: List[str] = []
    brain_parts: List[str] = []
    seen: set[str] = set()  # dedupe by preview text
    headers_seen: set[str] = set()  # dedupe by [file:ls-le]
    refs_headers_seen: set[str] = set()  # dedupe refs by header
    graph_headers_seen: set[str] = set()  # dedupe graph entries by header
    included_files: set[str] = set()

    # Build per-file centers from all hits to allow multi-segment snippets to include other hotspots
    file_hit_centers: Dict[str, List[int]] = {}
    for sc, fr, obj in hits_sorted:
        try:
            m = (obj.get("meta") or {})
            ls = int(m.get("line_start") or 0)
            le = int(m.get("line_end") or 0)
            c = int((ls + le) // 2) if (ls and le) else int(ls or le or 0)
            if c > 0:
                file_hit_centers.setdefault(fr, []).append(c)
        except Exception:
            continue

    # Disable total code budget if configured
    budget = None if PROJ_NO_CODE_BUDGET else (PROJ_TOTAL_CODE_BUDGET if (max_chars is None) else max_chars)
    try:
        _actdet({"hits": len(hits_sorted), "tasks": "0/0", "budget": (budget or 0)})
    except Exception:
        pass
    total_len = 0
    # Per-call cache to avoid re-reading the same file multiple times
    file_text_cache: Dict[str, str] = {}

    full_scope_used = 0
    codey_query = _is_code_like(query or "")  # currently informational; future heuristics may use it
    # Parallel snippet building with bounded concurrency
    _SNIP_CONC = 4
    sem = asyncio.Semaphore(_SNIP_CONC)
    # Additional throttled semaphore used only when system saturation is detected
    _SNIP_CONC_THR = 1
    throttled_sem = asyncio.Semaphore(_SNIP_CONC_THR)

    prepared: List[Tuple[int, str, Dict[str, Any], bool, List[int]]] = []  # (idx, file_rel, meta, prefer_full, extra_centers_abs)
    for idx, (score, file_rel, obj) in enumerate(hits_sorted):
        # Skip restricted files defensively (.jinx, log, etc.) and dedupe by preview text
        try:
            if is_restricted_path(str(file_rel or "")):
                continue
        except Exception:
            pass
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").strip()
        if pv and pv in seen:
            continue
        if pv:
            seen.add(pv)
        prefer_full = PROJ_ALWAYS_FULL_PY_SCOPE and (
            PROJ_FULL_SCOPE_TOP_N <= 0 or (full_scope_used < PROJ_FULL_SCOPE_TOP_N)
        )
        try:
            extra_centers_abs = sorted({int(x) for x in (file_hit_centers.get(file_rel) or []) if int(x) > 0})
        except Exception:
            extra_centers_abs = []
        prepared.append((idx, file_rel, meta, prefer_full, extra_centers_abs))

    async def _build(idx_i: int, file_rel_i: str, meta_i: Dict[str, Any], prefer_full_i: bool, centers_i: List[int]):
        async with sem:
            # Try cache first inside the worker to avoid main loop blocking
            def _run():
                key = make_snippet_cache_key(
                    file_rel_i,
                    meta_i,
                    query,
                    prefer_full_scope=prefer_full_i,
                    expand_callees=True,
                    extra_centers_abs=centers_i,
                )
                cached = get_cached_snippet(key)
                if cached is not None:
                    hdr_c, code_c, ls_c, le_c, is_full_c = cached
                    return (hdr_c, code_c, ls_c, le_c, is_full_c)
                res = build_snippet(
                    file_rel_i,
                    meta_i,
                    query,
                    max_chars=PROJ_SNIPPET_PER_HIT_CHARS,
                    prefer_full_scope=prefer_full_i,
                    expand_callees=True,
                    extra_centers_abs=centers_i,
                )
                try:
                    put_cached_snippet(key, res)
                except Exception:
                    pass
                return res
            # Under throttle, serialize heavy CPU via lock; else proceed normally
            if jx_state.throttle_event.is_set():
                async with THROTTLE_LOCK:
                    hdr, code, ls, le, is_full = await asyncio.to_thread(_run)
            else:
                hdr, code, ls, le, is_full = await asyncio.to_thread(_run)
            return (idx_i, file_rel_i, meta_i, hdr, code, ls, le, is_full)

    tasks = [asyncio.create_task(_build(*args)) for args in prepared]
    # Progress as tasks complete
    done = 0
    results: List[Tuple[int, str, Dict[str, Any], str, str, int, int, bool]] = []
    # Dosing config for throttled mode
    _DOSE_BATCH = 4
    _DOSE_MS = 8

    approx_len = 0
    for fut in asyncio.as_completed(tasks):
        try:
            r = await fut
        except Exception:
            continue
        results.append(r)
        done += 1
        try:
            _actdet({"hits": len(hits_sorted), "tasks": f"{done}/{len(prepared)}", "budget": (budget or 0)})
        except Exception:
            pass
        # Early cancellation if overall budget will be exceeded by accumulated snippet sizes
        if budget is not None:
            try:
                snip_len = len(r[3] or "") + len(r[4] or "")
            except Exception:
                snip_len = 0
            approx_len += snip_len
            if approx_len > budget and done > 0:
                # Cancel remaining tasks to save resources
                pend = [t for t in tasks if not t.done()]
                for t in pend:
                    t.cancel()
                if pend:
                    try:
                        await asyncio.gather(*pend, return_exceptions=True)
                    except Exception:
                        pass
                break
        if jx_state.throttle_event.is_set():
            if (done % _DOSE_BATCH) == 0:
                await asyncio.sleep(_DOSE_MS / 1000.0)
        else:
            if (done % 5) == 0:
                await asyncio.sleep(0)
    # Assemble in original order, enforcing budget and per-file consolidation
    for r in sorted(results, key=lambda t: t[0]):
        idx, file_rel, meta, header, code_block, use_ls, use_le, is_full_scope = r
        if PROJ_CONSOLIDATE_PER_FILE and file_rel in included_files:
            continue
        snippet_text = f"{header}\n{code_block}"
        if header in headers_seen:
            continue
        headers_seen.add(header)
        if budget is not None:
            would = total_len + len(snippet_text)
            if (not is_full_scope or not PROJ_ALWAYS_FULL_PY_SCOPE) and would > budget:
                if not parts:
                    parts.append(snippet_text)
                break
            total_len = would
        try:
            _actdet({"hits": len(hits_sorted), "tasks": f"{done}/{len(prepared)}", "budget": (budget or 0), "collected": total_len})
        except Exception:
            pass
        parts.append(snippet_text)
        if PROJ_CONSOLIDATE_PER_FILE:
            included_files.add(file_rel)
        if is_full_scope:
            full_scope_used += 1
        # API lens enrichment (lightweight, env-gated)
        _api_on = True
        if _api_on and file_rel.endswith('.py'):
            try:
                ap = _api_edges(file_rel, header, code_block)
                if ap is not None:
                    hdr_api, block_api = ap
                    if hdr_api not in graph_headers_seen:
                        graph_headers_seen.add(hdr_api)
                        graph_parts.append(f"{hdr_api}\n{block_api}")
            except Exception:
                pass
        # Optional callgraph enrichment for top hits (Python only)
        try:
            if PROJ_CALLGRAPH_ENABLED and file_rel.endswith('.py') and idx < max(0, PROJ_CALLGRAPH_TOP_HITS):
                pairs = await get_symbol_graph_cached(
                    file_rel,
                    use_ls or 0,
                    use_le or 0,
                    callers_limit=PROJ_CALLGRAPH_CALLERS_LIMIT,
                    callees_limit=PROJ_CALLGRAPH_CALLEES_LIMIT,
                    around=PROJ_SNIPPET_AROUND,
                    scan_cap_files=PROJ_MAX_FILES,
                    time_budget_ms=PROJ_CALLGRAPH_TIME_MS,
                )
                for hdr2, block in (pairs or []):
                    if hdr2 in graph_headers_seen:
                        continue
                    graph_headers_seen.add(hdr2)
                    graph_parts.append(f"{hdr2}\n{block}")
        except Exception:
            pass
        # Optionally add a couple of usage references for the enclosing symbol (Python only)
        try:
            # Allow env override for usage references limit
            usage_limit = PROJ_USAGE_REFS_LIMIT

            async def _collect_usages() -> list[tuple[str, str]]:
                out: list[tuple[str, str]] = []
                try:
                    file_text = file_text_cache.get(file_rel, "")
                    if not file_text:
                        try:
                            if jx_state.throttle_event.is_set():
                                async with THROTTLE_LOCK:
                                    file_text = await read_text_abs_thread(str(Path(ROOT) / file_rel))
                            else:
                                file_text = await read_text_abs_thread(str(Path(ROOT) / file_rel))
                            if file_text:
                                file_text_cache[file_rel] = file_text
                        except Exception:
                            file_text = ""
                    if file_rel.endswith('.py') and file_text:
                        cand_line = int((use_ls + use_le) // 2) if (use_ls and use_le) else int(use_ls or use_le or 0)
                        def _sym():
                            return get_python_symbol_at_line(file_text, cand_line)
                        if jx_state.throttle_event.is_set():
                            async with THROTTLE_LOCK:
                                sym_name, sym_kind = await asyncio.to_thread(_sym)
                        else:
                            sym_name, sym_kind = await asyncio.to_thread(_sym)
                        if sym_name:
                            usages = await find_usages_cached(sym_name, file_rel, limit=usage_limit, around=PROJ_SNIPPET_AROUND)
                            for fr, ua, ub, usnip, ulang in usages:
                                try:
                                    hdrx, blockx = format_usage_ref(
                                        sym_name,
                                        sym_kind,
                                        fr,
                                        int(ua or 0),
                                        int(ub or 0),
                                        usnip or "",
                                        ulang,
                                        origin_file=file_rel,
                                        origin_ls=int(use_ls or 0),
                                        origin_le=int(use_le or 0),
                                    )
                                except Exception:
                                    langx = ulang
                                    hdrx = f"[{fr}:{ua}-{ub}]"
                                    blockx = f"```{langx}\n{usnip}\n```" if langx else f"```\n{usnip}\n```"
                                out.append((hdrx, blockx))
                    # Fallback: literal-occurrences refs when no symbol usages were found
                    if not out and (query or "").strip():
                        # Literal refs collection tuning via env
                        _lim = 6 if _is_code_like(query or "") else 3
                        _ms = 300 if _is_code_like(query or "") else 200
                        def _lit_call():
                            try:
                                return stage_literal_hits(query, _lim, max_time_ms=_ms)
                            except Exception:
                                return []
                        lit_hits = await asyncio.to_thread(_lit_call)
                        for _sc2, rel2, obj2 in (lit_hits or [])[:_lim]:
                            try:
                                meta2 = (obj2.get("meta") or {})
                                ls2 = int(meta2.get("line_start") or 0)
                                le2 = int(meta2.get("line_end") or 0)
                                prev = (meta2.get("text_preview") or "").strip()
                                if not prev:
                                    continue
                                lang2 = lang_for_file(rel2)
                                try:
                                    hdrx, blockx = format_literal_ref(
                                        query,
                                        str(rel2),
                                        int(ls2 or 0),
                                        int(le2 or 0),
                                        prev,
                                        lang2,
                                        origin_file=file_rel,
                                        origin_ls=int(use_ls or 0),
                                        origin_le=int(use_le or 0),
                                    )
                                except Exception:
                                    hdrx = f"[{rel2}:{ls2}-{le2}]"
                                    blockx = f"```{lang2}\n{prev}\n```" if lang2 else f"```\n{prev}\n```"
                                out.append((hdrx, blockx))
                            except Exception:
                                continue
                except Exception:
                    return out
                return out

            pairs = await _collect_usages()
            for hdr3, block3 in pairs:
                if hdr3 in refs_headers_seen:
                    continue
                refs_headers_seen.add(hdr3)
                refs_parts.append(f"{hdr3}\n{block3}")
        except Exception:
            pass

    # Optionally include memory results even if code gathered nothing
    mem_parts: List[str] = []
    if _mem_enable and mem_task is not None:
        try:
            _mem_budget = 1200
            mem_lines = await mem_task  # unified lines already deduped and clamped
            if mem_lines:
                acc: List[str] = []
                total = 0
                for ln in mem_lines:
                    L = len(ln) + 1
                    if total + L > _mem_budget:
                        break
                    acc.append(ln)
                    total += L
                if acc:
                    mem_parts.append("\n".join(acc))
        except Exception:
            pass

    if not parts and mem_parts:
        return _build_memory_block(mem_parts)
    if not parts:
        return ""
    body = "\n".join(parts)
    code_block = _build_code_block([body])
    brain_block = ""
    try:
        if _brain_enable and brain_pairs:
            brain_block = _build_brain_block(brain_pairs)
    except Exception:
        brain_block = ""
    # Refs policy gating and size budget to avoid unnecessary tokens
    # Default to 'always' so references are visible by default; can be tuned via env
    refs_policy = "always"
    refs_min = 2
    refs_max_chars = 1600

    def _should_send_refs(codey: bool, count: int) -> bool:
        if refs_policy in ("never", "0", "off", "false", ""):
            return False
        if refs_policy in ("always", "1", "on", "true"):
            return True
        return bool(codey) or (count >= refs_min)

    refs_block = _build_refs_block(refs_parts, policy=refs_policy, refs_min=refs_min, refs_max_chars=refs_max_chars, codey=_is_code_like(query or ""))
    graph_block = _build_graph_block(graph_parts)
    mem_block = _build_memory_block(mem_parts)
    final_text = _join_blocks([code_block, brain_block, refs_block, graph_block, mem_block])
    try:
        _actdet_clear()
    except Exception:
        pass
    # Record attention: included code paths and query terms
    try:
        keys: list[str] = []
        for fr in included_files:
            keys.append(f"path: {fr}")
        for m in _re.finditer(r"(?u)[\\w\\.]{3,}", (query or "")):
            t = (m.group(0) or "").strip().lower()
            if t and len(t) >= 3:
                keys.append(f"term: {t}")
        if keys:
            await _att_rec(keys, weight=1.0)
    except Exception:
        pass
    
    # Record outcome for learning
    elapsed_ms = (time.perf_counter() - t0) * 1000
    try:
        from jinx.micro.brain.outcome_tracker import record_outcome
        success = len(parts) > 0
        asyncio.create_task(record_outcome(
            'context_build',
            success,
            {
                'query_len': len(query),
                'k': k,
                'hits': len(hits_sorted) if 'hits_sorted' in locals() else 0,
                'parts': len(parts),
                'total_chars': len(final_text),
                'brain_enabled': _brain_enable if '_brain_enable' in locals() else False,
            },
            latency_ms=elapsed_ms
        ))
    except Exception:
        pass
    
    return final_text


__all__ = [
    "build_project_context_for",
    "build_project_context_multi_for",
]


async def build_project_context_multi_for(queries: List[str], *, k: int | None = None, max_chars: int | None = None, max_time_ms: int | None = 300) -> str:
    k_eff = k or PROJ_DEFAULT_TOP_K
    per_query_k = max(1, int((k_eff + max(1, len(queries)) - 1) // max(1, len(queries))))
    hits = await retrieve_project_multi_top_k(queries, per_query_k=per_query_k, max_time_ms=max_time_ms)
    if not hits:
        return ""
    # Re-rank across all hits by combined query string
    hits_sorted = rerank_hits_unified(hits, " ".join(queries))
    parts: List[str] = []
    refs_parts: List[str] = []
    graph_parts: List[str] = []
    seen: set[str] = set()
    headers_seen: set[str] = set()
    refs_headers_seen: set[str] = set()
    graph_headers_seen: set[str] = set()
    included_files: set[str] = set()
    budget = None if PROJ_NO_CODE_BUDGET else (PROJ_TOTAL_CODE_BUDGET if (max_chars is None) else max_chars)
    total_len = 0

    full_scope_used = 0
    # Build per-file centers from all hits to allow multi-segment snippets to include other hotspots
    file_hit_centers: Dict[str, List[int]] = {}
    for sc, fr, obj in hits_sorted:
        try:
            m = (obj.get("meta") or {})
            ls = int(m.get("line_start") or 0)
            le = int(m.get("line_end") or 0)
            c = int((ls + le) // 2) if (ls and le) else int(ls or le or 0)
            if c > 0:
                file_hit_centers.setdefault(fr, []).append(c)
        except Exception:
            continue

    # Parallel snippet building with bounded semaphore
    _SNIP_CONC = 4
    sem = asyncio.Semaphore(_SNIP_CONC)

    q_join = " ".join(queries)[:512]
    # Include seeds into the combined query to bias multi retrieval
    try:
        if _get_seeds is not None:
            s_terms = _get_seeds(top_n=10)
        else:
            s_terms = []
    except Exception:
        s_terms = []
    if s_terms:
        q_join = (q_join + " " + " ".join(s_terms))[:768]
    codey_join = _is_code_like(q_join or "")
    # Brain activation (multi)
    _brain_enable_m = True
    _brain_topk_m = 12
    brain_pairs_m: list[tuple[str, float]] = []
    if _brain_enable_m:
        try:
            brain_pairs_m = await _brain_activate(q_join, top_k=_brain_topk_m)
        except Exception:
            brain_pairs_m = []
    # Memory retrieval (multi)
    _mem_enable_m = True
    _mem_k_m = 8
    mem_task_m: asyncio.Task | None = None
    if _mem_enable_m:
        try:
            _mem_preview_m = 160
            mem_task_m = asyncio.create_task(_mem_unified(q_join, k=_mem_k_m, preview_chars=_mem_preview_m))
        except Exception:
            mem_task_m = None
    prepared: List[Tuple[int, str, Dict[str, Any], bool, List[int]]] = []
    for idx, (score, file_rel, obj) in enumerate(hits_sorted):
        try:
            if is_restricted_path(str(file_rel or "")):
                continue
        except Exception:
            pass
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").strip()
        if pv and pv in seen:
            continue
        if pv:
            seen.add(pv)
        prefer_full = PROJ_ALWAYS_FULL_PY_SCOPE and (
            PROJ_FULL_SCOPE_TOP_N <= 0 or (full_scope_used < PROJ_FULL_SCOPE_TOP_N)
        )
        try:
            extra_centers_abs = sorted({int(x) for x in (file_hit_centers.get(file_rel) or []) if int(x) > 0})
        except Exception:
            extra_centers_abs = []
        prepared.append((idx, file_rel, meta, prefer_full, extra_centers_abs))

    async def _build(idx_i: int, file_rel_i: str, meta_i: Dict[str, Any], prefer_full_i: bool, centers_i: List[int]):
        async with sem:
            def _run():
                return build_snippet(
                    file_rel_i,
                    meta_i,
                    q_join,
                    max_chars=PROJ_SNIPPET_PER_HIT_CHARS,
                    prefer_full_scope=prefer_full_i,
                    expand_callees=True,
                    extra_centers_abs=centers_i,
                )
            hdr, code, ls, le, is_full = await asyncio.to_thread(_run)
            return (idx_i, file_rel_i, meta_i, hdr, code, ls, le, is_full)

    tasks = [asyncio.create_task(_build(*args)) for args in prepared]
    # Progress as tasks complete
    done = 0
    results: List[Tuple[int, str, Dict[str, Any], str, str, int, int, bool]] = []
    try:
        _actdet({"hits": len(hits_sorted), "tasks": f"0/{len(prepared)}"})
    except Exception:
        pass
    # Adaptive dosing in multi-query builder as well
    _DOSE_BATCH_M = 4
    _DOSE_MS_M = 8

    approx_len_m = 0
    for fut in asyncio.as_completed(tasks):
        try:
            r = await fut
        except Exception:
            continue
        results.append(r)
        done += 1
        try:
            _actdet({"hits": len(hits_sorted), "tasks": f"{done}/{len(prepared)}"})
        except Exception:
            pass
        if jx_state.throttle_event.is_set():
            if (done % _DOSE_BATCH_M) == 0:
                await asyncio.sleep(_DOSE_MS_M / 1000.0)
        else:
            if (done % 5) == 0:
                await asyncio.sleep(0)
        # Early cancellation on approximate budget in multi-query
        try:
            snip_len_m = len(r[3] or "") + len(r[4] or "")
        except Exception:
            snip_len_m = 0
        approx_len_m += snip_len_m
        if budget is not None and approx_len_m > budget and done > 0:
            pend = [t for t in tasks if not t.done()]
            for t in pend:
                t.cancel()
            if pend:
                try:
                    await asyncio.gather(*pend, return_exceptions=True)
                except Exception:
                    pass
            break
    for r in sorted(results, key=lambda t: t[0]):
        idx, file_rel, meta, header, code_block, use_ls, use_le, is_full_scope = r
        if PROJ_CONSOLIDATE_PER_FILE and file_rel in included_files:
            continue
        snippet_text = f"{header}\n{code_block}"
        if header in headers_seen:
            continue
        headers_seen.add(header)
        if budget is not None:
            would = total_len + len(snippet_text)
            if (not is_full_scope or not PROJ_ALWAYS_FULL_PY_SCOPE) and would > budget:
                if not parts:
                    parts.append(snippet_text)
                break
            total_len = would
        parts.append(snippet_text)
        if PROJ_CONSOLIDATE_PER_FILE:
            included_files.add(file_rel)
        if is_full_scope:
            full_scope_used += 1
        # Optional callgraph enrichment for top hits (Python only)
        try:
            if PROJ_CALLGRAPH_ENABLED and file_rel.endswith('.py') and idx < max(0, PROJ_CALLGRAPH_TOP_HITS):
                pairs = await get_symbol_graph_cached(
                    file_rel,
                    use_ls or 0,
                    use_le or 0,
                    callers_limit=PROJ_CALLGRAPH_CALLERS_LIMIT,
                    callees_limit=PROJ_CALLGRAPH_CALLEES_LIMIT,
                    around=PROJ_SNIPPET_AROUND,
                    scan_cap_files=PROJ_MAX_FILES,
                    time_budget_ms=PROJ_CALLGRAPH_TIME_MS,
                )
                for hdr2, block in (pairs or []):
                    if hdr2 in graph_headers_seen:
                        continue
                    graph_headers_seen.add(hdr2)
                    graph_parts.append(f"{hdr2}\n{block}")
        except Exception:
            pass
        # Optionally add a couple of usage references for the enclosing symbol (Python only)
        try:
            # Per-call cache to avoid re-reading same files
            file_text_cache: Dict[str, str] = {}

            async def _collect_usages() -> list[tuple[str, str]]:
                out: list[tuple[str, str]] = []
                try:
                    file_text = file_text_cache.get(file_rel, "")
                    if not file_text:
                        try:
                            if jx_state.throttle_event.is_set():
                                async with THROTTLE_LOCK:
                                    file_text = await read_text_abs_thread(str(Path(ROOT) / file_rel))
                            else:
                                file_text = await read_text_abs_thread(str(Path(ROOT) / file_rel))
                            if file_text:
                                file_text_cache[file_rel] = file_text
                        except Exception:
                            file_text = ""
                    if file_rel.endswith('.py') and file_text:
                        cand_line = int((use_ls + use_le) // 2) if (use_ls and use_le) else int(use_ls or use_le or 0)
                        def _sym():
                            return get_python_symbol_at_line(file_text, cand_line)
                        if jx_state.throttle_event.is_set():
                            async with THROTTLE_LOCK:
                                sym_name, sym_kind = await asyncio.to_thread(_sym)
                        else:
                            sym_name, sym_kind = await asyncio.to_thread(_sym)
                        if sym_name:
                            usages = await find_usages_cached(sym_name, file_rel, limit=PROJ_USAGE_REFS_LIMIT, around=PROJ_SNIPPET_AROUND)
                            for fr, ua, ub, usnip, ulang in usages:
                                try:
                                    hdrx, blockx = format_usage_ref(
                                        sym_name,
                                        sym_kind,
                                        fr,
                                        int(ua or 0),
                                        int(ub or 0),
                                        usnip or "",
                                        ulang,
                                        origin_file=file_rel,
                                        origin_ls=int(use_ls or 0),
                                        origin_le=int(use_le or 0),
                                    )
                                except Exception:
                                    langx = ulang
                                    hdrx = f"[{fr}:{ua}-{ub}]"
                                    blockx = f"```{langx}\n{usnip}\n```" if langx else f"```\n{usnip}\n```"
                                out.append((hdrx, blockx))
                    # Fallback: literal-occurrences refs when no symbol usages were found
                    if not out and (q_join or "").strip():
                        # Literal refs collection tuning via env
                        _lim = 6 if _is_code_like(q_join or "") else 3
                        _ms = 300 if _is_code_like(q_join or "") else 200
                        def _lit_call():
                            try:
                                return stage_literal_hits(q_join, _lim, max_time_ms=_ms)
                            except Exception:
                                return []
                        lit_hits = await asyncio.to_thread(_lit_call)
                        for _sc2, rel2, obj2 in (lit_hits or [])[:_lim]:
                            try:
                                meta2 = (obj2.get("meta") or {})
                                ls2 = int(meta2.get("line_start") or 0)
                                le2 = int(meta2.get("line_end") or 0)
                                prev = (meta2.get("text_preview") or "").strip()
                                if not prev:
                                    continue
                                lang2 = lang_for_file(rel2)
                                try:
                                    hdrx, blockx = format_literal_ref(
                                        q_join,
                                        str(rel2),
                                        int(ls2 or 0),
                                        int(le2 or 0),
                                        prev,
                                        lang2,
                                        origin_file=file_rel,
                                        origin_ls=int(use_ls or 0),
                                        origin_le=int(use_le or 0),
                                    )
                                except Exception:
                                    hdrx = f"[{rel2}:{ls2}-{le2}]"
                                    blockx = f"```{lang2}\n{prev}\n```" if lang2 else f"```\n{prev}\n```"
                                out.append((hdrx, blockx))
                            except Exception:
                                continue
                except Exception:
                    return out
                return out

            pairs = await _collect_usages()
            for hdr3, block3 in pairs:
                if hdr3 in refs_headers_seen:
                    continue
                refs_headers_seen.add(hdr3)
                refs_parts.append(f"{hdr3}\n{block3}")
        except Exception:
            pass

    # Optionally include memory results even if code gathered nothing (multi)
    mem_parts: List[str] = []
    if _mem_enable_m and mem_task_m is not None:
        try:
            _mem_budget_m = 1200
            mem_lines_m = await mem_task_m
            if mem_lines_m:
                acc_m: List[str] = []
                total_m = 0
                for ln in mem_lines_m:
                    L = len(ln) + 1
                    if total_m + L > _mem_budget_m:
                        break
                    acc_m.append(ln)
                    total_m += L
                if acc_m:
                    mem_parts.append("\n".join(acc_m))
        except Exception:
            pass

    if not parts and mem_parts:
        return _build_memory_block(mem_parts)
    if not parts:
        return ""
    body = "\n".join(parts)
    code_block = _build_code_block([body])
    brain_block = ""
    try:
        if _brain_enable_m and brain_pairs_m:
            brain_block = _build_brain_block(brain_pairs_m)
    except Exception:
        brain_block = ""
    # Optional brain dump to logs (multi)
    try:
        dump_on = False
        if dump_on:
            await _log_append(BLUE_WHISPERS, "[brain] multi-query dump:")
            for kkey, sc in (brain_pairs_m or [])[:8]:
                await _log_append(BLUE_WHISPERS, f"[brain.top] {kkey} ({sc:.2f})")
            att = _atten_get()
            if att:
                top_att = sorted(att.items(), key=lambda kv: -float(kv[1] or 0.0))[:8]
                for k, v in top_att:
                    if float(v or 0.0) > 0.0:
                        await _log_append(BLUE_WHISPERS, f"[atten] {k}={v:.3f}")
    except Exception:
        pass
    # Refs policy gating and size budget (multi-query). Default to 'always' so refs are visible by default.
    refs_policy = "always"
    refs_min = 2
    refs_max_chars = 1600

    def _should_send_refs_multi(codey: bool, count: int) -> bool:
        if refs_policy in ("never", "0", "off", "false", ""):
            return False
        if refs_policy in ("always", "1", "on", "true"):
            return True
        return bool(codey) or (count >= refs_min)

    refs_block = _build_refs_block(refs_parts, policy=refs_policy, refs_min=refs_min, refs_max_chars=refs_max_chars, codey=codey_join)
    graph_block = _build_graph_block(graph_parts)
    mem_block = _build_memory_block(mem_parts)
    # Record attention: included code paths and query terms (multi)
    try:
        keys2: list[str] = []
        for fr in included_files:
            keys2.append(f"path: {fr}")
        q_join = " ".join(queries)[:512]
        for m in _re.finditer(r"(?u)[\\w\\.]{3,}", q_join):
            t = (m.group(0) or "").strip().lower()
            if t and len(t) >= 3:
                keys2.append(f"term: {t}")
        if keys2:
            await _att_rec(keys2, weight=1.0)
    except Exception:
        pass
    return _join_blocks([code_block, brain_block, refs_block, graph_block, mem_block])

