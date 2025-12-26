from __future__ import annotations

import asyncio
import time
from typing import List, Tuple, Dict, Any
import hashlib

# AutoBrain adaptive configuration
try:
    from jinx.micro.runtime.autobrain_config import (
        get_int as _ab_int,
        get as _ab_get,
        record_outcome as _ab_record,
    )
    _AB_AVAILABLE = True
except Exception:
    _AB_AVAILABLE = False
    def _ab_int(name, ctx=None): return 5
    def _ab_get(name, ctx=None): return 0.25
    def _ab_record(name, success, latency_ms=0, ctx=None): pass

def _recent_items_snapshot():
    try:
        from jinx.micro.embeddings.pipeline import iter_recent_items as _it
    except Exception:
        return []
    try:
        return list(_it())
    except Exception:
        return []
import jinx.state as jx_state
from .paths import EMBED_ROOT
from .similarity import score_cosine_batch
from .text_clean import is_noise_text
from .scan_store import iter_items as scan_iter_items
from .embed_cache import embed_text_cached
from jinx.micro.runtime.task_ctx import get_current_group as _get_group
from .project_terms import extract_terms as _extract_terms
from .ann_index_runtime import search_ann_items as _search_ann_runtime
from jinx.micro.rt.slicer import TimeSlicer
try:
    from .prefetch_cache import get_base as _pref_get_base
except Exception:
    _pref_get_base = None  # type: ignore[assignment]
try:
    from jinx.micro.runtime.seeds import get_seeds as _get_seeds
except Exception:
    _get_seeds = None  # type: ignore[assignment]

DEFAULT_TOP_K = 5
# Balanced defaults; adapt at runtime via AutoBrain
SCORE_THRESHOLD = 0.25
MIN_PREVIEW_LEN = 8
MAX_FILES_PER_SOURCE = 500
MAX_SOURCES = 50
QUERY_MODEL = "text-embedding-3-small"
RECENCY_WINDOW_SEC = 24 * 3600
EXHAUSTIVE = False

# Shared hot-store for runtime items
_HOT_TTL_MS = 1500
from .hot_store import get_runtime_items_hot

async def _load_runtime_items() -> List[Tuple[str, Dict[str, Any]]]:
    return await asyncio.to_thread(scan_iter_items, EMBED_ROOT, MAX_FILES_PER_SOURCE, MAX_SOURCES)


async def _embed_query(text: str) -> List[float]:
    try:
        # Shared cached embedding call with TTL, coalescing, concurrency limit and timeout
        return await embed_text_cached(text, model=QUERY_MODEL)
    except Exception:
        # Best-effort: return empty vector on API failure
        return []


def _iter_items() -> List[Tuple[str, Dict[str, Any]]]:
    # Delegate on-disk scanning to a dedicated helper for clarity and reuse
    return scan_iter_items(EMBED_ROOT, MAX_FILES_PER_SOURCE, MAX_SOURCES)


def _terms_of(s: str) -> List[str]:
    try:
        return _extract_terms((s or "").strip().lower())
    except Exception:
        return []


def _term_overlap(q_terms: List[str], meta_terms: List[str]) -> float:
    if not q_terms or not meta_terms:
        return 0.0
    qs = set(q_terms)
    ms = set([t.lower() for t in meta_terms if t])
    inter = qs.intersection(ms)
    return float(len(inter)) / float(max(1, len(qs)))


async def retrieve_top_k(query: str, k: int | None = None, *, max_time_ms: int | None = 200) -> List[Tuple[float, str, Dict[str, Any]]]:
    # Adapt parameters based on query length (short queries get lower threshold and higher k)
    q = (query or "").strip()
    qlen = len(q)
    thr = SCORE_THRESHOLD
    k_eff = k or DEFAULT_TOP_K
    if qlen <= 12:
        thr = max(0.15, thr * 0.8)
        k_eff = max(k_eff, 8)
    elif qlen <= 24:
        thr = max(0.2, thr)
        k_eff = max(k_eff, 6)

    # Overlap query embedding with a hot-store refresh to reduce wall time
    qv_task = asyncio.create_task(_embed_query(query))
    hot_task = asyncio.create_task(get_runtime_items_hot(_load_runtime_items, _HOT_TTL_MS))
    qv = await qv_task
    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    now = time.time()
    t0 = time.perf_counter()
    # Precompute query terms once
    q_terms = _terms_of(q)
    # Time-slice gating: cooperative yield via TimeSlicer to keep UI responsive
    _SLICE_MS = 12
    ts = TimeSlicer(ms=_SLICE_MS)
    # Session affinity (prefer current per-turn group; fallback to env)
    try:
        _sess_cv = (_get_group() or "").strip()
        sess = _sess_cv or None
    except Exception:
        sess = None

    # 1) Fast-path: score in-memory recent items first
    state_boost = 1.1
    state_rec_mult = 0.5
    short_q = (qlen <= 80)
    _recent_objs: List[Dict[str, Any]] = []
    _recent_vecs: List[List[float]] = []
    _recent_meta: List[Dict[str, Any]] = []
    for obj in _recent_items_snapshot():
        meta = obj.get("meta", {})
        src_l = (meta.get("source") or "").strip().lower()
        if not (src_l == "dialogue" or src_l.startswith("sandbox/") or src_l == "state"):
            continue
        pv = (meta.get("text_preview") or "").strip()
        if len(pv) < MIN_PREVIEW_LEN or is_noise_text(pv):
            continue
        _recent_objs.append(obj)
        _recent_meta.append(meta)
        _recent_vecs.append(obj.get("embedding") or [])
        if len(_recent_objs) >= k_eff * 2:  # cap to a small multiple of k
            break
        await ts.maybe_yield()
    if _recent_vecs:
        # Offload cosine batch to threads to avoid blocking the event loop
        sims = await asyncio.to_thread(score_cosine_batch, qv, _recent_vecs)
        for obj, meta, sim in zip(_recent_objs, _recent_meta, sims):
            if sim < thr:
                continue
            ts_meta = float(meta.get("ts") or 0.0)
            age = max(0.0, now - ts_meta)
            rec = 0.0 if RECENCY_WINDOW_SEC <= 0 else max(0.0, 1.0 - (age / RECENCY_WINDOW_SEC))
            # Term overlap boost and session affinity
            terms = meta.get("terms") or []
            overlap = _term_overlap(q_terms, terms) if terms else 0.0
            sess_boost = 1.0
            if sess and (meta.get("session") or None) == sess:
                sess_boost = 1.08
            score = (0.7 * sim + 0.2 * rec + 0.1 * overlap) * sess_boost
            if (meta.get("source") or "").strip().lower() == "state" and short_q:
                score *= state_boost * (1.0 + state_rec_mult * rec)
            scored.append((score, meta.get("source", "recent"), obj))
            if len(scored) >= k_eff:
                break

    if len(scored) >= k_eff:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k_eff]
    eff_budget = None if EXHAUSTIVE else max_time_ms

    # 2) Persisted items: filter by source and preview, prefer current session, cap total, then ANN overlay
    items_all = await hot_task
    # Cap total items to bound runtime; lower under throttle
    cap_base = 4000
    cap = min(cap_base, 1500) if jx_state.throttle_event.is_set() else cap_base
    items_grp: List[Tuple[str, Dict[str, Any]]] = []
    items_oth: List[Tuple[str, Dict[str, Any]]] = []
    for ix, (src, obj) in enumerate(items_all or []):
        meta = (obj or {}).get("meta", {})
        src_l = (src or "").strip().lower()
        meta_src_l = (meta.get("source") or "").strip().lower()
        allow_src = (
            src_l == "dialogue" or src_l.startswith("sandbox/") or src_l == "state" or
            meta_src_l == "dialogue" or meta_src_l.startswith("sandbox/") or meta_src_l == "state"
        )
        if not allow_src:
            continue
        pv = (meta.get("text_preview") or "").strip()
        if len(pv) < MIN_PREVIEW_LEN or is_noise_text(pv):
            continue
        if sess and (meta.get("session") or None) == sess:
            items_grp.append((src, obj))
        else:
            items_oth.append((src, obj))
        if (ix % 256) == 255:
            # Periodic yield during accumulation to keep prompt responsive
            await asyncio.sleep(0)
        else:
            await ts.maybe_yield()
    # Prioritize session-matching items and cap
    items: List[Tuple[str, Dict[str, Any]]] = (items_grp + items_oth)[:cap]

    # ANN candidate generation
    overfetch = k_eff * 6
    # Reduce breadth under throttle
    if jx_state.throttle_event.is_set():
        overfetch = max(k_eff * 2, min(overfetch, k_eff * 3))
    try:
        def _rank_candidates() -> List[Tuple[int, float]]:
            return _search_ann_runtime(qv, items, top_n=min(len(items), max(k_eff, overfetch)))
        scored_candidates = await asyncio.to_thread(_rank_candidates)
    except Exception:
        scored_candidates = []  # type: ignore[name-defined]

    if scored_candidates:
        for idx_iter, (idx_c, sim) in enumerate(scored_candidates):
            if idx_c < 0 or idx_c >= len(items):
                continue
            src_i, obj_i = items[idx_c]
            if float(sim or 0.0) < thr:
                continue
            ts_meta = float(obj_i.get("meta", {}).get("ts") or 0.0)
            age = max(0.0, now - ts_meta)
            rec = 0.0 if RECENCY_WINDOW_SEC <= 0 else max(0.0, 1.0 - (age / RECENCY_WINDOW_SEC))
            # Term overlap and session boost
            meta_i = obj_i.get("meta", {})
            overlap = _term_overlap(q_terms, meta_i.get("terms") or [])
            sess_boost = 1.0
            if sess and (meta_i.get("session") or None) == sess:
                sess_boost = 1.08
            score = (0.72 * sim + 0.18 * rec + 0.10 * overlap) * sess_boost
            if (obj_i.get("meta", {}).get("source") or "").strip().lower() == "state" and short_q:
                score *= state_boost * (1.0 + state_rec_mult * rec)
            scored.append((score, obj_i.get("meta", {}).get("source", "persisted"), obj_i))
            if len(scored) >= k_eff:
                break
            if eff_budget is not None and (time.perf_counter() - t0) * 1000.0 > eff_budget:
                break
            if (idx_iter % 64) == 63:
                # Cooperative yield to keep prompt responsive during long ANN loops
                await asyncio.sleep(0)
            else:
                await ts.maybe_yield()
    else:
        # Batch cosine fallback
        B_base = 256
        # Adaptive batch size based on throttle
        B = min(B_base, 256) if jx_state.throttle_event.is_set() else B_base
        # Adaptive threshold raising once we have k candidates
        thr_curr = float(thr)
        buf_vecs: List[List[float]] = []
        buf_meta_src: List[Tuple[str, Dict[str, Any]]] = []
        async def _flush_async() -> None:
            nonlocal scored, buf_vecs, buf_meta_src
            if not buf_vecs:
                return
            # Offload cosine batch to a thread to keep UI responsive
            sims = await asyncio.to_thread(score_cosine_batch, qv, buf_vecs)
            for (src_i, obj_i), sim in zip(buf_meta_src, sims):
                if sim < thr_curr:
                    continue
                ts_meta = float(obj_i.get("meta", {}).get("ts") or 0.0)
                age = max(0.0, now - ts_meta)
                rec = 0.0 if RECENCY_WINDOW_SEC <= 0 else max(0.0, 1.0 - (age / RECENCY_WINDOW_SEC))
                score = 0.8 * sim + 0.2 * rec
                if (obj_i.get("meta", {}).get("source") or "").strip().lower() == "state" and short_q:
                    score *= state_boost * (1.0 + state_rec_mult * rec)
                scored.append((score, obj_i.get("meta", {}).get("source", "persisted"), obj_i))
            # If we have enough candidates, raise dynamic threshold to prune work
            if len(scored) >= k_eff:
                try:
                    import heapq as _hq
                    tops = _hq.nlargest(k_eff, [sc for sc, _src, _obj in scored])
                    min_in_topk = tops[-1] if tops else thr_curr
                    thr_curr = max(thr_curr, float(min_in_topk) * 0.9)
                except Exception:
                    pass
            buf_vecs = []
            buf_meta_src = []
        for idx, (src, obj) in enumerate(items):
            buf_vecs.append(obj.get("embedding") or [])
            buf_meta_src.append((src, obj))
            if len(buf_vecs) >= B:
                await _flush_async()
                # Yield frequently to keep prompt responsive
                await asyncio.sleep(0)
                if len(scored) >= k_eff:
                    break
                if eff_budget is not None and (time.perf_counter() - t0) * 1000.0 > eff_budget:
                    break
            elif (idx % 64) == 63:
                # Periodic yield even before batch flush to prevent UI stalls on large sets
                await asyncio.sleep(0)
            else:
                await ts.maybe_yield()
        if buf_vecs and len(scored) < k_eff and (eff_budget is None or (time.perf_counter() - t0) * 1000.0 <= eff_budget):
            await _flush_async()

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k_eff]


async def build_context_for(query: str, *, k: int | None = None, max_chars: int = 1500, max_time_ms: int | None = 220) -> str:
    """Build a compact context from top-k snippets (ordered by timestamp)."""
    # Fast-path: return prefetched base context when present
    try:
        if _pref_get_base is not None:
            pref = _pref_get_base(query)
            if pref:
                return pref
    except Exception:
        pass
    k = k or DEFAULT_TOP_K
    hits = await retrieve_top_k(query, k=k, max_time_ms=max_time_ms)
    # Opportunistic expansion with seeds/predictions (small extra budget)
    try:
        seeds = (_get_seeds(top_n=10) if _get_seeds is not None else [])  # type: ignore[misc]
    except Exception:
        seeds = []
    exp_hits: list[tuple[float, str, dict]] = []
    if seeds:
        try:
            exp_q = (query + " " + " ".join(seeds)).strip()
            if exp_q != (query or ""):
                extra_ms = int((max_time_ms or 220) * 0.6)
                exp_hits = await retrieve_top_k(exp_q, k=max(1, k // 2), max_time_ms=extra_ms)
        except Exception:
            exp_hits = []
    # Merge with dedupe by preview text
    if exp_hits:
        try:
            def _pv(h):
                try:
                    return (h[2].get("meta", {}).get("text_preview") or "").strip()
                except Exception:
                    return ""
            seen_pv = { _pv(h) for h in hits if _pv(h) }
            for h in exp_hits:
                pv = _pv(h)
                if pv and pv not in seen_pv:
                    hits.append(h)
                    seen_pv.add(pv)
        except Exception:
            pass
    if not hits:
        return ""
    seen: set[str] = set()
    seen_hash: set[str] = set()
    q_hash = hashlib.sha256((query or "").strip().encode("utf-8", errors="ignore")).hexdigest() if query else ""
    parts: List[str] = []
    # Optional grouping and labels
    show_labels = True
    group_on = True
    last_key = None
    total_chars = 0
    for score, src, obj in sorted(hits, key=lambda h: float((h[2].get("meta", {}).get("ts") or 0.0))):
        meta = obj.get("meta", {})
        pv = (meta.get("text_preview") or "").strip()
        csha = (meta.get("content_sha256") or "").strip()
        if not pv or pv in seen or is_noise_text(pv) or (q_hash and csha and csha == q_hash):
            continue
        # Determine grouping key: prefer file_rel when present, else source
        file_rel = str(meta.get("file_rel") or "").strip()
        key = file_rel if file_rel else f"source:{(meta.get('source') or src or 'unknown')}"
        # Add label when key changes (and grouping enabled)
        if group_on and show_labels and key != last_key:
            label = f"# file: {key}" if file_rel else f"# {key}"
            if total_chars + len(label) <= max_chars:
                parts.append(label)
                total_chars += len(label) + 1
            last_key = key
        # Add preview line
        if total_chars + len(pv) > max_chars:
            break
        seen.add(pv)
        if csha:
            if csha in seen_hash:
                continue
            seen_hash.add(csha)
        parts.append(pv)
        total_chars += len(pv) + 1
    if not parts:
        return ""
    body = "\n".join(parts)
    return f"<embeddings_context>\n{body}\n</embeddings_context>"
