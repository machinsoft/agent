from __future__ import annotations

import asyncio
import os
import contextlib
import re as _re
import time
from collections import deque, defaultdict
from typing import Sequence
from functools import lru_cache
from jinx.conversation.orchestrator import shatter
from jinx.spinner_service import sigil_spin
import jinx.state as jx_state
from jinx.micro.runtime.task_ctx import current_group
from jinx.micro.rt.backpressure import clear_throttle_if_ttl
from jinx.micro.runtime.plugins import publish_event as _publish_event
from jinx.micro.runtime.msg_id import (
    split_tags as _split_tags,
    ensure_message_id as _ensure_id,
    child_id as _child_id,
    wrap_with_tags as _wrap_tags,
)
from jinx.micro.runtime.dedup_registry import DedupRegistry as _DedupReg
try:
    from jinx.rt.wcet import update as _wcet_update, estimate_deadline_ms as _wcet_est
except Exception:
    _wcet_update = None  # type: ignore
    def _wcet_est(_op: str, base_ms: int) -> int:  # type: ignore
        return int(base_ms)


@lru_cache(maxsize=2048)
def _has_discourse_markers(text: str) -> bool:
    """Detect discourse markers using structure analysis with caching.
    
    Instead of hardcoded keyword lists, uses linguistic patterns:
    - Conjunctions and connectors at typical positions
    - Clause structure indicators (commas, semicolons)
    - Sentence complexity indicators
    """
    if not text or len(text) < 10:
        return False
    
    text_lower = text.lower()
    
    # Structural indicators of discourse (not single commands)
    # Look for:
    # 1. Multiple commas (clause structure)
    if text.count(',') >= 2:
        return True
    
    # 2. Semicolons (complex sentences)
    if ';' in text and text.count(';') < 3:  # But not just semicolon-separated list
        return True
    
    # 3. Common connectors at word boundaries (not just substring match)
    # Use regex for word boundary matching to avoid false positives
    connector_pattern = r'\b(however|therefore|because|although|moreover|furthermore|nevertheless|consequently|thus|hence|whereas)\b'
    if _re.search(connector_pattern, text_lower):
        return True
    
    # 4. Question + explanation pattern
    if '?' in text and text.rfind('?') < len(text) * 0.7:  # Question not at end
        return True
    
    return False


# Advanced multi-message splitting with semantic awareness
def _split_if_multi(msg: str, group_id: str) -> Sequence[str] | None:
    """Split a message into multiple sub-requests if it contains multiple distinct tasks.
    
    Uses advanced pattern recognition to identify:
    - Numbered lists (1., 2., 3. or 1), 2), 3))
    - Bullet points (-, *, •)
    - Semicolon-separated commands
    - Newline-separated short instructions
    
    Returns None if message should not be split, otherwise returns list of sub-messages.
    """
    try:
        _MAX_SPLIT = max(2, int(os.getenv("JINX_MULTI_SPLIT_MAX", "6")))
    except Exception:
        _MAX_SPLIT = 6
        
    try:
        _SPLIT_ON = str(os.getenv("JINX_MULTI_SPLIT_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        _SPLIT_ON = True
        
    if not _SPLIT_ON or not msg or len(msg.strip()) < 10:
        return None
        
    stripped = msg.strip()
    
    # Pattern 1: Numbered list with periods (1. Task one 2. Task two)
    numbered_period = _re.findall(r'(?:^|\n)\s*\d+\.\s+([^\n]+)', stripped)
    if len(numbered_period) >= 2 and len(numbered_period) <= _MAX_SPLIT:
        return [s.strip() for s in numbered_period if s.strip()]
    
    # Pattern 2: Numbered list with parentheses (1) Task one 2) Task two)
    numbered_paren = _re.findall(r'(?:^|\n)\s*\d+\)\s+([^\n]+)', stripped)
    if len(numbered_paren) >= 2 and len(numbered_paren) <= _MAX_SPLIT:
        return [s.strip() for s in numbered_paren if s.strip()]
    
    # Pattern 3: Bullet points (- item, * item, • item)
    bullets = _re.findall(r'(?:^|\n)\s*[-*•]\s+([^\n]+)', stripped)
    if len(bullets) >= 2 and len(bullets) <= _MAX_SPLIT:
        return [s.strip() for s in bullets if s.strip()]
    
    # Pattern 4: Semicolon-separated commands (short ones only)
    if ';' in stripped and stripped.count(';') <= _MAX_SPLIT:
        parts = [p.strip() for p in stripped.split(';') if p.strip()]
        # Only split if all parts are relatively short (likely commands)
        if 2 <= len(parts) <= _MAX_SPLIT and all(len(p) < 150 for p in parts):
            return parts
    
    # Pattern 5: Newline-separated short instructions
    lines = [ln.strip() for ln in stripped.split('\n') if ln.strip()]
    if 2 <= len(lines) <= _MAX_SPLIT:
        # Check if they look like short commands (all under 100 chars)
        # Exclude lines with discourse markers (complex sentences)
        if all(len(ln) < 100 and not _has_discourse_markers(ln) for ln in lines):
            return lines
    
    return None


# Fallback splitter without upper bound. Use simpler rules and accept large batches,
# higher-level capacity guards will bound pending size.
def _split_any_unbounded(msg: str) -> list[str] | None:
    if not msg:
        return None
    s = msg.strip()
    if not s:
        return None
    # Numbered list with period
    parts = _re.findall(r'(?:^|\n)\s*\d+\.\s+([^\n]+)', s)
    if len(parts) >= 2:
        return [p.strip() for p in parts if p.strip()]
    # Numbered list with paren
    parts = _re.findall(r'(?:^|\n)\s*\d+\)\s+([^\n]+)', s)
    if len(parts) >= 2:
        return [p.strip() for p in parts if p.strip()]
    # Bullets
    parts = _re.findall(r'(?:^|\n)\s*[-*•]\s+([^\n]+)', s)
    if len(parts) >= 2:
        return [p.strip() for p in parts if p.strip()]
    # Semicolons (short-ish)
    if ';' in s:
        parts = [p.strip() for p in s.split(';') if p.strip()]
        if len(parts) >= 2:
            return parts
    # Newlines
    parts = [ln.strip() for ln in s.split('\n') if ln.strip()]
    if len(parts) >= 2:
        return parts
    return None


async def frame_shift(q: asyncio.Queue[str]) -> None:
    """Process queue items with bounded concurrency and a single spinner.

    - Schedules up to JINX_FRAME_MAX_CONC conversation steps concurrently.
    - Keeps a single spinner active while there is any work in progress.
    - Preserves cooperative yields and respects shutdown/throttle signals.
    """
    # Concurrency limit for parallel shatter() tasks
    try:
        _MAX_CONC = max(1, int(os.getenv("JINX_FRAME_MAX_CONC", "2")))
    except Exception:
        _MAX_CONC = 2
    # Cap per-group pending queue to avoid unbounded growth
    try:
        _GMAX = max(1, int(os.getenv("JINX_GROUP_PENDING_MAX", "200")))
    except Exception:
        _GMAX = 200
    # Per-group concurrency cap (how many tasks from the same group can run concurrently)
    try:
        _GCONC = max(1, int(os.getenv("JINX_GROUP_MAX_CONC", "2")))
    except Exception:
        _GCONC = 2
    # Max items to drain from inbound queue per loop iteration
    try:
        _DRAIN_MAX = max(1, int(os.getenv("JINX_FRAME_DRAIN_MAX", "16")))
    except Exception:
        _DRAIN_MAX = 16
    # How many pending items per group to peek for simple-locator preference (0 = only head)
    try:
        _SIMPLE_SCAN = max(0, int(os.getenv("JINX_SIMPLE_LOCATOR_SCAN", "0")))
    except Exception:
        _SIMPLE_SCAN = 0
    # Threshold for embeddings-based locator classifier margin (pos - neg)
    # Now adaptive via threshold learner
    _LOC_THRESH = 0.06  # fallback only
    # Multi-split of a single inbound message into sub-requests
    try:
        _SPLIT_ON = str(os.getenv("JINX_MULTI_SPLIT_ENABLE", "1")).lower() not in ("", "0", "false", "off", "no")
    except Exception:
        _SPLIT_ON = True
    try:
        _SPLIT_MAX = max(2, int(os.getenv("JINX_MULTI_SPLIT_MAX", "6")))
    except Exception:
        _SPLIT_MAX = 6

    active: set[asyncio.Task] = set()
    task_group: dict[asyncio.Task, str] = {}
    pending_by_group: dict[str, deque[str]] = defaultdict(deque)
    group_active_count: dict[str, int] = {}
    groups_rr: list[str] = []
    rr_idx: int = 0

    def _group_of(msg: str) -> str:
        m = _re.match(r"\s*\[#group:([A-Za-z0-9_\-:.]{1,64})\]\s*(.*)", msg or "")
        if m:
            return (m.group(1) or "main").strip().lower() or "main"
        try:
            return (os.getenv("JINX_SESSION", "") or "main").strip() or "main"
        except Exception:
            return "main"

    def _strip_group_tag(msg: str) -> str:
        # Now strips both group/id tags via msg_id utilities
        _mid, _grp, _body = _split_tags(msg)
        return _body
    spin_evt: asyncio.Event | None = None
    spin_task: asyncio.Task | None = None
    spin_start_t: float | None = None

    async def _ensure_spinner() -> None:
        nonlocal spin_evt, spin_task, spin_start_t
        try:
            _spin_on = str(os.getenv("JINX_SPINNER_ENABLE", "1")).strip().lower() not in ("", "0", "false", "off", "no")
        except Exception:
            _spin_on = True
        if not _spin_on:
            return
        if spin_task is None or spin_task.done():
            spin_evt = asyncio.Event()
            spin_task = asyncio.create_task(sigil_spin(spin_evt))
            # Publish spinner start
            try:
                spin_start_t = time.perf_counter()
                _publish_event("spinner.start", {"t": spin_start_t})
            except Exception:
                pass

    async def _stop_spinner() -> None:
        nonlocal spin_evt, spin_task, spin_start_t
        if spin_evt is not None:
            spin_evt.set()
        if spin_task is not None and not spin_task.done():
            try:
                await asyncio.wait_for(spin_task, timeout=2.0)
            except asyncio.TimeoutError:
                spin_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await spin_task
        spin_evt = None
        spin_task = None
        # Publish spinner stop with duration
        try:
            t1 = time.perf_counter()
            dt = (t1 - spin_start_t) if spin_start_t else 0.0
            _publish_event("spinner.stop", {"dt": dt})
        except Exception:
            pass
        spin_start_t = None
        # Ensure throttle is cleared when spinner stops to allow intake to resume
        try:
            clear_throttle_if_ttl()
        except Exception:
            pass
        try:
            if jx_state.throttle_event.is_set():
                jx_state.throttle_event.clear()
                setattr(jx_state, "throttle_unset_ts", 0.0)
        except Exception:
            pass

    # Optional per-step hard RT budget (in milliseconds); 0 disables.
    try:
        step_ms = int(os.getenv("JINX_FRAME_STEP_RT_MS", "0"))
    except Exception:
        step_ms = 0

    # Append helper with capacity guard and event publishing
    def _pend_add(gid: str, text: str) -> bool:
        dq = pending_by_group[gid]
        try:
            cap = dq.maxlen
            if cap is not None and len(dq) >= cap:
                try:
                    _publish_event("queue.drop", {"group": gid, "text": text})
                except Exception:
                    pass
                return False
        except Exception:
            pass
        dq.append(text)
        return True

    async def _run_one(s: str, gid: str, _msg_id: str | None) -> None:
        try:
            tok = current_group.set(gid)
            t0 = time.perf_counter()
            # Dynamic per-request adaptation (autonomous gating/tuning)
            try:
                from jinx.micro.runtime.dynamic_config_plugin import adapt_config_for_request as _adapt_cfg
                import asyncio as _aio
                await _aio.wait_for(_adapt_cfg(s, context={"group": gid}), timeout=0.2)
            except Exception:
                pass
            if step_ms and step_ms > 0:
                conv_task = asyncio.create_task(shatter(s))
                try:
                    await asyncio.wait_for(conv_task, timeout=step_ms / 1000.0)
                except asyncio.TimeoutError:
                    conv_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await conv_task
            else:
                await shatter(s)
            try:
                dt = time.perf_counter() - t0
                _publish_event("turn.metrics", {"group": gid, "dt": dt})
                if _wcet_update is not None:
                    _wcet_update("turn", dt * 1000.0)
            except Exception:
                pass
        except Exception as e:
            # Publish error event for self-repair systems; keep scheduler alive
            try:
                import traceback as _tb
                _publish_event("turn.error", {"group": gid, "error": str(e), "tb": _tb.format_exc()})
            except Exception:
                pass
        finally:
            try:
                current_group.reset(tok)  # type: ignore[name-defined]
            except Exception:
                pass
            # Mark dedup lifecycle finished
            try:
                # Access closure of registry if present
                if _msg_id:
                    _dedup.on_finished(_msg_id)
            except Exception:
                pass

    def _score_locator_cached(msg: str) -> float:
        try:
            from jinx.micro.conversation.locator_semantics import get_locator_score_cached as _get_sc  # type: ignore
        except Exception:
            _get_sc = None  # type: ignore
        if _get_sc is None:
            return 0.0
        try:
            sc = _get_sc(msg)
        except Exception:
            sc = None
        return float(sc) if sc is not None else 0.0

    async def _pop_next_async(dq: deque[str], prefer_simple: bool) -> str | None:
        """Async version with adaptive threshold."""
        if not dq:
            return None
        if not prefer_simple or _SIMPLE_SCAN <= 0:
            return dq.popleft()
        
        # Get adaptive threshold from learner
        thresh = _LOC_THRESH  # fallback
        try:
            from jinx.micro.brain.threshold_learner import select_threshold as _sel_th
            thresh = await _sel_th('locator_thresh')
        except Exception:
            pass
        
        # Embeddings-based preference: scan first N pending for highest cached locator score
        n = min(len(dq), _SIMPLE_SCAN + 1)
        best_idx = -1
        best_sc = thresh
        # Enumerate without consuming: convert small head slice to list
        i = 0
        for m in dq:
            if i >= n:
                break
            sc = _score_locator_cached(m)
            if sc >= best_sc:
                best_sc = sc
                best_idx = i
            i += 1
        if best_idx <= 0:
            return dq.popleft()
        # Rotate to bring best to head, pop, then rotate back
        try:
            dq.rotate(-best_idx)
            item = dq.popleft()
            dq.rotate(best_idx)
            # Record outcome asynchronously
            try:
                from jinx.micro.brain.threshold_learner import record_threshold_outcome as _rec_th
                success = True  # selected fast-lane item
                asyncio.create_task(_rec_th('locator_thresh', thresh, success))
            except Exception:
                pass
            return item
        except Exception:
            # Fallback: pop head
            return dq.popleft()

    try:
        # De-dup registry: time-bounded recent-completion memory
        try:
            _DETTL = float(os.getenv("JINX_DEDUP_TTL", "600"))
        except Exception:
            _DETTL = 600.0
        _dedup = _DedupReg(_DETTL)
        while True:
            # Respect global shutdown fast-path
            if jx_state.shutdown_event.is_set():
                try:
                    from jinx.micro.logger.debug_logger import debug_log_sync
                    debug_log_sync("Shutdown event detected - exiting loop", "FRAME_SHIFT")
                except Exception:
                    pass
                break
            # Periodically clear throttle via TTL
            try:
                clear_throttle_if_ttl()
            except Exception:
                pass
            # Soft-throttle: only gate scheduling while there is active work; do not block intake forever
            if jx_state.throttle_event.is_set() and active:
                await asyncio.sleep(0.05)
                continue

            # Drain inbound queue briefly into per-group pending buffers
            drained = 0
            while drained < _DRAIN_MAX:  # cap per loop to avoid starvation
                try:
                    item = await asyncio.wait_for(q.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    break
                # Normalize/ensure ID and parse tags
                try:
                    item = _ensure_id(item)
                except Exception:
                    pass
                mid, grp_tag, body0 = _split_tags(item)
                gid = (grp_tag or _group_of(item) or "main").strip()
                # Try-admit by ID (skip duplicates already pending/inflight/done)
                try:
                    if mid and not _dedup.try_admit(mid):
                        continue
                except Exception:
                    pass
                if gid not in pending_by_group:
                    pending_by_group[gid] = deque(maxlen=_GMAX)
                    groups_rr.append(gid)
                # Attempt multi-split (e.g., multiple short tasks in one message)
                subs = _split_if_multi(body0, gid)
                if not subs:
                    subs = _split_any_unbounded(body0)
                if subs:
                    # Assign child IDs derived from parent id
                    _ad, _drop = 0, 0
                    for idx, it in enumerate(subs):
                        try:
                            cid = _child_id(mid, idx)
                            # Enforce dedup for child IDs as well
                            try:
                                if cid and not _dedup.try_admit(cid):
                                    _drop += 1
                                    continue
                            except Exception:
                                pass
                            submsg = _wrap_tags(gid, cid, it)
                        except Exception:
                            submsg = it
                            cid = None
                        if _pend_add(gid, submsg):
                            _ad += 1
                        else:
                            _drop += 1
                        try:
                            _publish_event("queue.intake", {"group": gid, "text": it})
                        except Exception:
                            pass
                    # Batch summary
                    try:
                        from jinx.log_paths import BLUE_WHISPERS as _BW
                        from jinx.logger.file_logger import append_line as _append
                        await _append(_BW, f"[intake] group={gid} parent={mid or '-'} subs={len(subs)} admitted={_ad} dropped={_drop}")
                    except Exception:
                        pass
                else:
                    _ad = 1 if _pend_add(gid, item) else 0
                    _drop = 0 if _ad == 1 else 1
                    if not subs:
                        try:
                            _publish_event("queue.intake", {"group": gid, "text": item})
                        except Exception:
                            pass
                    # Single-item summary
                    try:
                        from jinx.log_paths import BLUE_WHISPERS as _BW
                        from jinx.logger.file_logger import append_line as _append
                        await _append(_BW, f"[intake] group={gid} parent={mid or '-'} admitted={_ad} dropped={_drop}")
                    except Exception:
                        pass
                # Count this intake iteration regardless of split
                drained += 1

            # Fill up to concurrency limit with fair round-robin across groups
            filled = False
            if groups_rr:
                n = len(groups_rr)
                start = rr_idx % max(1, n)
                # Two passes: prefer simple-locator, then general
                for prefer_simple in (True, False):
                    i = 0
                    while len(active) < _MAX_CONC and i < n:
                        gid = groups_rr[(start + i) % n]
                        i += 1
                        # respect per-group concurrency cap
                        if int(group_active_count.get(gid, 0)) >= _GCONC:
                            continue
                        dq = pending_by_group.get(gid)
                        if not dq:
                            continue
                        raw = await _pop_next_async(dq, prefer_simple)
                        if raw is None:
                            continue
                        # Extract id/group/body and mark as scheduled
                        try:
                            _mid, _grp, _body = _split_tags(raw)
                        except Exception:
                            _mid, _grp, _body = (None, None, _strip_group_tag(raw))
                        try:
                            if _mid:
                                _dedup.on_scheduled(_mid)
                        except Exception:
                            pass
                        msg = _body
                        await _ensure_spinner()
                        # Deadline-aware scheduling (EDF skeleton): prefer schedule_turn
                        try:
                            from jinx.rt.scheduler import schedule_turn as _schedule_turn  # local import to avoid cycles
                            # Use step_ms deadline if set, else fallback to env or 15000 ms
                            try:
                                _fallback_dl = int(os.getenv("JINX_FRAME_DEADLINE_MS", "15000"))
                            except Exception:
                                _fallback_dl = 15000
                            base_dl = int(step_ms) if int(step_ms or 0) > 0 else _fallback_dl
                            _dl = _wcet_est("turn", base_dl)
                            t = _schedule_turn(lambda: _run_one(msg, gid, _mid), deadline_ms=_dl, name=f"turn:{gid}")
                        except Exception:
                            t = asyncio.create_task(_run_one(msg, gid, _mid))
                        active.add(t)
                        task_group[t] = gid
                        # increment per-group active count for fairness
                        try:
                            group_active_count[gid] = int(group_active_count.get(gid, 0)) + 1
                        except Exception:
                            group_active_count[gid] = 1
                        try:
                            _publish_event("turn.scheduled", {"group": gid, "text": msg})
                        except Exception:
                            pass
                        filled = True
                        if len(active) >= _MAX_CONC:
                            break
                rr_idx = (start + i) % max(1, len(groups_rr))

            # Reap any finished tasks
            done_now = [t for t in active if t.done()]
            for t in done_now:
                active.discard(t)
                with contextlib.suppress(Exception):
                    _ = t.result()
                # release per-group slot
                try:
                    gid = task_group.pop(t, None)
                    if gid is not None:
                        cur = int(group_active_count.get(gid, 0))
                        if cur > 0:
                            group_active_count[gid] = cur - 1
                    try:
                        _publish_event("turn.finished", {"group": gid})
                    except Exception:
                        pass
                except Exception:
                    pass

            # Prune empty groups from RR when they have no pending and are not active
            if groups_rr:
                new_rr = []
                for g in groups_rr:
                    dq = pending_by_group.get(g)
                    if (dq and len(dq) > 0) or (g in group_active_count and group_active_count[g] > 0):
                        new_rr.append(g)
                    else:
                        # drop empty group from maps
                        pending_by_group.pop(g, None)
                        group_active_count.pop(g, None)
                groups_rr = new_rr
                if rr_idx >= len(groups_rr):
                    rr_idx = 0

            # If nothing to do, wait briefly or for shutdown
            if not active and not filled:
                # No in-flight work; stop spinner if running
                await _stop_spinner()
                # Wait for either new item or shutdown
                get_task = asyncio.create_task(q.get())
                shut_task = asyncio.create_task(jx_state.shutdown_event.wait())
                done, _ = await asyncio.wait({get_task, shut_task}, return_when=asyncio.FIRST_COMPLETED)
                if shut_task in done:
                    if not get_task.done():
                        get_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await get_task
                    break
                # Got a new item; enqueue into group buffer and continue loop (apply same normalization/dedup)
                try:
                    item2 = get_task.result()
                except Exception:
                    continue
                # Ensure ID then parse
                try:
                    item2 = _ensure_id(item2)
                except Exception:
                    pass
                _mid2, _grp2, _body2 = _split_tags(item2)
                gid2 = (_grp2 or _group_of(item2) or "main").strip()
                try:
                    if _mid2 and not _dedup.try_admit(_mid2):
                        continue
                except Exception:
                    pass
                if gid2 not in pending_by_group:
                    pending_by_group[gid2] = deque(maxlen=_GMAX)
                    groups_rr.append(gid2)
                # Attempt multi-split for idle-path as well
                _subs2 = _split_if_multi(_body2, gid2)
                if not _subs2:
                    _subs2 = _split_any_unbounded(_body2)
                if _subs2:
                    _ad2, _drop2 = 0, 0
                    for idx, it in enumerate(_subs2):
                        try:
                            _cid2 = _child_id(_mid2, idx)
                            try:
                                if _cid2 and not _dedup.try_admit(_cid2):
                                    _drop2 += 1
                                    continue
                            except Exception:
                                pass
                            _sub2 = _wrap_tags(gid2, _cid2, it)
                        except Exception:
                            _sub2 = it
                        if _pend_add(gid2, _sub2):
                            _ad2 += 1
                        else:
                            _drop2 += 1
                else:
                    _ad2 = 1 if _pend_add(gid2, item2) else 0
                    _drop2 = 0 if _ad2 == 1 else 1
                # Idle-branch summary
                try:
                    from jinx.log_paths import BLUE_WHISPERS as _BW
                    from jinx.logger.file_logger import append_line as _append
                    await _append(_BW, f"[intake.idle] group={gid2} parent={_mid2 or '-'} subs={len(_subs2) if _subs2 else 0} admitted={_ad2} dropped={_drop2}")
                except Exception:
                    pass
            else:
                # Give control back to event loop to keep RT responsiveness
                await asyncio.sleep(0)
    finally:
        # Stop spinner and cancel any remaining tasks on exit
        with contextlib.suppress(Exception):
            await _stop_spinner()
        for t in list(active):
            if not t.done():
                t.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*active, return_exceptions=True)
