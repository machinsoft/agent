from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple, List
from threading import Lock as _TLock

from jinx.net import get_openai_client
from jinx.micro.parser.api import parse_tagged_blocks as _parse_blocks
import re as _re

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
    def _ab_int(name: str, ctx=None) -> int:
        return {"llm_max_conc": 4, "llm_timeout_ms": 20000}.get(name, 4)
    def _ab_get(name: str, ctx=None) -> float:
        return 4.0
    def _ab_record(name: str, success: bool, latency_ms: float = 0, ctx=None) -> None:
        pass

# TTL cache + request coalescing + concurrency limiting + timeouts for LLM Responses API
# Keyed by a stable fingerprint of (instructions, model, input_text, extra_kwargs)

_TTL_SEC = 300.0
_TIMEOUT_MS = 20000
_MAX_CONC = 4

# Family rate limiting (requests per window per normalized family key)
_FAM_RATE = 3
_FAM_WIN = 5.0
_FAM_DROP = True

_FAMILY_RATE_LIMITER: Dict[str, List[float]] = {}

_DUMP = False

_FPR_NORMALIZE = True
_FAM_CACHE_ON = True
_HARD_TIMEOUT_MS = 30000
_MULTI_SAMPLES = 1
_MULTI_HEDGE_MS = 0
_MULTI_CANCEL_LOSERS = True

_mem: Dict[str, Tuple[float, str]] = {}
_inflight: Dict[str, asyncio.Future] = {}
_family_inflight: Dict[str, asyncio.Future] = {}
_family_mem: Dict[str, Tuple[float, str]] = {}
_family_recent: Dict[str, List[float]] = {}
_inflight_tlock: _TLock = _TLock()
_sem = asyncio.Semaphore(max(1, _MAX_CONC))


def _now() -> float:
    try:
        return time.time()
    except Exception:
        return 0.0


def _ensure_non_empty(s: str) -> str:
    t = str(s or "")
    if t.strip():
        return t
    return "<llm_empty>\nOpenAI returned an empty response.\n</llm_empty>"


def _safe_jsonable(obj: Any, depth: int = 0) -> Any:
    """Best-effort transform to jsonable structure without exploding on exotic types.

    Limits depth to avoid huge payloads; falls back to repr for unknowns.
    """
    if depth > 4:
        return f"<{type(obj).__name__}:depth>"
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v, depth + 1) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [_safe_jsonable(v, depth + 1) for v in obj[:100]]  # cap length for stability
    try:
        return json.loads(json.dumps(obj))  # type: ignore[arg-type]
    except Exception:
        try:
            r = repr(obj)
            # trim very long reprs to keep key stable and small
            if len(r) > 256:
                r = r[:256] + "..."
            return r
        except Exception:
            return f"<{type(obj).__name__}>"


def _normalize_for_fingerprint(s: str) -> str:
    """Lightweight normalization to stabilize cache keys and coalesce duplicates.

    - Normalize newlines, collapse 3+ blanks to 2
    - Drop repeated paragraphs (>= 200 chars) keeping first occurrence
    - Trim to a conservative budget to cap key size
    """
    if not _FPR_NORMALIZE:
        return s or ""
    t = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    t = _re.sub(r"\n{3,}", "\n\n", t)
    parts = t.split("\n\n")
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        key = p.strip()
        if len(key) >= 200:
            if key in seen:
                continue
            seen.add(key)
        out.append(p)
    t = "\n\n".join(out)
    # Clip to 16k chars to keep key reasonable
    return t[:16000]


def _fingerprint(instructions: str, model: str, input_text: str, extra_kwargs: Dict[str, Any]) -> str:
    payload = {
        "i": _normalize_for_fingerprint(instructions or ""),
        "m": (model or ""),
        "t": _normalize_for_fingerprint(input_text or ""),
        "k": _safe_jsonable(extra_kwargs or {}),
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _fingerprint_family(instructions: str, model: str, input_text: str) -> str:
    """Family fingerprint ignoring extra kwargs.

    Used to coalesce outward calls for the same logical request shape regardless of
    small variations (like temperature) to guarantee single outbound call.
    """
    return _fingerprint(instructions, model, input_text, {})


async def _dump_line(line: str) -> None:
    if not _DUMP:
        return
    try:
        from jinx.logger.file_logger import append_line as _append
        from jinx.log_paths import BLUE_WHISPERS
        await _append(BLUE_WHISPERS, f"[llm_cache] {line}")
    except Exception:
        pass


def _is_forbidden_region(ex: BaseException) -> bool:
    """Detect provider 'unsupported country/region/territory' 403 errors robustly.

    We rely on best-effort string and attribute checks because SDK types may vary.
    """
    try:
        # Common SDKs expose http_status / status_code
        sc = getattr(ex, "status_code", None) or getattr(ex, "http_status", None) or getattr(ex, "status", None)
        if sc == 403:
            s = str(ex).lower()
            if "unsupported_country_region_territory" in s:
                return True
            if "request_forbidden" in s and "country" in s and "supported" in s:
                return True
    except Exception:
        pass
    # Fallback: message-only detection
    s = str(ex).lower()
    if "unsupported_country_region_territory" in s:
        return True
    return False


def _is_rate_limited(ex: BaseException) -> bool:
    """Detect provider 429 rate limit errors."""
    try:
        sc = getattr(ex, "status_code", None) or getattr(ex, "http_status", None) or getattr(ex, "status", None)
        if sc == 429:
            return True
    except Exception:
        pass
    s = str(ex).lower()
    if "rate" in s and "limit" in s:
        return True
    return False


def _family_allow(fam_key: str) -> bool:
    """Return True if a call to this family key is allowed under the rate window.

    If window exceeded, returns False (caller should coalesce/wait or serve cached).
    """
    if _FAM_RATE <= 0 or _FAM_WIN <= 0:
        return True
    now = _now()
    q = _family_recent.get(fam_key)
    if q is None:
        q = []
        _family_recent[fam_key] = q
    # prune
    i = 0
    for t in q:
        if now - t <= _FAM_WIN:
            break
        i += 1
    if i:
        del q[:i]
    if len(q) >= _FAM_RATE:
        return False
    q.append(now)
    return True


async def call_openai_cached(instructions: str, model: str, input_text: str, *, extra_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Cached/coalesced wrapper for OpenAI Responses API.

    Returns output_text (string). On API error, raises the exception (caller logs/handles).
    """
    ek = extra_kwargs or {}
    # Strip internal control keys from fingerprinting and outbound SDK kwargs
    ek_fpr = {str(k): v for k, v in ek.items() if not str(k).startswith("__")}
    key = _fingerprint(instructions, model, input_text, ek_fpr)
    fam_key = _fingerprint_family(instructions, model, input_text)
    no_family = bool(ek.get("__no_family__", False))
    # TTL cache lookup (exact)
    item = _mem.get(key)
    if item is not None:
        exp, val = item
        if exp >= _now():
            return _ensure_non_empty(val)
        else:
            _mem.pop(key, None)
    # Family-level TTL cache (optional): serve recent result regardless of extra kwargs
    fam_cache_on = _FAM_CACHE_ON
    if fam_cache_on:
        fitem = _family_mem.get(fam_key)
        if fitem is not None:
            fexp, fval = fitem
            if fexp >= _now():
                return _ensure_non_empty(fval)
            else:
                _family_mem.pop(fam_key, None)

    # Family rate limiting guard (before creating new inflight)
    if not no_family and not _family_allow(fam_key):
        await _dump_line("family_rate_limited")
        # If there's an inflight for this family, await it to avoid extra calls
        with _inflight_tlock:
            existing_fam = _family_inflight.get(fam_key)
        if existing_fam is not None:
            try:
                res = await existing_fam
                return _ensure_non_empty(str(res or ""))
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        # Serve recent family cache if any
        if fam_cache_on:
            fitem = _family_mem.get(fam_key)
            if fitem is not None and fitem[0] >= _now():
                return _ensure_non_empty(fitem[1])
        # Avoid silent drops: wait briefly for the window to open; fall back to a visible marker.
        try:
            q = _family_recent.get(fam_key) or []
            wait_s = (max(0.0, float(q[0] + _FAM_WIN - _now())) if q else float(_FAM_WIN))
            wait_s = min(wait_s, float(_FAM_WIN), 10.0)
        except Exception:
            wait_s = min(float(_FAM_WIN), 1.0)
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        if not _family_allow(fam_key):
            return "<llm_rate_limited>\nLocal rate limiter engaged; please retry in a moment.\n</llm_rate_limited>"

    # Coalescing (with race-free creation)
    loop = asyncio.get_running_loop()
    to_wait: asyncio.Future | None = None
    fut: asyncio.Future
    # Cross-thread safe critical section for inflight maps
    with _inflight_tlock:
        # Exact-key inflight first
        existing_exact = _inflight.get(key)
        if existing_exact is not None:
            to_wait = existing_exact
        else:
            # Family-level inflight (unless disabled)
            if not no_family:
                existing_fam = _family_inflight.get(fam_key)
                if existing_fam is not None:
                    to_wait = existing_fam
            if to_wait is None:
                fut = loop.create_future()
                _inflight[key] = fut
                if not no_family:
                    _family_inflight[fam_key] = fut
    if to_wait is not None:
        try:
            res = await to_wait
            return _ensure_non_empty(str(res or ""))
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            if _is_forbidden_region(ex):
                return "<llm_forbidden_region>\nOpenAI is not available in this region.\n</llm_forbidden_region>"
            if _is_rate_limited(ex):
                return "<llm_rate_limited>\nOpenAI rate limited the request; please retry shortly.\n</llm_rate_limited>"
            # If the inflight failed, continue to execute fresh with a new future
            with _inflight_tlock:
                fut = loop.create_future()
                _inflight[key] = fut
                if not no_family:
                    _family_inflight[fam_key] = fut
    soft_timeout = False
    async with _sem:
        await _dump_line(f"call key={key[:8]} model={model} ilen={len(instructions)} tlen={len(input_text)}")
        def _worker():
            client = get_openai_client()
            ek_api = {str(k): v for k, v in ek.items() if not str(k).startswith("__")}
            # Responses API requires a non-empty 'input'/'prompt'/'conversation_id'/'previous_response_id'.
            # When input_text is empty, fall back to using 'instructions' as input to satisfy the API.
            safe_input = input_text if (input_text or "").strip() else instructions
            return client.responses.create(
                instructions=instructions,
                model=model,
                input=safe_input,
                **ek_api,
            )
        # Launch background task so we can safely wait on shared fut even if a soft timeout occurs
        task: asyncio.Task = asyncio.create_task(asyncio.to_thread(_worker))

        def _on_done(t: asyncio.Task) -> None:
            try:
                if t.cancelled():
                    # Propagate cancellation to awaiters without leaking to event loop logs
                    if not fut.done():
                        fut.set_exception(asyncio.CancelledError())
                    return
                r = t.result()
                out = _ensure_non_empty(str(getattr(r, "output_text", "")))
                expiry = _now() + max(1.0, _TTL_SEC)
                _mem[key] = (expiry, out)
                if fam_cache_on:
                    _family_mem[fam_key] = (expiry, out)
                if not fut.done():
                    fut.set_result(out)
            except BaseException as ex:
                try:
                    if not fut.done():
                        fut.set_exception(ex)
                except BaseException:
                    pass
            finally:
                try:
                    _inflight.pop(key, None)
                except Exception:
                    pass
                # Clear family mapping if set
                if not no_family:
                    try:
                        if _family_inflight.get(fam_key) is fut:
                            _family_inflight.pop(fam_key, None)
                    except Exception:
                        pass

        task.add_done_callback(_on_done)
        # Implement soft timeout without cancelling the underlying task
        timeout_sec = max(0.1, _TIMEOUT_MS / 1000)
        timeout_task = asyncio.create_task(asyncio.sleep(timeout_sec))
        done, _ = await asyncio.wait({task, timeout_task}, return_when=asyncio.FIRST_COMPLETED)
        # Clean up timeout task if still pending
        if not timeout_task.done():
            timeout_task.cancel()
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass
        if task in done:
            try:
                r = await task
            except asyncio.CancelledError:
                # Should not happen since we didn't cancel; treat as transient
                soft_timeout = True
                await _dump_line("soft_timeout_cancelled")
            except BaseException as ex:
                # Convert immediate failure into soft path so we await the shared future.
                # This ensures we 'consume' the future's exception (set by the callback)
                # to avoid 'Future exception was never retrieved' warnings.
                soft_timeout = True
                if _is_forbidden_region(ex):
                    await _dump_line("provider_forbidden_region_immediate")
                else:
                    await _dump_line(f"task_failed:{type(ex).__name__}")
            else:
                out = _ensure_non_empty(str(getattr(r, "output_text", "")))
                # Callback will also set cache/fut and pop inflight; just return out here
                return out
        else:
            soft_timeout = True
            await _dump_line("soft_timeout")

    # If we timed out, release the semaphore first, then await the shared inflight future.
    if soft_timeout:
        await _dump_line("awaiting inflight outside semaphore")
        try:
            try:
                res = await asyncio.wait_for(fut, timeout=max(0.1, float(_HARD_TIMEOUT_MS) / 1000.0))
                return _ensure_non_empty(str(res or ""))
            except asyncio.TimeoutError:
                # Hard timeout: release waiters and avoid infinite hang if SDK thread is stuck.
                tout = "<llm_timeout>\nOpenAI call timed out.\n</llm_timeout>"
                try:
                    if not fut.done():
                        fut.set_result(tout)
                except Exception:
                    pass
                try:
                    # Best-effort: clear inflight maps so future calls can retry.
                    with _inflight_tlock:
                        _inflight.pop(key, None)
                        if not no_family:
                            _family_inflight.pop(fam_key, None)
                except Exception:
                    pass
                await _dump_line("hard_timeout")
                return tout
        except BaseException as ex:
            # Propagate the underlying error if the background task failed
            if _is_forbidden_region(ex) or _is_rate_limited(ex):
                msg = (
                    "<llm_forbidden_region>\nOpenAI is not available in this region.\n</llm_forbidden_region>"
                    if _is_forbidden_region(ex)
                    else "<llm_rate_limited>\nOpenAI rate limited the request; please retry shortly.\n</llm_rate_limited>"
                )
                # Negative cache for a short period to avoid rapid retries.
                try:
                    nexp = _now() + min(60.0, max(1.0, _TTL_SEC))
                    _mem[key] = (nexp, msg)
                    if fam_cache_on:
                        _family_mem[fam_key] = (nexp, msg)
                except Exception:
                    pass
                await _dump_line("provider_soft_block")
                try:
                    if not fut.done():
                        fut.set_result(msg)
                except Exception:
                    pass
                return msg
            raise ex

async def call_openai_multi_validated(
    instructions: str,
    model: str,
    input_text: str,
    *,
    code_id: str,
    base_extra_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Run multiple cached LLM calls in parallel and return the first valid output.

    - Variations are done via temperature tweaks (kept small to preserve determinism).
    - Validation: output must contain exactly one <python_{code_id}> block.
    - Does not cancel in-flight calls so they can populate the TTL cache for future turns.
    """
    n = max(1, int(_MULTI_SAMPLES))
    # Conservative small variations
    temps_all: List[float] = [0.2, 0.5, 0.8, 0.3, 0.7]
    temps = temps_all[:max(1, n)]
    extra = dict(base_extra_kwargs or {})
    hedge_ms = int(_MULTI_HEDGE_MS)
    cancel_losers = bool(_MULTI_CANCEL_LOSERS)

    async def _one(t: float, register_family: bool) -> str:
        kw = dict(extra)
        # Temperature is widely supported in Responses API kwargs
        kw["temperature"] = t
        # Only the first sample registers family inflight; others opt-out to avoid collapsing race
        if not register_family:
            kw["__no_family__"] = True
        return await call_openai_cached(instructions, model, input_text, extra_kwargs=kw)

    # Start first immediately
    tasks: List[asyncio.Task] = []
    if not temps:
        temps = [0.2]
    t0 = asyncio.create_task(_one(temps[0], True))
    tasks.append(t0)
    # Optional: start one additional hedged request after a short delay if first hasn't finished
    if len(temps) > 1 and hedge_ms > 0:
        try:
            await asyncio.wait_for(asyncio.sleep(max(0.0, hedge_ms) / 1000.0), timeout=max(0.05, hedge_ms / 1000.0))
        except Exception:
            pass
        if not t0.done():
            t1 = asyncio.create_task(_one(temps[1], False))
            tasks.append(t1)

    first: str | None = None
    for fut in asyncio.as_completed(tasks):
        try:
            out = await fut
        except Exception:
            continue
        if first is None:
            first = out  # remember earliest even if invalid, as fallback
        try:
            pairs = _parse_blocks(out, code_id)
        except Exception:
            pairs = []
        # Strict: exactly one matching code block and non-empty content
        good = 0
        for tag, core in pairs:
            if tag.strip() == f"python_{code_id}" and (core or "").strip():
                good += 1
        if good == 1:
            # Best-effort cancel losers to reduce outbound traffic
            if cancel_losers:
                for t in tasks:
                    if t is not fut and not t.done():
                        t.cancel()
                        try:
                            await t
                        except Exception:
                            pass
            return out
    # If none validated, return earliest completed output
    return first or ""
