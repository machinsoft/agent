from __future__ import annotations

import asyncio
import re as _re
from typing import Any

from jinx.logging_service import bomb_log
from jinx.micro.rag.file_search import build_file_search_tools
from jinx.net import get_openai_client
from .llm_cache import call_openai_cached, call_openai_multi_validated
from jinx.micro.text.heuristics import is_code_like as _is_code_like
import asyncio as _asyncio
import queue as _queue


_PROMPT_DEDUP_ON = True
_PROMPT_MAX_INST_CHARS = 12000
_PROMPT_MAX_INPUT_CHARS = 12000
_PROMPT_MIN_PARA_CHARS = 200
_PROMPT_MIN_LINE_CHARS = 80
_PROMPT_SHINGLE_CHARS = 80
_PROMPT_SHINGLE_STEP = 20
_PROMPT_DUP_OVERLAP = 0.9
_STREAM_INST_MAX_CHARS = 16000
_STREAM_INPUT_MAX_CHARS = 12000


def _normalize_prompt_payload(instructions: str, input_text: str) -> tuple[str, str]:
    """Deduplicate and clip prompt pieces to reduce token waste.

    - Removes duplicate large paragraphs (first occurrence kept) across instructions and input.
    - Collapses excessive blank lines and consecutive duplicate long lines.
    - Clips by env budgets. Env gates allow turning off if needed.
    """
    if not _PROMPT_DEDUP_ON:
        return instructions, input_text
    # Budgets / thresholds
    max_ins = _PROMPT_MAX_INST_CHARS
    max_inp = _PROMPT_MAX_INPUT_CHARS
    min_para = _PROMPT_MIN_PARA_CHARS
    min_line = _PROMPT_MIN_LINE_CHARS
    sh_sz = _PROMPT_SHINGLE_CHARS
    sh_step = _PROMPT_SHINGLE_STEP
    dup_overlap = _PROMPT_DUP_OVERLAP

    def _shingles(s: str) -> set[str]:
        t = s
        L = len(t)
        if L <= sh_sz:
            return {t} if t else set()
        out: set[str] = set()
        i = 0
        while i + sh_sz <= L:
            out.add(t[i : i + sh_sz])
            i += max(1, sh_step)
        return out

    def _is_near_dup(p: str, seen_sh: set[str]) -> bool:
        if not p:
            return False
        sh = _shingles(p)
        if not sh:
            return False
        inter = len([x for x in sh if x in seen_sh])
        ratio = inter / max(1, len(sh))
        return ratio >= dup_overlap

    def _norm_one(s: str, seen: set[str], seen_sh: set[str]) -> str:
        if not s:
            return ""
        t = s.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse excessive blank lines
        t = _re.sub(r"\n{3,}", "\n\n", t)
        # Deduplicate paragraphs (double-newline separated) if sufficiently large
        parts = t.split("\n\n")
        out_parts: list[str] = []
        for p in parts:
            key = p.strip()
            if len(key) >= min_para:
                # exact dup
                if key in seen:
                    continue
                # near-dup via shingle overlap
                if _is_near_dup(key, seen_sh):
                    continue
                seen.add(key)
                for x in _shingles(key):
                    seen_sh.add(x)
            out_parts.append(p)
        t = "\n\n".join(out_parts)
        # Remove consecutive duplicate long lines
        lines = t.splitlines()
        out_lines: list[str] = []
        prev = None
        for ln in lines:
            if prev is not None and ln == prev and len(ln) >= min_line:
                # skip duplicate
                continue
            out_lines.append(ln)
            prev = ln
        return "\n".join(out_lines)

    seen: set[str] = set()
    seen_sh: set[str] = set()
    ins = _norm_one(instructions or "", seen, seen_sh)
    inp = _norm_one(input_text or "", seen, seen_sh)
    # Clip to budgets
    ins = ins[:max_ins]
    inp = inp[:max_inp]
    return ins, inp


async def call_openai(instructions: str, model: str, input_text: str) -> str:
    """Call OpenAI Responses API and return output text.

    Uses to_thread to run the sync SDK call and relies on the shared retry helper
    at the caller site to provide resiliency.
    """
    # Normalize to prevent duplicated large chunks from reaching the API
    instructions, input_text = _normalize_prompt_payload(instructions, input_text)
    extra_kwargs: dict[str, Any]
    if not _is_code_like(input_text or ""):
        extra_kwargs = {}
    else:
        try:
            extra_kwargs = build_file_search_tools()
        except Exception:
            extra_kwargs = {}
    try:
        return await call_openai_cached(
            instructions=instructions,
            model=model,
            input_text=input_text,
            extra_kwargs=extra_kwargs,
        )
    except Exception as e:
        await bomb_log(f"ERROR cortex exploded: {e}")
        raise


async def call_openai_validated(instructions: str, model: str, input_text: str, *, code_id: str) -> str:
    """Preferred LLM path: multi-sample race with strict validation and TTL cache.

    Enabled by default via env (JINX_LLM_MULTI_ENABLE=1). Falls back to single-sample
    cached call when disabled.
    """
    # Normalize first
    instructions, input_text = _normalize_prompt_payload(instructions, input_text)
    # Hard clamp for streaming payloads (env-configurable)
    smax_i = _STREAM_INST_MAX_CHARS
    smax_t = _STREAM_INPUT_MAX_CHARS
    if len(instructions) > smax_i:
        instructions = instructions[:smax_i]
    if len(input_text) > smax_t:
        input_text = input_text[:smax_t]
    if not _is_code_like(input_text or ""):
        extra_kwargs: dict[str, Any] = {}
    else:
        try:
            extra_kwargs = build_file_search_tools()
        except Exception:
            extra_kwargs = {}
    return await call_openai_multi_validated(
        instructions=instructions,
        model=model,
        input_text=input_text,
        code_id=code_id,
        base_extra_kwargs=extra_kwargs,
    )


async def call_openai_stream_first_block(
    instructions: str,
    model: str,
    input_text: str,
    *,
    code_id: str,
    on_first_block: callable | None = None,
) -> str:
    """Stream Responses API, fire early when first complete <python_{code_id}> block appears.

    Fallback to validated non-stream call on any streaming error.
    """
    # Normalize first
    instructions, input_text = _normalize_prompt_payload(instructions, input_text)
    # File Search gating
    if not _is_code_like(input_text or ""):
        extra_kwargs: dict[str, Any] = {}
    else:
        try:
            extra_kwargs = build_file_search_tools()
        except Exception:
            extra_kwargs = {}

    ltag = f"<python_{code_id}>"
    rtag = f"</python_{code_id}>"
    buf: list[str] = []
    fired = False

    def _worker(q: _queue.Queue[str]) -> None:
        try:
            client = get_openai_client()
            # Prefer streaming API if available in SDK
            stream_fn = getattr(getattr(client, "responses", client), "stream", None)
            if stream_fn is None:
                raise RuntimeError("responses.stream_not_supported")
            with client.responses.stream(
                instructions=instructions,
                model=model,
                input=input_text,
                **{k: v for k, v in (extra_kwargs or {}).items() if not str(k).startswith("__")},
            ) as stream:
                for event in stream:
                    try:
                        typ = getattr(event, "type", "") or ""
                    except Exception:
                        typ = ""
                    piece = ""
                    # Common event types in Responses streaming
                    if typ.endswith(".delta"):
                        piece = getattr(event, "delta", "") or ""
                    elif typ.endswith("output_text"):
                        piece = getattr(event, "output_text", "") or getattr(event, "text", "") or ""
                    else:
                        piece = getattr(event, "delta", "") or getattr(event, "text", "") or ""
                    if piece:
                        q.put(piece)
            q.put("__DONE__")
        except Exception as e:
            q.put(f"__ERROR__:{e}")

    q: _queue.Queue[str] = _queue.Queue()
    # Run the streaming worker in a thread
    worker_task = _asyncio.create_task(_asyncio.to_thread(_worker, q))

    async def _get_next() -> str:
        return await _asyncio.to_thread(q.get)

    try:
        while True:
            chunk = await _get_next()
            if chunk == "__DONE__":
                break
            if isinstance(chunk, str) and chunk.startswith("__ERROR__:"):
                raise RuntimeError(chunk)
            if chunk:
                buf.append(chunk)
                if not fired:
                    text = "".join(buf)
                    if ltag in text:
                        li = text.find(ltag)
                        ri = text.find(rtag, li + len(ltag))
                        if ri != -1:
                            body = text[li + len(ltag): ri]
                            if (body or "").strip():
                                fired = True
                                if on_first_block:
                                    try:
                                        _asyncio.create_task(on_first_block(body, code_id))
                                    except Exception:
                                        pass
        # Join worker
        try:
            await worker_task
        except Exception:
            pass
        return "".join(buf)
    except Exception:
        # Fallback to validated path (single outbound call if streaming never started)
        return await call_openai_validated(instructions, model, input_text, code_id=code_id)
