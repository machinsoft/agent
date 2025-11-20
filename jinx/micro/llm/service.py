from __future__ import annotations

import os
from jinx.openai_mod import build_header_and_tag
from .openai_caller import call_openai, call_openai_validated, call_openai_stream_first_block
from jinx.log_paths import OPENAI_REQUESTS_DIR_GENERAL
from jinx.logger.openai_requests import write_openai_request_dump, write_openai_response_append
from jinx.micro.memory.storage import write_token_hint
from jinx.retry import detonate_payload
from .prompt_compose import compose_dynamic_prompt
from .macro_registry import MacroContext, expand_dynamic_macros
from .macro_providers import register_builtin_macros
from .macro_plugins import load_macro_plugins
from jinx.micro.conversation.cont import load_last_anchors
import platform
import sys
import datetime as _dt
from .prompt_filters import sanitize_prompt_for_external_api
from jinx.micro.text.heuristics import is_code_like as _is_code_like
import asyncio as _asyncio
from jinx.micro.rt.timing import timing_section
from jinx.micro.embeddings.unified_context import build_unified_context_for
from jinx.micro.embeddings.context_compact import compact_context
from jinx.micro.llm.enrichers import auto_context_lines, auto_code_lines
from jinx.micro.llm.enrichers.exports import (
    patch_exports_lines as _patch_exports_lines,
    verify_exports_lines as _verify_exports_lines,
    run_exports_lines as _run_exports_lines,
)
from jinx.micro.memory.turn_summaries import read_group_summary as _read_group_summary
from jinx.micro.runtime.task_ctx import get_current_group as _get_group
from jinx.micro.llm.prompt_brain import compose_policy_tail, record_prompt_outcome


async def code_primer(prompt_override: str | None = None) -> tuple[str, str]:
    """Build instruction header and return it with a code tag identifier.

    Returns (header_plus_prompt, code_tag_id).
    """
    return await build_header_and_tag(prompt_override)




async def _prepare_request(txt: str, *, prompt_override: str | None = None) -> tuple[str, str, str, str, str]:
    """Compose instructions with brain systems integration and ML enhancement."""
    jx, tag = await code_primer(prompt_override)
    # Precompute sanitized input for optional enrichers that need it earlier
    stxt = sanitize_prompt_for_external_api(txt or "")
    
    # Cooperative yield - always enabled for RT constraints
    async def _yield0() -> None:
        await _asyncio.sleep(0)
    
    # Process with context processor for machine-level understanding
    if txt and ("<embeddings_" in txt or "<user>" in txt or "<evidence>" in txt):
        from jinx.micro.llm.context_processor import process_all_context, build_agent_context_summary
        context_data = process_all_context(txt)
        agent_summary = build_agent_context_summary(context_data)
        
        # Inject agent summary as machine-level context
        if agent_summary:
            jx = jx + f"\n<!-- Machine Context Summary:\n{agent_summary}\n-->\n"
    # Expand dynamic prompt macros in real time (vars/env/anchors/sys/runtime/exports + custom providers)
    try:
        jx = await compose_dynamic_prompt(jx, key=tag)
        # Auto-inject compact board state macro (JIN-FEN) to replace long history
        try:
            board_on = str(os.getenv("JINX_BOARD_PROMPT", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            board_on = True
        if board_on and ("<$board" not in jx):
            jx = "<board fen>\n" + jx
        await _yield0()
        # Inject compact per-group rolling summary (if any)
        try:
            gid = _get_group()
        except Exception:
            gid = "main"
        try:
            gctx = await _read_group_summary(gid, max_chars=1200)
        except Exception:
            gctx = ""
        if gctx:
            jx = jx + "\n" + gctx + "\n"
        await _yield0()
        # Unified embeddings context (code+brain+refs+graph+memory)
        try:
            # Use configurable timeout for unified context
            unified_timeout = int(os.getenv("EMBED_UNIFIED_MAX_TIME_MS", "3000"))
            _ctx = await build_unified_context_for(txt or "", max_chars=None, max_time_ms=unified_timeout)
        except Exception:
            _ctx = ""
        have_unified_ctx = bool((_ctx or "").strip())
        if have_unified_ctx:
            try:
                # Default ON: machine-level compaction for <embeddings_*> blocks
                cmp_on = str(os.getenv("JINX_CTX_COMPACT", "1")).lower() not in ("", "0", "false", "off", "no")
            except Exception:
                cmp_on = True
            _ctx_final = compact_context(_ctx) if cmp_on else _ctx
            jx = jx + "\n" + _ctx_final + "\n"
        # Auto-inject helpful embedding macros so the user doesn't need to type them (fallback if unified ctx missing)
        try:
            auto_on = str(os.getenv("JINX_AUTOMACROS", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            auto_on = True
        if auto_on and (not have_unified_ctx) and ("{{m:" not in jx or "{{m:emb:" not in jx or "{{m:mem:" not in jx):
            lines = await auto_context_lines(txt)
            if lines:
                jx = jx + "\n" + "\n".join(lines) + "\n"
        # Auto-inject code intelligence lines (usage/def) to help answer "where is this used?"
        try:
            code_auto_on = str(os.getenv("JINX_AUTOMACRO_CODE", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            code_auto_on = True
        if code_auto_on:
            clines = await auto_code_lines(txt)
            if clines:
                jx = jx + "\n" + "\n".join(clines) + "\n"
        await _yield0()
        # Optional CodeGraph snippets enrichment
        try:
            if str(os.getenv("JINX_CODEGRAPH_CTX", "1")).lower() not in ("", "0", "false", "off", "no"):
                from jinx.codegraph.service import snippets_for_text as _cg_snips
                pairs = await _cg_snips(stxt, max_tokens=8, max_snippets=4)
                if pairs:
                    cg = []
                    for hdr, block in pairs:
                        cg.append(hdr)
                        cg.append(block)
                    jx = jx + "\n" + "\n".join(cg) + "\n"
        except Exception:
            pass
        # Optionally include recent patch previews/commits from runtime exports
        try:
            include_patch = str(os.getenv("JINX_AUTOMACRO_PATCH_EXPORTS", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            include_patch = True
        if include_patch and ("{{export:" not in jx or "{{export:last_patch_" not in jx):
            exp_lines = await _patch_exports_lines()
            if exp_lines:
                jx = jx + "\n" + "\n".join(exp_lines) + "\n"
        await _yield0()
        # Optionally include last verification results
        try:
            include_verify = str(os.getenv("JINX_AUTOMACRO_VERIFY_EXPORTS", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            include_verify = True
        if include_verify and ("{{export:" not in jx or "{{export:last_verify_" not in jx):
            vlines = await _verify_exports_lines()
            if vlines:
                jx = jx + "\n" + "\n".join(vlines) + "\n"
        await _yield0()
        # Optionally include last sandbox run artifacts (stdout/stderr/status) via macros
        try:
            include_run = str(os.getenv("JINX_AUTOMACRO_RUN_EXPORTS", "1")).lower() not in ("", "0", "false", "off", "no")
        except Exception:
            include_run = True
        if include_run and ("{{m:run:" not in jx):
            rlines = await _run_exports_lines(None)
            if rlines:
                jx = jx + "\n" + "\n".join(rlines) + "\n"
        # Build macro context and expand provider macros {{m:ns:arg1:arg2}}
        try:
            anc = await load_last_anchors()
        except Exception:
            anc = {}
        try:
            from jinx.micro.runtime.api import list_programs as _list_programs
            progs = await _list_programs()
        except Exception:
            progs = []
        await _yield0()
        ctx = MacroContext(
            key=tag,
            anchors={k: [str(x) for x in (anc.get(k) or [])] for k in ("questions","symbols","paths")},
            programs=progs,
            os_name=platform.system(),
            py_ver=sys.version.split(" ")[0],
            cwd=os.getcwd() if hasattr(os, "getcwd") else "",
            now_iso=_dt.datetime.now().isoformat(timespec="seconds"),
            now_epoch=str(int(_dt.datetime.now().timestamp())),
            input_text=txt or "",
        )
        # Ensure built-in providers and plugin macros are registered/loaded
        # Initialize macro providers/plugins once per process
        import asyncio as _asyncio
        _init_lock = getattr(spark_openai, "_macro_init_lock", None)
        if _init_lock is None:
            _init_lock = _asyncio.Lock()
            setattr(spark_openai, "_macro_init_lock", _init_lock)
        if not getattr(spark_openai, "_macro_inited", False):
            async with _init_lock:
                if not getattr(spark_openai, "_macro_inited", False):
                    try:
                        from jinx.micro.logger.debug_logger import debug_log
                        await debug_log("Registering builtin macros...", "MACROS")
                        await register_builtin_macros()
                        await debug_log("✓ Builtin macros registered", "MACROS")
                    except Exception as e:
                        try:
                            await debug_log(f"✗ Failed to register builtin macros: {e}", "MACROS")
                        except Exception:
                            pass
                    try:
                        await load_macro_plugins()
                    except Exception:
                        pass
                    setattr(spark_openai, "_macro_inited", True)
        try:
            max_exp = int(os.getenv("JINX_PROMPT_MACRO_MAX", "50"))
        except Exception:
            max_exp = 50
        
        # Check for macros in prompt
        import re
        from jinx.micro.logger.debug_logger import debug_log
        macro_pattern = r'<\$(\w+)(?:\s+[^>]+)?>'
        macros_found = re.findall(macro_pattern, jx)
        
        if macros_found:
            await debug_log(f"Found {len(macros_found)} macros in prompt: {', '.join(set(macros_found))}", "MACROS")
        else:
            await debug_log("No macros found in prompt", "MACROS")
        
        await debug_log(f"Expanding macros (max={max_exp})...", "MACROS")
        jx_before_len = len(jx)
        jx = await expand_dynamic_macros(jx, ctx, max_expansions=max_exp)
        jx_after_len = len(jx)
        
        if jx_after_len != jx_before_len:
            await debug_log(f"✓ Expansion changed prompt size: {jx_before_len} → {jx_after_len} chars", "MACROS")
        else:
            await debug_log(f"No expansion occurred (size unchanged: {jx_before_len} chars)", "MACROS")
        await _yield0()
        # Append policy tail (prompt brain) to bias outputs based on past outcomes
        try:
            tail = await compose_policy_tail(tag, stxt, have_unified_ctx=have_unified_ctx)
            if tail:
                jx = jx + tail
        except Exception:
            pass
        # Best-effort token hint (chars/4 heuristic) for dynamic memory budgets
        try:
            est_tokens = max(0, (len(jx) + len(txt or "")) // 4)
            await write_token_hint(est_tokens)
        except Exception:
            pass
    except Exception:
        pass
    # Dynamic context guide for burning_logic prompt only
    try:
        active_prompt = (prompt_override or os.getenv("JINX_PROMPT") or os.getenv("PROMPT_NAME") or "burning_logic")
        if active_prompt.strip().lower() == "burning_logic":
            # Detect all context tags present in final composed prompt
            import re as _re
            context_tags = set(_re.findall(r"<([a-z_]+)>", jx))
            from jinx.prompts.burning_logic import _context_guide as _bl_guide
            guide = _bl_guide(context_tags)
            if guide:
                # Insert after header, before main content
                lines = jx.split("\n", 10)
                # Find insertion point (after header lines starting with key:/os:/arch:/etc)
                insert_idx = 0
                for i, ln in enumerate(lines):
                    if ln.startswith(("pulse:", "key:", "os:", "arch:", "host:", "user:")):
                        insert_idx = i + 1
                    else:
                        break
                lines.insert(insert_idx, guide)
                jx = "\n".join(lines)
    except Exception:
        pass
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    # Sanitize prompts to avoid leaking internal .jinx paths/content
    sx = sanitize_prompt_for_external_api(jx)
    return jx, tag, model, sx, stxt


async def spark_openai(txt: str, *, prompt_override: str | None = None) -> tuple[str, str]:
    """Call OpenAI Responses API and return output text with the code tag.

    Returns (output_text, code_tag_id).
    """
    jx, tag, model, sx, stxt = await _prepare_request(txt, prompt_override=prompt_override)

    async def openai_task() -> tuple[str, str]:
        req_path: str = ""
        import asyncio as _asyncio
        # Overlap request dump with LLM call
        dump_task = _asyncio.create_task(write_openai_request_dump(
            target_dir=OPENAI_REQUESTS_DIR_GENERAL,
            kind="GENERAL",
            instructions=sx,
            input_text=stxt,
            model=model,
        ))
        # Preferred: validated multi-sample path
        try:
            from jinx.observability.otel import span as _span
        except Exception:
            from contextlib import nullcontext as _span  # type: ignore
        try:
            async with timing_section("llm.call"):
                with _span("llm.call"):
                    out = await call_openai_validated(sx, model, stxt, code_id=tag)
        except Exception:
            # Fallback to legacy single-sample on error
            async with timing_section("llm.call_legacy"):
                with _span("llm.call_legacy"):
                    out = await call_openai(sx, model, stxt)
        # If still empty (e.g., provider forbidden region soft-path), fallback to legacy path
        if not (out or "").strip():
            try:
                async with timing_section("llm.call_empty_fallback"):
                    out = await call_openai(sx, model, stxt)
            except Exception:
                out = out or ""
        # Tiny consensus refinement (budgeted)
        try:
            from .consensus import refine_output as _refine
            out = await _refine(sx, model, stxt, out)
        except Exception:
            pass
        # Get dump path (await, then append in background)
        try:
            req_path = await dump_task
        except Exception:
            req_path = ""
        try:
            _asyncio.create_task(write_openai_response_append(req_path, "GENERAL", out))
        except Exception:
            pass
        # Record outcome in prompt brain (best-effort)
        try:
            await record_prompt_outcome(tag, out)
        except Exception:
            pass
        return (out, tag)

    # Avoid duplicate outbound API calls on post-call exceptions by disabling retries here.
    # Lower-level resiliency is provided by caching/coalescing/multi-path logic.
    return await detonate_payload(openai_task, retries=1)


async def spark_openai_streaming(txt: str, *, prompt_override: str | None = None, on_first_block=None) -> tuple[str, str]:
    """Streaming LLM call with early execution on first complete <python_{tag}> block.

    Returns (full_output_text, code_tag_id).
    """
    jx, tag, model, sx, stxt = await _prepare_request(txt, prompt_override=prompt_override)

    async def openai_task() -> tuple[str, str]:
        req_path: str = ""
        import asyncio as _asyncio
        dump_task = _asyncio.create_task(write_openai_request_dump(
            target_dir=OPENAI_REQUESTS_DIR_GENERAL,
            kind="GENERAL",
            instructions=sx,
            input_text=stxt,
            model=model,
        ))
        try:
            from jinx.observability.otel import span as _span
        except Exception:
            from contextlib import nullcontext as _span  # type: ignore
        try:
            async with timing_section("llm.stream"):
                with _span("llm.stream"):
                    out = await call_openai_stream_first_block(sx, model, stxt, code_id=tag, on_first_block=on_first_block)
        except Exception:
            async with timing_section("llm.call_fallback"):
                with _span("llm.call_fallback"):
                    out = await call_openai_validated(sx, model, stxt, code_id=tag)
        # Empty output guard: fallback to validated, then legacy single-sample as last resort
        if not (out or "").strip():
            try:
                async with timing_section("llm.stream_empty_fallback_validated"):
                    out = await call_openai_validated(sx, model, stxt, code_id=tag)
            except Exception:
                pass
            if not (out or "").strip():
                try:
                    async with timing_section("llm.stream_empty_fallback_legacy"):
                        out = await call_openai(sx, model, stxt)
                except Exception:
                    out = out or ""
        # Tiny consensus refinement (budgeted)
        try:
            from .consensus import refine_output as _refine
            out = await _refine(sx, model, stxt, out)
        except Exception:
            pass
        try:
            req_path = await dump_task
        except Exception:
            req_path = ""
        try:
            _asyncio.create_task(write_openai_response_append(req_path, "GENERAL", out))
        except Exception:
            pass
        # Record outcome in prompt brain (best-effort)
        try:
            await record_prompt_outcome(tag, out)
        except Exception:
            pass
        return (out, tag)

    return await detonate_payload(openai_task, retries=1)
