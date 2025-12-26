from __future__ import annotations

import os
import time as _time
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
from jinx.micro.event_synthesis.api import (
    build_event_stream_block as _build_event_stream_block,
    build_event_state_block as _build_event_state_block,
)
from jinx.micro.llm.prompt_brain import compose_policy_tail, record_prompt_outcome

# AutoBrain adaptive configuration
try:
    from jinx.micro.runtime.autobrain_config import (
        get_int as _ab_int,
        record_outcome as _ab_record,
        record_failure as _ab_fail,
    )
    _AB_OK = True
except Exception:
    _AB_OK = False
    def _ab_int(n, c=None): return 20000
    def _ab_record(n, s, l=0, c=None): pass
    def _ab_fail(c, e): pass

_PROMPT_MACRO_MAX = 50
_DEFAULT_PROMPT = "burning_logic"
_DEFAULT_MODEL = "gpt-4.1"

async def code_primer(
    prompt_override: str | None = None,
    *,
    has_code_task: bool = True,
    has_complex_reasoning: bool = False,
    has_embeddings: bool = False,
    is_error_recovery: bool = False,
    task_count: int = 0,
) -> tuple[str, str]:
    """Build instruction header and return it with a code tag identifier.

    Returns (header_plus_prompt, code_tag_id).
    Uses modular prompt system to minimize tokens.
    """
    return await build_header_and_tag(
        prompt_override,
        has_code_task=has_code_task,
        has_complex_reasoning=has_complex_reasoning,
        has_embeddings=has_embeddings,
        is_error_recovery=is_error_recovery,
        task_count=task_count,
    )




async def _prepare_request(txt: str, *, prompt_override: str | None = None) -> tuple[str, str, str, str, str]:
    """Compose instructions with brain systems integration and ML enhancement."""
    # Detect task characteristics for modular prompt
    _txt = txt or ""
    has_embeddings = "<embeddings_" in _txt
    has_error = "<error>" in _txt or "error" in _txt.lower()[:200]
    is_complex = len(_txt) > 1000 or has_embeddings
    
    jx, tag = await code_primer(
        prompt_override,
        has_code_task=True,
        has_complex_reasoning=is_complex,
        has_embeddings=has_embeddings,
        is_error_recovery=has_error,
    )
    # Precompute sanitized input for optional enrichers that need it earlier
    stxt = sanitize_prompt_for_external_api(_txt)
    
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
        if "<$board" not in jx:
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
        ev_state = _build_event_state_block(gid, max_chars=900)
        if ev_state:
            jx = jx + "\n" + ev_state + "\n"
        ev_block = _build_event_stream_block(gid, max_events=48, max_chars=1800)
        if ev_block:
            jx = jx + "\n" + ev_block + "\n"
        # Inject architectural memory context (what we're working on)
        try:
            from jinx.micro.runtime.arch_memory import build_context_block as _arch_ctx
            arch_block = _arch_ctx()
            if arch_block and len(arch_block) > 20:
                jx = jx + "\n" + arch_block + "\n"
        except Exception:
            pass
        # Inject evolution context (goals, learnings) - minimal, only active goals
        try:
            from jinx.micro.runtime.self_evolution import build_evolution_context as _evo_ctx
            evo_block = _evo_ctx()
            if evo_block and len(evo_block) > 20:
                jx = jx + "\n" + evo_block + "\n"
        except Exception:
            pass
        # Inject strategic plan context (active plans and current subtask)
        try:
            from jinx.micro.runtime.strategic_planner import build_plan_context as _plan_ctx
            plan_block = _plan_ctx()
            if plan_block and len(plan_block) > 20:
                jx = jx + "\n" + plan_block + "\n"
        except Exception:
            pass
        await _yield0()
        # Unified embeddings context (code+brain+refs+graph+memory)
        try:
            _ctx = await build_unified_context_for(txt or "", max_chars=None, max_time_ms=3000)
        except Exception:
            _ctx = ""
        have_unified_ctx = bool((_ctx or "").strip())
        if have_unified_ctx:
            _ctx_final = compact_context(_ctx)
            jx = jx + "\n" + _ctx_final + "\n"
        # Auto-inject helpful embedding macros so the user doesn't need to type them (fallback if unified ctx missing)
        if (not have_unified_ctx) and ("{{m:" not in jx or "{{m:emb:" not in jx or "{{m:mem:" not in jx):
            lines = await auto_context_lines(txt)
            if lines:
                jx = jx + "\n" + "\n".join(lines) + "\n"
        # Auto-inject code intelligence lines (usage/def) to help answer "where is this used?"
        if True:
            clines = await auto_code_lines(txt)
            if clines:
                jx = jx + "\n" + "\n".join(clines) + "\n"
        await _yield0()
        # Optional CodeGraph snippets enrichment
        try:
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
        if ("{{export:" not in jx or "{{export:last_patch_" not in jx):
            exp_lines = await _patch_exports_lines()
            if exp_lines:
                jx = jx + "\n" + "\n".join(exp_lines) + "\n"
        await _yield0()
        # Optionally include last verification results
        if ("{{export:" not in jx or "{{export:last_verify_" not in jx):
            vlines = await _verify_exports_lines()
            if vlines:
                jx = jx + "\n" + "\n".join(vlines) + "\n"
        await _yield0()
        # Optionally include last sandbox run artifacts (stdout/stderr/status) via macros
        if ("{{m:run:" not in jx):
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
        max_exp = _PROMPT_MACRO_MAX
        
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
        active_prompt = (prompt_override or _DEFAULT_PROMPT)
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
    model = _DEFAULT_MODEL
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

    try:
        return await detonate_payload(openai_task, retries=1)
    except Exception:
        out = f"<python_{tag}>\nprint(\"Jinx offline: LLM unavailable.\")\n</python_{tag}>"
        return (out, tag)


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

    try:
        return await detonate_payload(openai_task, retries=1)
    except Exception:
        out = f"<python_{tag}>\nprint(\"Jinx offline: LLM unavailable.\")\n</python_{tag}>"
        return (out, tag)
