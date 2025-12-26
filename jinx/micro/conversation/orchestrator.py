from __future__ import annotations

import traceback
from typing import Optional
import re
import asyncio
import time

# AutoBrain adaptive configuration
try:
    from jinx.micro.runtime.autobrain_config import (
        get_int as _ab_int,
        get as _ab_get,
        record_outcome as _ab_record,
        record_failure as _ab_fail,
        TaskContext as _AbCtx,
    )
    _AB_OK = True
except Exception:
    _AB_OK = False
    def _ab_int(n, c=None): return 45
    def _ab_get(n, c=None): return 45.0
    def _ab_record(n, s, l=0, c=None): pass
    def _ab_fail(c, e): pass
    class _AbCtx:
        def __init__(self, **kw): pass

from jinx.logging_service import glitch_pulse, bomb_log, blast_mem
from jinx.openai_service import spark_openai
from jinx.error_service import dec_pulse
from jinx.conversation import build_chains, run_blocks
from jinx.micro.ui.output import pretty_echo, pretty_echo_async as _pretty_echo_async
from jinx.micro.conversation.sandbox_view import show_sandbox_tail
from jinx.micro.conversation.error_report import corrupt_report
from jinx.logger.file_logger import append_line as _log_append
from jinx.log_paths import BLUE_WHISPERS
from jinx.micro.recursor.normalizer import normalize_output_blocks
from jinx.micro.parser.api import parse_tagged_blocks
from jinx.micro.embeddings.retrieval import build_context_for
from jinx.micro.embeddings.project_retrieval import build_project_context_for, build_project_context_multi_for
from jinx.micro.embeddings.pipeline import embed_text
from jinx.conversation.formatting import build_header, ensure_header_block_separation
from jinx.micro.embeddings.unifier import build_unified_brain_block as _build_unified_brain
from jinx.micro.memory.storage import read_evergreen
from jinx.micro.memory.storage import read_channel as _read_channel
from jinx.micro.conversation.memory_sanitize import sanitize_transcript_for_memory
from jinx.micro.embeddings.project_config import ENABLE as PROJ_EMB_ENABLE
from jinx.micro.embeddings.project_paths import PROJECT_FILES_DIR
from jinx.micro.llm.chain_persist import persist_memory
from jinx.micro.llm.kernel_sanitizer import sanitize_kernels as _sanitize_kernels
from jinx.micro.exec.executor import spike_exec as _spike_exec
from jinx.safety import chaos_taboo as _chaos_taboo
from jinx.micro.runtime.patcher import ensure_patcher_running as _ensure_patcher
from jinx.micro.conversation.cont import (
    augment_query_for_retrieval as _augment_query,
    maybe_reuse_last_context as _reuse_proj_ctx,
    save_last_context as _save_proj_ctx,
    extract_anchors as _extract_anchors,
    load_last_anchors as _load_last_anchors,
    render_continuity_block as _render_cont_block,
    last_agent_question as _last_q,
    last_user_query as _last_u,
    is_short_followup as _is_short,
    detect_topic_shift as _topic_shift,
    maybe_compact_state_frames as _compact_frames,
)
from jinx.micro.memory.turn_summaries import append_group_summary as _append_group_summary
from jinx.micro.conversation.cont.classify import find_semantic_question as _find_semq
from jinx.micro.conversation.state_frame import build_state_frame
from jinx.micro.memory.router import assemble_memroute as _memroute
from jinx.micro.runtime.api import ensure_runtime as _ensure_runtime
from jinx.micro.verify.verifier import ensure_verifier_running as _ensure_verifier
from jinx.micro.text.heuristics import is_code_like as _is_code_like
from jinx.micro.llm.service import spark_openai as _spark_llm, spark_openai_streaming as _spark_llm_stream
from jinx.micro.conversation.proj_context_enricher import build_project_context_enriched as _build_proj_ctx_enriched
from jinx.micro.conversation.error_payload import attach_error_code as _attach_error_code
from jinx.micro.memory.evergreen_select import select_evergreen_for as _select_evg
from jinx.micro.embeddings.memory_context import build_memory_context_for as _build_mem_ctx
from jinx.micro.memory.api_memory import build_api_memory_block as _build_api_mem, append_turn as _append_turn
from jinx.micro.conversation.turns_infer import infer_turn_query as _infer_turn
from jinx.micro.memory.turns import get_user_message as _turn_user, get_jinx_reply_to as _turn_jinx, parse_active_turns as _turns_all
from jinx.micro.conversation.memory_reasoner import infer_memory_action as _infer_memsel
from jinx.micro.memory.pin_store import load_pins as _pins_load
from jinx.micro.conversation.prefilter import likely_memory_action as _likely_mem
from jinx.micro.conversation.debug import log_debug
from jinx.micro.conversation.memory_program import infer_memory_program as _mem_program
from jinx.micro.embeddings.context_compact import compact_context as _compact_ctx
from jinx.micro.memory.telemetry import log_metric as _log_metric
from jinx.micro.rt.coop import coop as _coop
from jinx.micro.runtime.task_ctx import get_current_group as _get_group
from jinx.micro.event_synthesis.api import record_event as _record_event
from jinx.micro.rt.activity import set_activity as _act, clear_activity as _act_clear, set_activity_detail as _actdet, clear_activity_detail as _actdet_clear
from jinx.micro.rt.rt_budget import run_bounded as _bounded_run
from jinx.micro.conversation.phases import (
    call_llm as _phase_llm,
    execute_blocks as _phase_exec,
    build_runtime_base_ctx as _phase_base_ctx,
    build_runtime_mem_ctx as _phase_mem_ctx,
    build_project_context_enriched as _phase_proj_ctx,
)


async def shatter(x: str, err: Optional[str] = None, turn_id: Optional[str] = None) -> None:
    """Drive a single conversation step and optionally handle an error context.

    turn_id: Optional external correlation id from the scheduler for observability.
    """
    from jinx.micro.logger.debug_logger import debug_log
    await debug_log(f"START processing [id={turn_id or '-'}]: {x[:80]}", "SHATTER")
    gid = _get_group()
    _record_event("turn.start", {"turn_id": turn_id or "", "has_error": bool(err), "input_len": len(x or "")}, group=gid, weight=4.0)
    
    # === BRAIN TASK TRACKING ===
    _brain_task_id: str = ""
    try:
        from jinx.micro.runtime.brain import get_brain
        _brain = get_brain()
        _brain_task_id = _brain.begin_task((x or "")[:120], {"group": gid, "turn_id": turn_id})
    except Exception:
        pass
    
    # === DYNAMIC CONFIGURATION ADAPTATION ===
    # AI-powered configuration tuning based on request type
    request_start_time = time.time()
    try:
        from jinx.micro.runtime.dynamic_config_plugin import adapt_config_for_request
        
        # Build context for task detection
        adaptation_context = {
            'recent_error': bool(err),
            'transcript_size': 0,  # Will be filled below
        }
        
        try:
            await asyncio.wait_for(adapt_config_for_request(x, adaptation_context), timeout=0.25)
        except Exception:
            try:
                asyncio.create_task(adapt_config_for_request(x, adaptation_context))
            except Exception:
                pass
    except Exception:
        pass  # Silent fail - don't break on adaptation errors
    
    try:
        # Ensure micro-program runtime and event bridge are active before any code execution
        try:
            await debug_log("Ensuring runtime...", "SHATTER")
            try:
                await asyncio.wait_for(_ensure_runtime(), timeout=0.8)
            except Exception:
                try:
                    asyncio.create_task(_ensure_runtime())
                except Exception:
                    pass
            try:
                await asyncio.wait_for(_ensure_patcher(), timeout=0.3)
            except Exception:
                try:
                    asyncio.create_task(_ensure_patcher())
                except Exception:
                    pass
            try:
                await asyncio.wait_for(_ensure_verifier(), timeout=0.3)
            except Exception:
                try:
                    asyncio.create_task(_ensure_verifier())
                except Exception:
                    pass
            await debug_log("Runtime ready", "SHATTER")
        except Exception as e:
            await debug_log(f"Runtime setup error: {e}", "SHATTER")
            pass
        # Append the user input to the transcript first to ensure ordering
        await debug_log("Appending to transcript...", "SHATTER")
        if x and x.strip():
            _record_event("user.input", {"turn_id": turn_id or "", "text": x.strip()}, group=gid, weight=6.0)
            try:
                await asyncio.wait_for(blast_mem(f"User: {x.strip()}"), timeout=0.25)
            except Exception:
                try:
                    asyncio.create_task(blast_mem(f"User: {x.strip()}"))
                except Exception:
                    pass
            # Also embed the raw user input for retrieval (source: dialogue) in background
            try:
                asyncio.create_task(embed_text(x.strip(), source="dialogue", kind="user"))
            except Exception:
                pass
        await debug_log("Reading transcript...", "SHATTER")
        _act("reading transcript")
        try:
            synth = await asyncio.wait_for(glitch_pulse(), timeout=0.6)
        except Exception:
            synth = ""
        await _coop()
        # Do not include the transcript in 'chains' since it is placed into <memory>
        # Do not inject error text into the body chains; it will live in <error>
        await debug_log("Building chains...", "SHATTER")
        _act("building prompt header")
        chains, decay = build_chains("", None)
        await _coop()
        await debug_log(f"Chains built, decay={decay}", "SHATTER")
        # Build standardized header blocks in a stable order before the main chains
        # 1) <embeddings_context> from recent dialogue/sandbox using current input as query,
        #    plus project code embeddings context assembled from emb/ when available
        # Continuity: augment retrieval query on short clarifications
        topic_shifted = False
        reuse_for_log = False
        try:
            # Prefer error text for retrieval if present; fallback to user text, then transcript
            q_raw = (x or err or synth or "")
            continuity_on = True
            anchors = {}
            if continuity_on:
                try:
                    cur = _extract_anchors(synth or "")
                except Exception:
                    cur = {}
                # Optional: boost with semantic question detector (language-agnostic)
                try:
                    semq = await _find_semq(synth or "")
                    if semq:
                        qs = [semq]
                        for qline in (cur.get("questions") or []):
                            if qline != semq:
                                qs.append(qline)
                        cur["questions"] = qs
                except Exception:
                    pass
                try:
                    prev = await _load_last_anchors()
                except Exception:
                    prev = {}
                # merge anchors (current first, then previous uniques), cap lists
                anchors = {k: list(dict.fromkeys((cur.get(k) or []) + (prev.get(k) or [])))[:10] for k in set((cur or {}).keys()) | set((prev or {}).keys())}
                eff_q = _augment_query((x or err or ""), synth or "", anchors=anchors)
            else:
                eff_q = q_raw
            # Launch runtime context retrieval concurrently with project context assembly
            await debug_log("Launching context retrieval tasks...", "SHATTER")
            _act("retrieving runtime context")
            base_ctx_task = asyncio.create_task(_phase_base_ctx(eff_q))
            # Optional: embeddings-backed memory context (no evergreen), env-gated (default OFF for API)
            mem_ctx_task = None
            try:
                mem_ctx_task = asyncio.create_task(_phase_mem_ctx(eff_q))
            except Exception:
                mem_ctx_task = None
            await _coop()
        except Exception as e:
            await debug_log(f"Exception in context launch: {e}", "SHATTER")
            base_ctx_task = asyncio.create_task(asyncio.sleep(0.0))  # type: ignore
        # Always build project context; retrieval enforces its own tight budgets
        await debug_log("Building project context...", "SHATTER")
        proj_ctx = ""
        try:
            _q = eff_q
            # Delegate enrichment/build to the dedicated micro-module (deduplicated logic)
            _act("assembling project context")
            proj_ctx_task: asyncio.Task[str] = asyncio.create_task(_phase_proj_ctx(_q, user_text=x or "", synth=synth or ""))
            # Await both contexts with strict RT budgets (disable by setting env to 0)
            try:
                _actdet({"stage": "base_ctx", "rem_ms": 220})
            except Exception:
                pass
            await debug_log("Awaiting base_ctx_task...", "SHATTER")
            try:
                base_ctx = await base_ctx_task
                await debug_log("base_ctx_task done", "SHATTER")
            except asyncio.CancelledError as e:
                await debug_log("base_ctx_task TIMEOUT/CANCELLED - continuing with empty context", "SHATTER")
                base_ctx = ""  # Graceful degradation
            except Exception as e:
                await debug_log(f"base_ctx_task failed: {e}", "SHATTER")
                base_ctx = ""  # Graceful degradation
            # Await memory context if launched (internal only)
            mem_ctx = ""
            if 'mem_ctx_task' in locals() and mem_ctx_task is not None:
                try:
                    _actdet({"stage": "mem_ctx", "rem_ms": 160})
                except Exception:
                    pass
                mem_ctx = await mem_ctx_task
            try:
                _actdet({"stage": "proj_ctx", "rem_ms": 260})
            except Exception:
                pass
            await debug_log("Awaiting proj_ctx_task...", "SHATTER")
            try:
                proj_ctx = await proj_ctx_task
                await debug_log("proj_ctx_task done", "SHATTER")
            except asyncio.CancelledError as e:
                await debug_log("proj_ctx_task TIMEOUT/CANCELLED - continuing with empty context", "SHATTER")
                proj_ctx = ""  # Graceful degradation - continue without project context
            except Exception as e:
                await debug_log(f"proj_ctx_task failed: {e}", "SHATTER")
                proj_ctx = ""  # Graceful degradation
            await _coop()
            await debug_log(f"Contexts retrieved: base={len(base_ctx)}, proj={len(proj_ctx)}", "SHATTER")
            
            # Check if shutdown requested
            import jinx.state as _jx_state
            if _jx_state.shutdown_event.is_set():
                await debug_log("SHUTDOWN EVENT IS SET! Exiting...", "SHATTER")
                return
            # _build_proj_ctx_enriched already implements its own fallback
            # Continuity: if still empty and this is a short clarification, reuse last cached project context
            if not proj_ctx:
                reuse = ""
                try:
                    ts_check = True
                    if ts_check and _is_short(x or ""):
                        shifted = await _topic_shift(_q)
                        topic_shifted = topic_shifted or bool(shifted)
                        if not shifted:
                            reuse = await _reuse_proj_ctx(x or "", proj_ctx, synth or "")
                    else:
                        reuse = await _reuse_proj_ctx(x or "", proj_ctx, synth or "")
                except Exception:
                    reuse = ""
                if reuse:
                    proj_ctx = reuse
                    reuse_for_log = True
        except Exception as e:
            # If project assembly failed early, still await runtime context task
            await debug_log(f"Exception in project context assembly: {e}", "SHATTER")
            try:
                base_ctx = await base_ctx_task
            except Exception as e2:
                await debug_log(f"Exception awaiting base_ctx_task in fallback: {e2}", "SHATTER")
                base_ctx = ""
            proj_ctx = ""
        # If runtime retrieval wasn't launched (fallback path), ensure base_ctx exists
        if 'base_ctx' not in locals():
            try:
                base_ctx = await _phase_base_ctx(eff_q)
            except Exception:
                base_ctx = ""
        if 'mem_ctx' not in locals():
            mem_ctx = ""
        await _coop()
        # Persist last project context snapshot for continuity cache
        try:
            await _save_proj_ctx(proj_ctx or "", anchors=anchors if 'anchors' in locals() else None)
        except Exception:
            pass
        plan_ctx = ""
        # Optional continuity block for the main brain
        try:
            cont_block = _render_cont_block(
                anchors if 'anchors' in locals() else None,
                _last_q(synth or ""),
                _last_u(synth or ""),
                _is_short(x or ""),
            )
        except Exception:
            cont_block = ""
        # Eagerly embed continuity block (trimmed) for state retrieval coverage
        if cont_block:
            try:
                asyncio.create_task(embed_text((cont_block or "")[:512], source="state", kind="cont"))
            except Exception:
                pass
        await _coop()
        # Optional: auto-turns resolver — hybrid (fast+LLM) that injects a tiny <turns> block when the user asks about Nth message
        turns_block = ""
        try:
            tq = await _infer_turn(x or "")
        except Exception:
            tq = None
        if tq:
            # Confidence gating to avoid false positives
            conf_min = 0.3
            try:
                kind = (tq.get("kind") or "pair").strip().lower()
                idx = int(tq.get("index") or 0)
                conf = float(tq.get("confidence") or 0.0)
            except Exception:
                kind = "pair"; idx = 0; conf = 0.0
            if idx > 0 and conf >= conf_min:
                try:
                    cap_one = 800
                    if kind == "user":
                        body = (await _turn_user(idx))
                        if body:
                            if cap_one > 0 and len(body) > cap_one:
                                body = body[:cap_one]
                            turns_block = f"<turns>\n[User:{idx}]\n{body}\n</turns>"
                    elif kind == "jinx":
                        body = (await _turn_jinx(idx))
                        if body:
                            if cap_one > 0 and len(body) > cap_one:
                                body = body[:cap_one]
                            turns_block = f"<turns>\n[Jinx:{idx}]\n{body}\n</turns>"
                    else:
                        turns = await _turns_all()
                        if 0 < idx <= len(turns):
                            u = (turns[idx-1].get("user") or "").strip()
                            a = (turns[idx-1].get("jinx") or "").strip()
                            cap_pair = 1200
                            tiny = (u + "\n" + a).strip()
                            if cap_pair > 0 and len(tiny) > cap_pair:
                                tiny = tiny[:cap_pair]
                            turns_block = f"<turns>\n[Pair:{idx}]\n{tiny}\n</turns>"
                except Exception:
                    turns_block = ""
        # Eagerly embed turns block (trimmed) for state retrieval coverage
        if turns_block:
            try:
                asyncio.create_task(embed_text((turns_block or "")[:512], source="state", kind="turns"))
            except Exception:
                pass

        # Optional: memory program — plan+execute ops (memroute/pins/topics/channels). Prefer this over simple selector.
        memsel_block = ""
        prog_blocks: dict[str, str] = {}
        try:
            if _likely_mem(x or ""):
                prog_blocks = await _mem_program(x or "")
        except Exception:
            prog_blocks = {}
        # Merge any blocks returned by program
        if prog_blocks:
            try:
                if prog_blocks.get("memory_selected"):
                    memsel_block = prog_blocks["memory_selected"].strip()
                if prog_blocks.get("pins"):
                    pins_block = prog_blocks["pins"].strip()
                    memsel_block = (memsel_block + "\n\n" + pins_block).strip() if memsel_block else pins_block
            except Exception:
                memsel_block = memsel_block or ""

        # If program produced nothing, fall back to memory reasoner — decide if routed memory or pins should be injected compactly
        ma = None
        if not memsel_block:
            try:
                if _likely_mem(x or ""):
                    ma = await _infer_memsel(x or "")
            except Exception:
                ma = None
        if (not memsel_block) and ma:
            mconf_min = 0.4
            action = str(ma.get("action") or "").strip().lower()
            params = ma.get("params") or {}
            try:
                mconf = float(ma.get("confidence") or 0.0)
            except Exception:
                mconf = 0.0
            if mconf >= mconf_min:
                try:
                    if action == "memroute":
                        q = str(params.get("query") or "")
                        try:
                            kk = int(params.get("k") or 0)
                        except Exception:
                            kk = 0
                        if kk <= 0:
                            kk = 8
                        kk = max(1, min(16, kk))
                        pv = 160
                        lines = await _memroute(q, k=kk, preview_chars=pv)
                        body = "\n".join([ln for ln in (lines or [])[:kk] if ln])
                        cap = 1200
                        if body and cap > 0 and len(body) > cap:
                            body = body[:cap]
                        if body:
                            memsel_block = f"<memory_selected>\n{body}\n</memory_selected>"
                    elif action == "pins":
                        try:
                            pins = _pins_load()
                        except Exception:
                            pins = []
                        if pins:
                            body = "\n".join(pins[:8])
                            memsel_block = f"<pins>\n{body}\n</pins>"
                except Exception:
                    memsel_block = ""
        elif _likely_mem(x or ""):
            # Fallback path: inject a minimal memroute block using the whole question as query
            try:
                await log_debug("JINX_MEMSEL", "fallback_memroute")
                fb_k = 4
                fb_k = max(1, min(8, fb_k))
                pv = 160
                q = (x or "").strip()[:240]
                lines = await _memroute(q, k=fb_k, preview_chars=pv)
                body = "\n".join([ln for ln in (lines or [])[:fb_k] if ln])
                if body:
                    cap = 900
                    if len(body) > cap:
                        body = body[:cap]
                    memsel_block = f"<memory_selected>\n{body}\n</memory_selected>"
            except Exception:
                pass

        # Do NOT send <embeddings_memory> to API by default; keep only base/project/planner/continuity (+ optional <turns>/<memory_selected>)
        # Eagerly embed memory selection block (trimmed) for state retrieval coverage
        if memsel_block:
            try:
                asyncio.create_task(embed_text((memsel_block or "")[:512], source="state", kind="memsel"))
            except Exception:
                pass
        ctx = "\n".join([c for c in [base_ctx, proj_ctx, cont_block, turns_block, memsel_block] if c])
        # Integrate resolved resources (from Resource Locator plugin) into context
        try:
            import jinx.state as _jx_state
            resolved = list(getattr(_jx_state, 'resolved_resources_last', []) or [])
            primary = getattr(_jx_state, 'primary_resource', None)
        except Exception:
            resolved = []
            primary = None
        resolved_block = ""
        preview_block = ""
        if resolved:
            try:
                lines = []
                for r in resolved[:5]:
                    try:
                        sc = float(r.get('score') or 0.0)
                    except Exception:
                        sc = 0.0
                    rel = (r.get('rel') or r.get('path') or '').strip()
                    lines.append(f"{sc:.2f} {rel}")
                if lines:
                    resolved_block = "<resolved_files>\n" + "\n".join(lines) + "\n</resolved_files>"
            except Exception:
                resolved_block = ""
        # Primary file preview (small size)
        if primary and isinstance(primary, dict):
            try:
                from jinx.micro.runtime.file_reader import read_file_preview as _read_preview
                path = str(primary.get('path') or '')
                rel = str(primary.get('rel') or '')
                if path:
                    text, _tr = _read_preview(path, max_chars=4000, head_lines=200, tail_lines=80)
                    if text:
                        show_path = rel or path
                        preview_block = f"<file_preview path=\"{show_path}\">\n" + text + "\n</file_preview>"
            except Exception:
                preview_block = ""
        if resolved_block or preview_block:
            ctx = "\n".join([c for c in [ctx, resolved_block, preview_block] if c])

        # Automated action routing: attempt self-executing code modification
        # Disabled embedding of auto_action_report into prompts to avoid polluting OpenAI API input.
        # Routing still executes internally if enabled, but no report is appended to ctx.
        try:
            from jinx.micro.runtime.action_router import auto_route_and_execute as _auto_route
            _ = await _auto_route(x or "", budget_ms=1500)
        except Exception:
            pass
        # Compaction for orchestrator chains
        if ctx:
            try:
                ctx = _compact_ctx(ctx)
            except Exception:
                pass
        await _coop()

        # Continuity: persist a compact state frame via embeddings for next turns
        try:
            guid = ""
            state_frame = build_state_frame(
                user_text=(x or ""),
                synth=synth or "",
                anchors=anchors if 'anchors' in locals() else None,
                guidance=guid,
                cont_block=cont_block,
                error_summary=(err.strip() if err and isinstance(err, str) else ""),
            )
            if state_frame and state_frame.strip():
                import hashlib as _hashlib
                from jinx.micro.conversation.cont import load_cache_meta as _load_meta, save_last_context_with_meta as _save_meta
                sha = _hashlib.sha256(state_frame.encode("utf-8", errors="ignore")).hexdigest()
                try:
                    meta = await _load_meta()
                except Exception:
                    meta = {}
                if (meta.get("frame_sha") or "") != sha:
                    await embed_text(state_frame, source="state", kind="frame")
                    try:
                        await _save_meta(proj_ctx or "", anchors if 'anchors' in locals() else None, frame_sha=sha)
                    except Exception:
                        pass
            try:
                await _compact_frames()
            except Exception:
                pass
        except Exception:
            pass
        # 2) <memory> from file-based view (active.md or active.compact.md). Default ON.
        try:
            is_followup = _is_short(x or "")
        except Exception:
            is_followup = False
        try:
            mem_text = ""
            mem_text = await _build_api_mem(is_followup, topic_shifted)
        except Exception:
            mem_text = ""
        # 2.5) <evergreen> persistent durable facts
        # Default: do NOT include evergreen content in the LLM payload.
        # If explicitly enabled via JINX_EVERGREEN_SEND=1, include a compact selection.
        evergreen_text = ""
        try:
            send_evg = False
            if send_evg:
                q_for_evg = _q if '_q' in locals() else (x or "")
                evergreen_text = await _select_evg(q_for_evg, anchors=anchors if 'anchors' in locals() else None)
        except Exception:
            evergreen_text = ""
        # Continuity: optionally gate evergreen (when sending) by topic shift on short follow-ups
        if evergreen_text:
            try:
                if _is_short(x or ""):
                    try:
                        shifted = await _topic_shift(_q)
                    except Exception:
                        shifted = False
                    topic_shifted = topic_shifted or bool(shifted)
                    if shifted:
                        evergreen_text = ""
            except Exception:
                pass
        # Optional: persist memory snapshot as Markdown for project embeddings ingestion
        try:
            await persist_memory(mem_text, evergreen_text, user_text=(x or ""), plan_goal="")
        except Exception:
            pass
        # 3) <task> reflects the immediate objective: when handling an error,
        #    avoid copying traceback or transcript into <task>.
        #    Continuity augmentation disabled: use only the current user input.
        if err and err.strip():
            task_text = ""
        else:
            task_text = (x or "").strip()
        # Optional <error> block carries execution or prior error details
        error_text = (err.strip() if err and err.strip() else None)

        # Assemble header using shared formatting utilities
        header_text = build_header(ctx, mem_text, task_text, error_text, evergreen_text)
        # Unified brain: fuse <embeddings_*> + <memory> + <task> (+evergreen) into a single guidance block
        try:
            unified_brain = _build_unified_brain(ctx or "", mem_text or "", task_text or "", evergreen_text or "")
        except Exception:
            unified_brain = ""
        if unified_brain:
            header_text = (header_text + "\n\n" + unified_brain) if header_text else unified_brain
        if header_text:
            chains = header_text + ("\n\n" + chains if chains else "")
        # Optional: lightweight telemetry about context block sizes
        try:
            pass
        except Exception:
            pass
        # Continuity dev echo (optional): tiny trace line for observability
        try:
            pass
        except Exception:
            pass
        # If an error is present, enforce a decay hit to drive auto-fix loop
        if err and err.strip():
            decay = max(decay, 50)
        if decay:
            await dec_pulse(decay)
        # Final normalization guard
        chains = ensure_header_block_separation(chains)
        
        # === JINX IDENTITY GUARANTEE ===
        # Jinx MUST always know who she is. Identity is non-negotiable.
        try:
            from jinx.micro.runtime.prompt_constructor import ensure_jinx_identity
            chains = ensure_jinx_identity(chains)
        except Exception:
            pass
        
        # === MULTI-TASK PROMPT INJECTION ===
        # If there are pending tasks in queue, inject multi-task context
        try:
            from jinx.micro.runtime.task_ctx import get_current_group
            from jinx.micro.runtime.prompt_constructor import inject_multi_task_context
            
            # Check for pending tasks in current group
            pending_count = 0
            pending_texts = []
            try:
                from jinx.micro.runtime.frame_shift import get_pending_tasks_for_group
                gid = get_current_group()
                pending_texts = get_pending_tasks_for_group(gid, limit=5)
                pending_count = len(pending_texts)
            except Exception:
                pass
            
            # Inject multi-task handling if multiple tasks pending
            if pending_count > 0:
                all_tasks = [x] + pending_texts[:4]  # Current + up to 4 pending
                chains = inject_multi_task_context(chains, len(all_tasks), all_tasks)
        except Exception:
            pass
        
        await debug_log(f"Final chains prepared, len={len(chains)}", "SHATTER")
        # Use a dedicated recovery prompt only when fixing an error; otherwise default prompt
        prompt_override = "burning_logic_recovery" if (err and err.strip()) else None
        # Streaming fast-path (env-gated): early-run on first complete code block
        executed_early: bool = False
        printed_tail_early: bool = False
        stream_on = True
        await debug_log("Ready to call LLM", "SHATTER")

        # Early execution callback (receives code body and code_id)
        async def _early_exec(body: str, cid: str) -> None:
            nonlocal executed_early
            nonlocal printed_tail_early
            if executed_early:
                return
            # Heuristic guard: skip early run when the first complete block is not code-like
            # (e.g., model emitted <python_question_...> or prose instead of executable code)
            try:
                if not _is_code_like(body or ""):
                    return
            except Exception:
                # Fail-closed: if heuristic unavailable, do not early-execute
                return
            minimal = f"<python_{cid}>\n{body}\n</python_{cid}>"
            async def _early_err(e: Optional[str]) -> None:
                if not e:
                    return
                _record_event("exec.error", {"turn_id": turn_id or "", "phase": "early", "error": e, "code_id": cid}, group=gid, weight=6.0)
                try:
                    # Label the preview so it doesn't look like a duplicate answer box
                    await bomb_log(f"[early exec error] code_id={cid} turn_id={turn_id or '-'}")
                    pretty_echo(minimal)
                    await show_sandbox_tail()
                except Exception:
                    pass
                # Attach the executed code to the error payload so recovery sees the code to fix
                payload = _attach_error_code(e or "", None, cid, code_body=body)
                try:
                    await corrupt_report(payload)
                except Exception:
                    pass
            try:
                ok = await run_blocks(minimal, cid, _early_err)
                if ok:
                    executed_early = True
                    await show_sandbox_tail()
                    printed_tail_early = True
                    try:
                        await embed_text(minimal.strip(), source="dialogue", kind="agent")
                    except Exception:
                        pass
            except Exception:
                pass

        _act("calling LLM")
        await debug_log("Calling LLM...", "SHATTER")
        out, code_id = await _phase_llm(chains, prompt_override=prompt_override, stream_on=stream_on, on_first_block=_early_exec)
        await debug_log(f"LLM returned {len(out)} chars, code_id={code_id}", "SHATTER")
        _record_event("llm.output", {"turn_id": turn_id or "", "code_id": code_id, "chars": len(out or ""), "preview": (out or "")[:320]}, group=gid, weight=3.5)
        await _coop()
        _act("normalizing model output")
        # Normalize model output to ensure exactly one <python_{code_id}> block and proper fences
        try:
            out = normalize_output_blocks(out, code_id)
        except Exception:
            pass
        # Always show the model output box for the current turn (once)
        printed_out_box: bool = False
        try:
            await _pretty_echo_async(out)
            printed_out_box = True
        except Exception:
            try:
                # Fallback to sync printing in a thread
                import asyncio as _aio
                await _aio.to_thread(pretty_echo, out)
                printed_out_box = True
            except Exception:
                pass

        # Ensure that on any execution error we also show the raw model output
        async def on_exec_error(err_msg: Optional[str]) -> None:
            # Sandbox callback sends None on success — ignore to avoid duplicate log prints
            if not err_msg:
                return
            _record_event("exec.error", {"turn_id": turn_id or "", "phase": "main", "error": err_msg, "code_id": code_id}, group=gid, weight=6.0)
            # Avoid re-printing the same model box; it's already shown above
            await show_sandbox_tail()
            # Attach the executed code to the error payload so recovery sees the code to fix
            payload = _attach_error_code(err_msg or "", out, code_id)
            await corrupt_report(payload)

        # If early executed successfully, treat as executed to prevent duplicate run/print
        _act("executing code blocks")
        await debug_log(f"Executing... (early_executed={executed_early})", "SHATTER")
        executed = True if executed_early else await _phase_exec(out, code_id, on_exec_error)
        await debug_log(f"Execution complete (executed={executed})", "SHATTER")
        _record_event("exec.done", {"turn_id": turn_id or "", "code_id": code_id, "executed": bool(executed), "early": bool(executed_early)}, group=gid, weight=2.5)
        if not executed:
            try:
                await asyncio.wait_for(bomb_log(f"No executable <python_{code_id}> block found in model output; displaying raw output."), timeout=0.25)
            except Exception:
                try:
                    asyncio.create_task(bomb_log(f"No executable <python_{code_id}> block found in model output; displaying raw output."))
                except Exception:
                    pass
            # Already printed above
            await dec_pulse(10)
            # Log a clean Jinx line (prefer question content); avoid raw tags
            try:
                pairs = parse_tagged_blocks(out, code_id)
            except Exception:
                pairs = []
            qtext = ""
            for tag, core in pairs:
                if tag.startswith("python_question_"):
                    qtext = (core or "").strip()
                    break
            if not qtext:
                try:
                    txt = out or ""
                    txt = re.sub(r"<[^>]+>.*?</[^>]+>", "", txt, flags=re.DOTALL)
                    txt = re.sub(r"<[^>]+>", "", txt)
                    qtext = txt.strip()
                except Exception:
                    qtext = (out or "").strip()
            if qtext:
                await blast_mem(f"Jinx: {qtext}")
            # Append turn to file-based memory (best-effort)
            try:
                await _append_turn((x or ""), (out or ""))
            except Exception:
                pass
            # Update rolling group summary with a text-only agent line
            try:
                await _append_group_summary((x or ""), qtext or "")
            except Exception:
                pass
            # Ingest compact memory signal for this turn (group-affine via ContextVar)
            try:
                if qtext:
                    import asyncio as _aio
                    _aio.create_task(embed_text(qtext, source="state", kind="mem"))
            except Exception:
                pass
        else:
            # After successful execution, also surface the latest sandbox log context (avoid duplicate if already printed early)
            if not printed_tail_early:
                await show_sandbox_tail()
            # Also embed the agent output for retrieval (source: dialogue)
            try:
                await embed_text(out.strip(), source="dialogue", kind="agent")
            except Exception:
                pass
            # Append turn to file-based memory (best-effort)
            try:
                # Do not persist a literal '<no_response>' as the user line
                ux = (x or "").strip()
                await _append_turn((ux if ux != "<no_response>" else ""), (out or ""))
            except Exception:
                pass
            # Update rolling group summary using the full model output (textual)
            try:
                await _append_group_summary((x or ""), (out or ""))
            except Exception:
                pass
            # Ingest compact memory signal for this executed turn
            try:
                if out:
                    import asyncio as _aio
                    # Prefer a trimmed version if too long
                    _aio.create_task(embed_text((out or "")[:512], source="state", kind="mem"))
            except Exception:
                pass
            # Mark last agent reply time (use loop.time to match input watchdog clock)
            try:
                import asyncio as _aio
                import jinx.state as _jx_state
                _jx_state.last_agent_reply_ts = float(_aio.get_running_loop().time())
            except Exception:
                pass
        # Non-executed branch: mark last agent reply as well to honor TIMEOUT logic
        if not executed:
            try:
                import asyncio as _aio
                import jinx.state as _jx_state
                _jx_state.last_agent_reply_ts = float(_aio.get_running_loop().time())
            except Exception:
                pass
    except Exception as e:
        await debug_log(f"EXCEPTION: {e}", "SHATTER")
        tb_str = traceback.format_exc()
        await bomb_log(tb_str)
        
        # Attempt self-healing
        try:
            from jinx.micro.runtime.self_healing import auto_heal_error
            
            healed = await auto_heal_error(
                type(e).__name__,
                str(e),
                tb_str
            )
            
            if healed:
                await debug_log("Self-healing successful! Continuing...", "SHATTER")
                # Don't decay pulse if healed
                return
        except Exception:
            pass
        
        await dec_pulse(50)
    finally:
        await debug_log("Finally block - cleaning up", "SHATTER")
        _act_clear()
        
        # === RECORD OPERATION RESULT FOR LEARNING ===
        try:
            from jinx.micro.runtime.dynamic_config_plugin import record_request_result
            
            request_latency = (time.time() - request_start_time) * 1000  # Convert to ms
            request_success = True  # If we reached finally without unhandled exception, it's success
            
            await record_request_result(request_success, request_latency, x)
        except Exception:
            pass
        
        # === AUTOBRAIN LEARNING FEEDBACK ===
        try:
            elapsed_ms = (time.time() - request_start_time) * 1000
            _ab_record("turn_timeout_sec", True, elapsed_ms)
            _ab_record("stage_projctx_ms", True, elapsed_ms)
        except Exception:
            pass
        
        # === BRAIN TASK COMPLETION ===
        try:
            if _brain_task_id:
                _brain.complete_task(success=True, result=f"completed in {(time.time() - request_start_time):.1f}s")
        except Exception:
            pass
        
        # Run memory optimization after each model interaction using a per-turn snapshot (bounded, skip on shutdown)
        try:
            import jinx.state as _jx_state
            if not _jx_state.shutdown_event.is_set():
                snap = await glitch_pulse()
                from jinx.micro.memory.optimizer import submit as _opt_submit
                _ = await _bounded_run(_opt_submit(snap), 800)
        except Exception:
            pass
        await debug_log("END - function complete", "SHATTER")
