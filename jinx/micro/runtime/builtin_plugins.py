from __future__ import annotations

import asyncio
from typing import Any

from jinx.micro.runtime.plugins import register_plugin, subscribe_event
from jinx.micro.runtime.resource_locator_plugin import (
    register_resource_locator_plugin as _register_resource_locator_plugin,
)
from jinx.micro.runtime.event_synthesis_plugin import (
    register_event_synthesis_plugin as _register_event_synthesis_plugin,
)
from jinx.logger.file_logger import append_line as _append
from jinx.log_paths import BLUE_WHISPERS


def register_builtin_plugins() -> None:
    """Register a set of first-party plugins with modern, RT-aware behavior.

    Plugins are off by default unless enabled via JINX_PLUGINS or future toggles.
    - telemetry: lightweight metrics trace to BLUE_WHISPERS (env: 'telemetry:on').
    """

    async def _telemetry_start(ctx) -> None:  # type: ignore[no-redef]
        # local semaphore to avoid log storm
        sem = asyncio.Semaphore(4)

        async def _log(topic: str, payload: Any) -> None:
            async with sem:
                try:
                    await _append(BLUE_WHISPERS, f"[telemetry] {topic} {payload}")
                except Exception:
                    pass

        subscribe_event("queue.intake", plugin="telemetry", callback=_log)
        subscribe_event("turn.scheduled", plugin="telemetry", callback=_log)
        subscribe_event("turn.finished", plugin="telemetry", callback=_log)
        subscribe_event("turn.metrics", plugin="telemetry", callback=_log)
        subscribe_event("spinner.start", plugin="telemetry", callback=_log)
        subscribe_event("spinner.stop", plugin="telemetry", callback=_log)

    async def _telemetry_stop(ctx) -> None:  # type: ignore[no-redef]
        # No unsubscribe API yet; bus entries will be GC'ed on process exit.
        # Stop hook kept for symmetry and future resource cleanup.
        return None

    register_plugin(
        "telemetry",
        start=_telemetry_start,
        stop=_telemetry_stop,
        enabled=False,  # opt-in via JINX_PLUGINS="telemetry:on"
        priority=50,
        version="1.0.0",
        deps=[],
        features={"telemetry"},
    )

    # Autonomous prefetcher: warms project/dialogue contexts on intake for zero-latency turns
    async def _prefetch_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        from typing import Any

        sem = asyncio.Semaphore(2)

        async def _do_prefetch(q: str) -> None:
            if not (q or "").strip():
                return
            async with sem:
                try:
                    from jinx.micro.runtime.prefetch_broker import prefetch_project_ctx, prefetch_base_ctx
                except Exception:
                    return
                proj_ms = 260
                base_ms = 120
                await asyncio.gather(
                    prefetch_project_ctx(q, max_time_ms=proj_ms),
                    prefetch_base_ctx(q, max_time_ms=base_ms),
                )

        async def _on_intake(_topic: str, payload: Any) -> None:
            try:
                q = str((payload or {}).get("text") or "").strip()
            except Exception:
                q = ""
            if not q:
                return
            try:
                # Small debounce: yield to allow grouping of burst inputs
                await asyncio.sleep(0)
            except Exception:
                pass
            asyncio.create_task(_do_prefetch(q))

        # Subscribe to queue intake
        subscribe_event("queue.intake", plugin="prefetch", callback=_on_intake)

    async def _prefetch_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "prefetch",
        start=_prefetch_start,
        stop=_prefetch_stop,
        enabled=True,  # autonomous by default
        priority=40,
        version="1.0.0",
        deps=[],
        features={"prefetch"},
    )

    _register_event_synthesis_plugin()

    # AutoBrain: Intelligent autonomous configuration system
    async def _autobrain_start(ctx) -> None:  # type: ignore[no-redef]
        try:
            from jinx.micro.runtime.autobrain_config import start_optimization_task, self_repair_check
            # Run initial self-repair check
            repairs = self_repair_check()
            if repairs:
                try:
                    from jinx.micro.ui.output import pretty_echo_async
                    await pretty_echo_async(
                        f"<autobrain_init>\nSelf-repair: {len(repairs)} fixes applied\n</autobrain_init>",
                        title="AutoBrain"
                    )
                except Exception:
                    pass
            # Start background optimization loop
            start_optimization_task()
        except Exception:
            pass

    async def _autobrain_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "autobrain",
        start=_autobrain_start,
        stop=_autobrain_stop,
        enabled=True,  # always enabled - core autonomous intelligence
        priority=5,  # high priority - start early
        version="1.0.0",
        deps=[],
        features={"autobrain", "autonomous", "self-tuning"},
    )

    # Architectural Memory: persistent task tracking and context
    async def _arch_memory_start(ctx) -> None:  # type: ignore[no-redef]
        try:
            from jinx.micro.runtime.arch_memory import (
                get_memory_summary,
                cleanup_old_tasks,
                get_pending_tasks,
            )
            # Initial cleanup of old tasks
            removed = await cleanup_old_tasks(max_age_hours=48.0)
            
            # Log summary
            summary = get_memory_summary()
            pending = get_pending_tasks()
            
            if pending or summary.get("tasks", {}).get("in_progress", 0) > 0:
                try:
                    from jinx.micro.ui.output import pretty_echo_async
                    await pretty_echo_async(
                        f"<arch_memory>\nResuming: {len(pending)} pending tasks, "
                        f"{summary.get('tasks', {}).get('in_progress', 0)} in progress\n</arch_memory>",
                        title="ArchMemory"
                    )
                except Exception:
                    pass
            
            # Start periodic cleanup task
            import asyncio
            async def _periodic_cleanup():
                while True:
                    await asyncio.sleep(3600)  # Every hour
                    try:
                        await cleanup_old_tasks(max_age_hours=24.0)
                    except Exception:
                        pass
            
            asyncio.create_task(_periodic_cleanup(), name="arch-memory-cleanup")
        except Exception:
            pass

    async def _arch_memory_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "arch_memory",
        start=_arch_memory_start,
        stop=_arch_memory_stop,
        enabled=True,  # always enabled - core memory system
        priority=6,  # after autobrain
        version="1.0.0",
        deps=["autobrain"],
        features={"arch_memory", "task_tracking", "context_persistence"},
    )

    # Self-Evolution Engine: goal-driven self-improvement
    async def _evolution_start(ctx) -> None:  # type: ignore[no-redef]
        try:
            from jinx.micro.runtime.self_evolution import (
                initialize_system_goals,
                run_evolution_cycle,
                get_evolution_summary,
            )
            
            # Initialize system goals (self-improvement, user success, etc.)
            initialize_system_goals()
            
            # Start periodic evolution cycle (every 10 minutes, minimal LLM usage)
            import asyncio
            async def _evolution_loop():
                while True:
                    await asyncio.sleep(600)  # 10 minutes
                    try:
                        results = await run_evolution_cycle()
                        
                        # Apply learning confidence decay
                        try:
                            from jinx.micro.runtime.self_evolution import (
                                decay_learning_confidence,
                                prune_low_confidence_learnings,
                                auto_create_suggested_goals,
                            )
                            decayed = decay_learning_confidence()
                            pruned = prune_low_confidence_learnings()
                            new_goals = auto_create_suggested_goals()
                        except Exception:
                            decayed = pruned = 0
                            new_goals = []
                        
                        if results.get("llm_called") or new_goals:
                            try:
                                from jinx.micro.ui.output import pretty_echo_async
                                msg = f"<evolution>\nAnalyzed {results.get('analyzed_goals', 0)} goals"
                                if new_goals:
                                    msg += f", suggested {len(new_goals)} new goals"
                                if decayed:
                                    msg += f", decayed {decayed} learnings"
                                msg += "\n</evolution>"
                                await pretty_echo_async(msg, title="Evolution")
                            except Exception:
                                pass
                    except Exception:
                        pass
            
            asyncio.create_task(_evolution_loop(), name="self-evolution")
            
            # Log brain status on startup
            try:
                from jinx.micro.runtime.brain import brain_check
                from jinx.micro.ui.output import pretty_echo_async
                status_line = brain_check()
                await pretty_echo_async(f"<brain_init>\n{status_line}\n</brain_init>", title="Brain")
            except Exception:
                pass
                    
        except Exception:
            pass

    async def _evolution_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "self_evolution",
        start=_evolution_start,
        stop=_evolution_stop,
        enabled=True,  # always enabled - core self-improvement
        priority=7,  # after arch_memory
        version="1.0.0",
        deps=["arch_memory"],
        features={"evolution", "self_improvement", "goal_driven"},
    )

    # Brain Health Monitor: periodic health check and auto-optimization
    async def _brain_health_start(ctx) -> None:  # type: ignore[no-redef]
        try:
            from jinx.micro.runtime.brain import get_brain
            import asyncio
            
            brain = get_brain()
            
            async def _health_loop():
                while True:
                    await asyncio.sleep(300)  # Every 5 minutes
                    try:
                        status = brain.status()
                        
                        # Auto-optimize if health degraded
                        if status.health == "degraded":
                            try:
                                from jinx.micro.runtime.autobrain_config import self_repair_check
                                repairs = self_repair_check()
                                if repairs:
                                    from jinx.micro.ui.output import pretty_echo_async
                                    await pretty_echo_async(
                                        f"<brain_health>\nAuto-repair: {len(repairs)} fixes\n</brain_health>",
                                        title="Brain"
                                    )
                            except Exception:
                                pass
                        
                        # Trigger evolution if many failures
                        if status.autobrain_success_rate < 0.7:
                            try:
                                await brain.analyze_and_improve()
                            except Exception:
                                pass
                                
                    except Exception:
                        pass
            
            asyncio.create_task(_health_loop(), name="brain-health-monitor")
            
        except Exception:
            pass

    async def _brain_health_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "brain_health",
        start=_brain_health_start,
        stop=_brain_health_stop,
        enabled=True,
        priority=8,  # after self_evolution
        version="1.0.0",
        deps=["self_evolution"],
        features={"health_monitor", "auto_optimize"},
    )

    # Brain Metrics: periodic metrics collection
    async def _metrics_start(ctx) -> None:  # type: ignore[no-redef]
        try:
            from jinx.micro.runtime.brain_metrics import take_snapshot
            import asyncio
            
            async def _metrics_loop():
                while True:
                    await asyncio.sleep(600)  # Every 10 minutes
                    try:
                        take_snapshot()
                    except Exception:
                        pass
            
            asyncio.create_task(_metrics_loop(), name="brain-metrics")
        except Exception:
            pass

    async def _metrics_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "brain_metrics",
        start=_metrics_start,
        stop=_metrics_stop,
        enabled=True,
        priority=9,
        version="1.0.0",
        deps=["brain_health"],
        features={"metrics", "analytics"},
    )

    # Autonomous Monitor: continuous issue detection, auto-repair, smart rollback
    async def _autonomous_monitor_start(ctx) -> None:  # type: ignore[no-redef]
        try:
            from jinx.micro.runtime.autonomous_monitor import run_monitoring_cycle, get_issue_summary
            from jinx.micro.runtime.smart_rollback import take_snapshot, SnapshotType
            import asyncio
            
            # Take initial snapshot
            take_snapshot(SnapshotType.AUTO, "System startup", "startup")
            
            async def _monitor_loop():
                while True:
                    await asyncio.sleep(120)  # Every 2 minutes
                    try:
                        # Take periodic snapshot
                        take_snapshot(SnapshotType.AUTO, "Periodic snapshot", "periodic")
                        
                        # Run monitoring cycle
                        results = await run_monitoring_cycle()
                        
                        # Report if issues found or repairs made
                        if results.get("issues_detected", 0) > 0 or results.get("repairs_attempted", 0) > 0:
                            try:
                                from jinx.micro.ui.output import pretty_echo_async
                                summary = get_issue_summary()
                                msg = f"<monitor>\nActive issues: {summary['active_issues']}"
                                if results.get("repairs_successful", 0) > 0:
                                    msg += f" | Repairs: {results['repairs_successful']}/{results['repairs_attempted']}"
                                msg += "\n</monitor>"
                                await pretty_echo_async(msg, title="Monitor")
                            except Exception:
                                pass
                    except Exception:
                        pass
            
            asyncio.create_task(_monitor_loop(), name="autonomous-monitor")
            
        except Exception:
            pass

    async def _autonomous_monitor_stop(ctx) -> None:  # type: ignore[no-redef]
        # Take final snapshot before shutdown
        try:
            from jinx.micro.runtime.smart_rollback import take_snapshot, SnapshotType
            take_snapshot(SnapshotType.CHECKPOINT, "Pre-shutdown checkpoint", "shutdown")
        except Exception:
            pass

    register_plugin(
        "autonomous_monitor",
        start=_autonomous_monitor_start,
        stop=_autonomous_monitor_stop,
        enabled=True,
        priority=10,  # after brain_metrics
        version="1.0.0",
        deps=["brain_metrics"],
        features={"monitoring", "auto_repair", "rollback"},
    )

    # Cognitive seeds: maintain short-term salient tokens from recent inputs for query expansion
    async def _cog_start(ctx) -> None:  # type: ignore[no-redef]
        import time
        import re
        import jinx.state as jx_state
        from typing import Any

        TOK = re.compile(r"(?u)[\w\.]{3,}")
        ttl = 30.0

        def _update(text: str) -> None:
            s = (text or "").strip()
            if not s:
                return
            seen = set()
            out = []
            for m in TOK.finditer(s):
                t = (m.group(0) or "").strip().lower()
                if len(t) >= 3 and t not in seen:
                    seen.add(t)
                    out.append(t)
                if len(out) >= 12:
                    break
            try:
                jx_state.cog_seeds_terms = out
                jx_state.cog_seeds_ts = float(time.perf_counter())
                jx_state.cog_seeds_ttl = float(ttl)
            except Exception:
                pass

        async def _on_intake(_topic: str, payload: Any) -> None:
            try:
                text = str((payload or {}).get("text") or "")
            except Exception:
                text = ""
            if text:
                _update(text)

        async def _on_finished(_topic: str, payload: Any) -> None:
            # On turn end, expire if TTL elapsed
            try:
                ts = float(getattr(jx_state, "cog_seeds_ts", 0.0) or 0.0)
                tll = float(getattr(jx_state, "cog_seeds_ttl", ttl) or ttl)
            except Exception:
                ts = 0.0; tll = ttl
            if ts > 0.0:
                try:
                    if (time.perf_counter() - ts) >= tll:
                        jx_state.cog_seeds_terms = []
                        jx_state.cog_seeds_ts = 0.0
                except Exception:
                    pass

        subscribe_event("queue.intake", plugin="cog", callback=_on_intake)
        subscribe_event("turn.finished", plugin="cog", callback=_on_finished)

    async def _cog_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "cog",
        start=_cog_start,
        stop=_cog_stop,
        enabled=True,  # autonomous by default
        priority=30,
        version="1.0.0",
        deps=[],
        features={"cog"},
    )

    # Autodiscovery: scan micro-packages for auto_init/auto and run auto_start/auto_stop
    async def _auto_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        import importlib
        from pathlib import Path
        from typing import Any

        base_dir = str(Path(__file__).resolve().parent.parent)  # jinx/micro
        conc = 3
        start_ms = 400
        sem = asyncio.Semaphore(max(1, conc))

        # track for stop
        started: list[tuple[str, Any]] = []

        async def _call_start(mod) -> None:
            try:
                fn = getattr(mod, "auto_start", None)
                if not fn:
                    return
                res = fn(ctx) if fn.__code__.co_argcount >= 1 else fn()  # type: ignore[attr-defined]
                if asyncio.iscoroutine(res):
                    await asyncio.wait_for(res, timeout=start_ms / 1000.0)  # type: ignore[arg-type]
            except Exception:
                return

        async def _one(pkg: str) -> None:
            async with sem:
                for suffix in ("auto_init", "auto"):
                    modname = f"jinx.micro.{pkg}.{suffix}"
                    try:
                        mod = importlib.import_module(modname)
                    except Exception:
                        continue
                    await _call_start(mod)
                    try:
                        started.append((modname, getattr(mod, "auto_stop", None)))
                    except Exception:
                        started.append((modname, None))
                    break

        # schedule tasks for subpackages
        tasks: list[asyncio.Task] = []
        try:
            for name in os.listdir(base_dir):
                if not name or name.startswith("_"):
                    continue
                p = os.path.join(base_dir, name)
                if os.path.isdir(p):
                    tasks.append(asyncio.create_task(_one(name)))
        except Exception:
            tasks = tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Also execute any autoapi-registered functions (opt-in decorators)
        try:
            from jinx.micro.runtime.autoapi import AUTO_START_FUNCS as _ASF
        except Exception:
            _ASF = []  # type: ignore[assignment]
        if _ASF:
            for fn in list(_ASF):
                try:
                    res = fn(ctx) if fn.__code__.co_argcount >= 1 else fn()  # type: ignore[attr-defined]
                    if asyncio.iscoroutine(res):
                        await asyncio.wait_for(res, timeout=start_ms / 1000.0)  # type: ignore[arg-type]
                except Exception:
                    continue

        # Store list for stop
        try:
            setattr(ctx, "_autodisc_started", started)
        except Exception:
            pass

    async def _auto_stop(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        try:
            started = list(getattr(ctx, "_autodisc_started", []) or [])
        except Exception:
            started = []
        if not started:
            pass
        stop_ms = 300
        tasks: list[asyncio.Task] = []
        for _modname, fn in started[::-1]:
            if not fn:
                continue
            try:
                res = fn(ctx) if fn.__code__.co_argcount >= 1 else fn()  # type: ignore[attr-defined]
                if asyncio.iscoroutine(res):
                    tasks.append(asyncio.create_task(asyncio.wait_for(res, timeout=stop_ms / 1000.0)))  # type: ignore[arg-type]
            except Exception:
                continue
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Also execute autoapi stop functions
        try:
            from jinx.micro.runtime.autoapi import AUTO_STOP_FUNCS as _ZSF
        except Exception:
            _ZSF = []  # type: ignore[assignment]
        if _ZSF:
            ztasks: list[asyncio.Task] = []
            for fn in list(_ZSF):
                try:
                    res = fn(ctx) if fn.__code__.co_argcount >= 1 else fn()  # type: ignore[attr-defined]
                    if asyncio.iscoroutine(res):
                        ztasks.append(asyncio.create_task(asyncio.wait_for(res, timeout=stop_ms / 1000.0)))  # type: ignore[arg-type]
                except Exception:
                    continue
            if ztasks:
                await asyncio.gather(*ztasks, return_exceptions=True)

    register_plugin(
        "autodiscovery",
        start=_auto_start,
        stop=_auto_stop,
        enabled=True,  # autonomous by default
        priority=20,
        version="1.0.0",
        deps=[],
        features={"autodiscovery"},
    )

    # Foresight: learn token/bigram frequencies and predict likely next tokens
    async def _foresight_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        import time
        from typing import Any, List
        import jinx.state as jx_state

        topk = 4
        ttl = 20.0

        # Lazy imports
        try:
            from jinx.micro.runtime.foresight_store import load_state, save_state, update_tokens, predict_next
        except Exception:
            return

        try:
            from jinx.micro.runtime.prefetch_broker import prefetch_project_ctx, prefetch_base_ctx
        except Exception:
            prefetch_project_ctx = None  # type: ignore
            prefetch_base_ctx = None  # type: ignore

        state = load_state()
        sem = asyncio.Semaphore(2)

        async def _prefetch_variants(q: str, terms: List[str]) -> None:
            if not prefetch_project_ctx or not prefetch_base_ctx:
                return
            async with sem:
                proj_ms = 240
                base_ms = 100
                tasks = []
                for t in terms[:max(1, topk)]:
                    qv = (q + " " + t).strip()
                    async def _one(qx: str) -> None:
                        await asyncio.gather(
                            prefetch_project_ctx(qx, max_time_ms=proj_ms),
                            prefetch_base_ctx(qx, max_time_ms=base_ms),
                        )
                    tasks.append(asyncio.create_task(_one(qv)))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

        async def _on_intake(_topic: str, payload: Any) -> None:
            q = str((payload or {}).get("text") or "").strip()
            if not q:
                return
            # Learn from current input
            update_tokens(state, q, w=1.0)
            save_state(state)
            # Seeds: prefer cognitive seeds when present
            try:
                seeds = list(getattr(jx_state, "cog_seeds_terms", []) or [])
            except Exception:
                seeds = []
            preds = predict_next(state, seeds=seeds, top_k=max(1, topk))
            try:
                jx_state.foresight_terms = preds
                jx_state.foresight_ts = float(time.perf_counter())
                jx_state.foresight_ttl = float(ttl)
            except Exception:
                pass
            # Prefetch predicted variants
            try:
                await _prefetch_variants(q, preds)
            except Exception:
                pass

        # Subscribe to queue intake
        subscribe_event("queue.intake", plugin="foresight", callback=_on_intake)

    async def _foresight_stop(ctx) -> None:  # type: ignore[no-redef]
        # No stateful tasks to cancel here; foresight_store is file-backed
        return None

    register_plugin(
        "foresight",
        start=_foresight_start,
        stop=_foresight_stop,
        enabled=True,  # autonomous by default
        priority=25,
        version="1.0.0",
        deps=[],
        features={"foresight"},
    )

    # Oracle: build a long-horizon term graph and predict next tokens via personalized PageRank
    async def _oracle_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        from typing import Any, List
        import jinx.state as jx_state

        topk = 6
        ttl = 28.0

        # Lazy imports
        try:
            from jinx.micro.runtime.oracle_graph import load_graph, save_graph, update_from_text, prune, predict_ppr
        except Exception:
            return
        try:
            from jinx.micro.embeddings.project_retrieval import build_project_context_for as _build_project
            from jinx.micro.embeddings.retrieval import build_context_for as _build_base
            from jinx.micro.embeddings.prefetch_cache import put_project, put_base
        except Exception:
            _build_project = None  # type: ignore
            _build_base = None  # type: ignore
            put_project = None  # type: ignore
            put_base = None  # type: ignore

        adj = load_graph(max_nodes=12000)
        sem = asyncio.Semaphore(2)

        async def _prefetch_variants(q: str, terms: List[str]) -> None:
            if not prefetch_project_ctx or not prefetch_base_ctx:
                return
            async with sem:
                proj_ms = 220
                base_ms = 90
                tasks = []
                for t in terms[:max(1, topk)]:
                    qv = (q + " " + t).strip()
                    async def _one(qx: str) -> None:
                        await asyncio.gather(
                            prefetch_project_ctx(qx, max_time_ms=proj_ms),
                            prefetch_base_ctx(qx, max_time_ms=base_ms),
                        )
                    tasks.append(asyncio.create_task(_one(qv)))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

        async def _on_intake(_topic: str, payload: Any) -> None:
            q = str((payload or {}).get("text") or "").strip()
            if not q:
                return
            # Update long-horizon graph and prune lightly
            update_from_text(adj, q, window=4, w=1.0)
            prune(adj, max_nodes=12000, max_deg=72)
            # Seeds: combine cognitive + foresight
            try:
                seeds = list(getattr(jx_state, "cog_seeds_terms", []) or []) + list(getattr(jx_state, "foresight_terms", []) or [])
            except Exception:
                seeds = []
            preds = predict_ppr(adj, seeds=seeds, top_k=max(1, topk))
            try:
                jx_state.oracle_terms = preds
                jx_state.oracle_ts = float(__import__('time').perf_counter())
                jx_state.oracle_ttl = float(ttl)
            except Exception:
                pass
            # Prefetch predicted variants
            try:
                await _prefetch_variants(q, preds)
            except Exception:
                pass

        async def _on_finished(_topic: str, payload: Any) -> None:
            # Persist graph
            try:
                save_graph(adj)
            except Exception:
                pass

        subscribe_event("queue.intake", plugin="oracle", callback=_on_intake)
        subscribe_event("turn.finished", plugin="oracle", callback=_on_finished)

    async def _oracle_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "oracle",
        start=_oracle_start,
        stop=_oracle_stop,
        enabled=True,  # autonomous by default
        priority=23,
        version="1.0.0",
        deps=[],
        features={"oracle"},
    )

    # Hypersigil: variable-order n-gram model for next-token and short sequence predictions
    async def _hypersigil_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        import time
        from typing import Any, List
        import jinx.state as jx_state

        topk = 5
        seq_len = 2
        ttl = 24.0

        try:
            from jinx.micro.runtime.hypersigil_store import load_store, save_store, update_ngrams, predict_next_tokens, predict_next_sequences
        except Exception:
            return
        try:
            from jinx.micro.embeddings.project_retrieval import build_project_context_for as _build_project
            from jinx.micro.embeddings.retrieval import build_context_for as _build_base
            from jinx.micro.embeddings.prefetch_cache import put_project, put_base
        except Exception:
            _build_project = None  # type: ignore
            _build_base = None  # type: ignore
            put_project = None  # type: ignore
            put_base = None  # type: ignore

        ng = load_store(max_keys=20000)
        sem = asyncio.Semaphore(2)

        async def _prefetch_variants(q: str, terms: List[str], seqs: List[List[str]]) -> None:
            if not prefetch_project_ctx or not prefetch_base_ctx:
                return
            async with sem:
                proj_ms = 220
                base_ms = 90
                tasks = []
                # single-token variants
                for t in terms[:max(1, topk)]:
                    qv = (q + " " + t).strip()
                    async def _one_tok(qx: str) -> None:
                        await asyncio.gather(
                            prefetch_project_ctx(qx, max_time_ms=proj_ms),
                            prefetch_base_ctx(qx, max_time_ms=base_ms),
                        )
                    tasks.append(asyncio.create_task(_one_tok(qv)))
                # short-sequence variants
                for s in seqs[:max(1, topk)]:
                    if not s:
                        continue
                    qv = (q + " " + " ".join(s)).strip()
                    async def _one_seq(qx: str) -> None:
                        await asyncio.gather(
                            prefetch_project_ctx(qx, max_time_ms=proj_ms),
                            prefetch_base_ctx(qx, max_time_ms=base_ms),
                        )
                    tasks.append(asyncio.create_task(_one_seq(qv)))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

        async def _on_intake(_topic: str, payload: Any) -> None:
            q = str((payload or {}).get("text") or "").strip()
            if not q:
                return
            update_ngrams(ng, q, max_order=4, w=1.0)
            # Seeds preference: cog + foresight + oracle
            try:
                seeds = list(getattr(jx_state, "cog_seeds_terms", []) or []) + list(getattr(jx_state, "foresight_terms", []) or []) + list(getattr(jx_state, "oracle_terms", []) or [])
            except Exception:
                seeds = []
            terms = predict_next_tokens(ng, seeds, top_k=max(1, topk))
            seqs = predict_next_sequences(ng, seeds, seq_len=max(1, seq_len), top_k=max(1, topk))
            try:
                jx_state.hsigil_terms = terms
                jx_state.hsigil_seqs = seqs
                jx_state.hsigil_ts = float(time.perf_counter())
                jx_state.hsigil_ttl = float(ttl)
            except Exception:
                pass
            try:
                await _prefetch_variants(q, terms, seqs)
            except Exception:
                pass

        async def _on_finished(_topic: str, payload: Any) -> None:
            try:
                save_store(ng)
            except Exception:
                pass

        subscribe_event("queue.intake", plugin="hypersigil", callback=_on_intake)
        subscribe_event("turn.finished", plugin="hypersigil", callback=_on_finished)

    async def _hypersigil_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "hypersigil",
        start=_hypersigil_start,
        stop=_hypersigil_stop,
        enabled=True,  # autonomous by default
        priority=22,
        version="1.0.0",
        deps=[],
        features={"hypersigil"},
    )

    # Embed Observer: eagerly embed user intake text for global embeddings coverage
    async def _embedobs_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        from typing import Any
        try:
            from jinx.micro.embeddings.pipeline import embed_text as _embed_text
        except Exception:
            _embed_text = None  # type: ignore

        sem = asyncio.Semaphore(2)

        async def _on_intake(_topic: str, payload: Any) -> None:
            if _embed_text is None:
                return
            try:
                text = str((payload or {}).get("text") or "").strip()
            except Exception:
                text = ""
            if not text:
                return
            async with sem:
                try:
                    # Best-effort, background embedding for dialogue-user source
                    await _embed_text(text, source="dialogue", kind="user")
                except Exception:
                    pass

        subscribe_event("queue.intake", plugin="embed_observer", callback=_on_intake)

    async def _embedobs_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "embed_observer",
        start=_embedobs_start,
        stop=_embedobs_stop,
        enabled=True,
        priority=45,
        version="1.0.0",
        deps=[],
        features={"embeddings"},
    )

    # Locator semantics: embed-based classification of intake messages (language-agnostic)
    async def _locsem_start(ctx) -> None:  # type: ignore[no-redef]
        from typing import Any
        try:
            from jinx.micro.conversation.locator_semantics import schedule_classify as _sched
        except Exception:
            _sched = None  # type: ignore

        async def _on_intake(_topic: str, payload: Any) -> None:
            if _sched is None:
                return
            try:
                text = str((payload or {}).get("text") or "").strip()
            except Exception:
                text = ""
            if text:
                try:
                    _sched(text)
                except Exception:
                    pass

        subscribe_event("queue.intake", plugin="locator_semantics", callback=_on_intake)

    async def _locsem_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "locator_semantics",
        start=_locsem_start,
        stop=_locsem_stop,
        enabled=True,  # autonomous by default
        priority=44,
        version="1.0.0",
        deps=[],
        features={"embeddings"},
    )
    # Resource Locator: fast file auto-discovery and resolution on intake
    _register_resource_locator_plugin()

    # Arch boot: optionally submit an API architecture task at startup
    async def _arch_boot_start(ctx) -> None:  # type: ignore[no-redef]
        from pathlib import Path
        # Mission Planner is enabled by default; skip arch_boot
        return
        # Fully autonomous context derivation and spec synthesis
        try:
            from jinx.micro.llm.prompting import derive_basic_context, build_api_spec_prompt
            from jinx.micro.llm.service import spark_openai as _spark
            from jinx.micro.runtime.api import submit_task as _submit
        except Exception:
            return
        # arch_boot is disabled when Mission Planner is active (default)
        pass

    async def _arch_boot_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "arch_boot",
        start=_arch_boot_start,
        stop=_arch_boot_stop,
        enabled=True,  # enabled; only runs if env provided
        priority=15,
        version="1.0.0",
        deps=[],
        features={"arch_boot"},
    )

    # Turn counters: maintain active turn metrics for quiesce/drain
    async def _turnc_start(ctx) -> None:  # type: ignore[no-redef]
        import jinx.state as jx_state
        from typing import Any

        try:
            jx_state.active_turns = int(getattr(jx_state, "active_turns", 0) or 0)
        except Exception:
            pass

        async def _on_sched(_topic: str, payload: Any) -> None:
            try:
                jx_state.active_turns = int(getattr(jx_state, "active_turns", 0) or 0) + 1
            except Exception:
                pass

        async def _on_finish(_topic: str, payload: Any) -> None:
            try:
                cur = int(getattr(jx_state, "active_turns", 0) or 0)
                jx_state.active_turns = max(0, cur - 1)
            except Exception:
                pass

        subscribe_event("turn.scheduled", plugin="turn_counters", callback=_on_sched)
        subscribe_event("turn.finished", plugin="turn_counters", callback=_on_finish)

    async def _turnc_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "turn_counters",
        start=_turnc_start,
        stop=_turnc_stop,
        enabled=True,
        priority=10,
        version="1.0.0",
        deps=[],
        features={"metrics"},
    )

    # Reprogram intake: convert "/reprogram <goal>" or natural-language triggers into a self-reprogram request
    async def _reprog_start(ctx) -> None:  # type: ignore[no-redef]
        from typing import Any
        import re as _re

        async def _on_intake(_topic: str, payload: Any) -> None:
            try:
                text = str((payload or {}).get("text") or "").strip()
            except Exception:
                text = ""
            if not text:
                return
            low = text.lower()
            goal: str | None = None
            if low.startswith("/reprogram "):
                goal = text.split(" ", 1)[1].strip()
            else:
                # Natural-language triggers (RU/EN)
                if ("перепрограммируй" in low) or ("self-reprogram" in low) or (low.startswith("reprogram jinx")) or ("modify yourself" in low):
                    # Extract goal after trigger if possible, else use whole text
                    m = _re.search(r"(?:/reprogram|перепрограммируй|self-reprogram|reprogram jinx|modify yourself)[:\s]+(.+)", low)
                    goal = (m.group(1).strip() if m else text)
            if goal:
                try:
                    from jinx.micro.runtime.api import submit_task as _submit
                    await _submit("reprogram.request", goal=goal)
                except Exception:
                    pass

        subscribe_event("queue.intake", plugin="reprogram", callback=_on_intake)

    async def _reprog_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "reprogram",
        start=_reprog_start,
        stop=_reprog_stop,
        enabled=True,
        priority=12,
        version="1.0.0",
        deps=[],
        features={"reprogram"},
    )

    # Shadow canary: in green mode, ack *.in files with *.ok in shadow dir
    async def _shadow_start(ctx) -> None:  # type: ignore[no-redef]
        from pathlib import Path
        import asyncio
        import time

        # Shadow dir is derived from project root if present
        try:
            from jinx.bootstrap.env import ROOT
            d = str(Path(ROOT) / ".jinx_shadow")
        except Exception:
            d = ""
        if not d or not Path(d).is_dir():
            return
        period = 0.2
        sem = asyncio.Semaphore(4)

        async def _ack_one(p: str) -> None:
            async with sem:
                try:
                    if not p.endswith(".in"):
                        return
                    base = p[:-3]
                    ack = base + ".ok"
                    # Lightweight health op (bounded)
                    try:
                        import numpy as _np
                        from jinx.micro.embeddings.vector_stage_semantic import _cosine_similarity as _c
                        a = _np.array([1,0,0], dtype=_np.float32); b = _np.array([1,0,0], dtype=_np.float32)
                        _ = _c(a,b)
                    except Exception:
                        pass
                    with open(ack, "w", encoding="utf-8") as f:
                        f.write("ok")
                except Exception:
                    pass

        async def _loop() -> None:
            while True:
                try:
                    files = []
                    try:
                        dp = Path(d)
                        files = [str(dp / x.name) for x in dp.iterdir() if x.name.endswith('.in')]
                    except Exception:
                        files = []
                    tasks = [asyncio.create_task(_ack_one(p)) for p in files[:16]]
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                except Exception:
                    await asyncio.sleep(period)
                await asyncio.sleep(period)

        asyncio.create_task(_loop())

    async def _shadow_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "shadow_canary",
        start=_shadow_start,
        stop=_shadow_stop,
        enabled=True,
        priority=12,
        version="1.0.0",
        deps=[],
        features={"canary"},
    )

    # Memory Board: compressed state snapshot (board.json + board_fen.txt)
    async def _board_start(ctx) -> None:  # type: ignore[no-redef]
        import asyncio
        from jinx.micro.memory.board_state import touch_board as _touch, read_board as _read_board, maybe_embed_board as _embed_board
        from jinx.micro.runtime.plugins import subscribe_event
        from jinx.micro.runtime.api import register_prompt_macro as _reg_macro

        # Seed initial snapshot
        try:
            await _touch()
        except Exception:
            pass

        # Macro provider: <$board fen|json|field:name>
        async def _macro_board(args, ctxmacro):  # type: ignore[no-redef]
            try:
                args = list(args or [])
                mode = (args[0] if args else "json").strip().lower()
            except Exception:
                mode = "json"
            try:
                st = await _read_board()
            except Exception:
                st = {}
            if mode == "fen":
                try:
                    # Build fen on the fly from dict to avoid reading file again
                    from jinx.micro.memory.board_state import BoardState
                    bs = BoardState()
                    # assign minimal fields
                    bs.session = str(st.get("session") or "main")
                    bs.active_turns = int(st.get("active_turns") or 0)
                    bs.turns_total = int(st.get("turns_total") or 0)
                    bs.errors_total = int(st.get("errors_total") or 0)
                    bs.patches_ok = int(st.get("patches_ok") or 0)
                    bs.patches_fail = int(st.get("patches_fail") or 0)
                    bs.selfupdate_success = int(st.get("selfupdate_success") or 0)
                    bs.selfupdate_fail = int(st.get("selfupdate_fail") or 0)
                    bs.skills = set(st.get("skills") or [])
                    bs.api_intents = int(st.get("api_intents") or 0)
                    bs.api_endpoints_seen = set(st.get("api_endpoints_seen") or [])
                    return bs.fen()
                except Exception:
                    pass
            elif mode.startswith("field:"):
                key = mode.split(":", 1)[1]
                try:
                    val = st.get(key)
                    return "" if val is None else str(val)
                except Exception:
                    return ""
            else:
                try:
                    import json
                    return json.dumps(st, ensure_ascii=False)
                except Exception:
                    return "{}"

        try:
            await _reg_macro("board", _macro_board)
        except Exception:
            pass

        # Event subscriptions to maintain the board
        async def _on_intake(_topic, payload):
            try:
                q = str((payload or {}).get("text") or "")
                await _touch(last_query=q)
                await _embed_board("intake")
            except Exception:
                pass

        async def _on_turn_finished(_topic, payload):
            try:
                await _touch(turns_inc=1)
                await _embed_board("turn")
            except Exception:
                pass

        async def _on_error(_topic, payload):
            try:
                err = str((payload or {}).get("error") or "")
                await _touch(errors_inc=1, last_error=err)
                await _embed_board("error")
            except Exception:
                pass

        async def _on_patch_report(_topic, payload):
            try:
                ok = bool((payload or {}).get("success"))
                msg = str((payload or {}).get("message") or "")
                if ok:
                    await _touch(patch_ok=True, patch_msg=msg)
                else:
                    await _touch(patch_fail=True, patch_msg=msg)
                await _embed_board("patch")
            except Exception:
                pass

        async def _on_skill(_topic, payload):
            try:
                sk = str((payload or {}).get("path") or (payload or {}).get("skill") or "")
                if sk:
                    await _touch(skill_add=sk)
                    await _embed_board("skill")
            except Exception:
                pass

        async def _on_arch(_topic, payload):
            try:
                await _touch(api_intent=True)
                await _embed_board("arch")
            except Exception:
                pass

        async def _on_su_ok(_t, _p):
            try:
                await _touch(selfupdate_ok=True)
                await _embed_board("su_ok")
            except Exception:
                pass

        async def _on_su_fail(_t, _p):
            try:
                await _touch(selfupdate_fail=True)
                await _embed_board("su_fail")
            except Exception:
                pass

        subscribe_event("queue.intake", plugin="memory_board", callback=_on_intake)
        subscribe_event("turn.finished", plugin="memory_board", callback=_on_turn_finished)
        subscribe_event("turn.error", plugin="memory_board", callback=_on_error)
        subscribe_event("auto.patch.report", plugin="memory_board", callback=_on_patch_report)
        subscribe_event("auto.skill_acquired", plugin="memory_board", callback=_on_skill)
        subscribe_event("auto.arch.requested", plugin="memory_board", callback=_on_arch)
        subscribe_event("selfupdate.success", plugin="memory_board", callback=_on_su_ok)
        subscribe_event("selfupdate.preflight_failed", plugin="memory_board", callback=_on_su_fail)
        subscribe_event("selfupdate.handshake_failed", plugin="memory_board", callback=_on_su_fail)
        subscribe_event("selfupdate.green_failed", plugin="memory_board", callback=_on_su_fail)

    async def _board_stop(ctx) -> None:  # type: ignore[no-redef]
        return None

    register_plugin(
        "memory_board",
        start=_board_start,
        stop=_board_stop,
        enabled=True,
        priority=11,
        version="1.0.0",
        deps=[],
        features={"memory", "board"},
    )

__all__ = ["register_builtin_plugins"]
