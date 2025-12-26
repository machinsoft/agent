"""Async runtime loop for the Jinx agent.

``pulse_core()`` wires input, spinner, and conversation services together and
drives the primary event loop. It is intentionally lightweight, cancellation
friendly, and side-effect contained for testability.
"""

from __future__ import annotations

import asyncio
from jinx.banner_service import show_banner
import concurrent.futures as _cf
from jinx.utils import chaos_patch
from jinx.runtime import start_input_task, frame_shift as _frame_shift
from jinx.embeddings import (
    start_embeddings_task,
    start_project_embeddings_task,
    stop_embeddings_task,
    stop_project_embeddings_task,
)
import jinx.state as jx_state
import contextlib
from jinx.memory.optimizer import stop as stop_memory_optimizer, start_memory_optimizer_task
from jinx.conversation.error_worker import stop_error_worker
from jinx.settings import Settings
from jinx.supervisor import run_supervisor, SupervisedJob
from jinx.priority import start_priority_dispatcher_task
from jinx.autotune import start_autotune_task
from jinx.watchdog import start_watchdog_task
from jinx.micro.embeddings.retrieval_core import shutdown_proc_pool as _retr_pool_shutdown
from jinx.micro.net.client import prewarm_openai_client as _prewarm_openai
from jinx.micro.runtime.api import stop_selfstudy as _stop_selfstudy
from jinx.micro.runtime.api import ensure_runtime as _ensure_runtime
from jinx.micro.runtime.plugins import (
    start_plugins as _start_plugins,
    stop_plugins as _stop_plugins,
    set_plugin_context as _set_plugin_ctx,
    PluginContext as _PluginContext,
    publish_event as _publish_event,
)
from jinx.micro.runtime.builtin_plugins import register_builtin_plugins as _reg_builtin_plugins
from jinx.micro.runtime.self_update_handshake import set_online as _hs_online, set_healthy as _hs_healthy

async def pulse_core(settings: Settings | None = None) -> None:
    """Run the main asynchronous processing loop.

    The loop:
    - Shows the startup banner.
    - Starts an input task that feeds a queue with user messages.
    - Optionally dispatches through a priority-aware relay to the frame processor.
    - Displays a spinner per message while executing the conversation step.
    - Supervises background tasks with auto-restart and graceful shutdown.
    """
    show_banner()

    # Resolve settings deterministically
    cfg = settings or Settings()
    cfg.apply_to_state()
    # Start reproducibility recorder (best-effort)
    _run_id = None
    try:
        from jinx.observability.recorder import start_run_record
        _run_id = start_run_record({"settings": cfg.to_dict()})
    except Exception:
        _run_id = None
    # Minimal startup summary to stdout (no CLI required)
    try:
        print(
            f"‖ Auto-tune: prio={'on' if cfg.runtime.use_priority_queue else 'off'}, "
            f"threads={cfg.runtime.threads_max_workers}, "
            f"queue={cfg.runtime.queue_maxsize}, rt={cfg.runtime.hard_rt_budget_ms}ms"
        )
    except Exception:
        pass
    # Startup healthcheck to BLUE_WHISPERS
    try:
        from jinx.logger.file_logger import append_line as _append
        from jinx.log_paths import BLUE_WHISPERS
        ak_on = bool((cfg.openai.api_key or "").strip())
        model = (cfg.openai.model or "").strip() or "?"
        proxy = (cfg.openai.proxy or "").strip()
        conc = "2"
        prio = ("on" if cfg.runtime.use_priority_queue else "off")
        await _append(BLUE_WHISPERS, f"[health] api_key={'present' if ak_on else 'absent'} model={model} proxy={'set' if proxy else 'none'} conc={conc} prio={prio}")
    except Exception:
        pass

    # Prewarm OpenAI client synchronously (safe outside event loop busy sections)
    try:
        _prewarm_openai()
    except Exception:
        pass

    # Ensure micro-runtime bridge/self-study/repair are started before main jobs
    try:
        await _ensure_runtime()
    except Exception:
        pass

    # Mark runtime as online for self-update green handshake (no-op if not in green mode)
    try:
        _hs_online()
    except Exception:
        pass

    # Bounded queues to avoid unbounded memory growth under bursts
    q_in: asyncio.Queue[str] = asyncio.Queue(maxsize=cfg.runtime.queue_maxsize)
    q_proc: asyncio.Queue[str] = asyncio.Queue(maxsize=cfg.runtime.queue_maxsize)

    # Always route through the dispatcher so autotune can toggle priority dynamically
    q_for_frame = q_proc

    async with chaos_patch():
        # Configure default thread pool for to_thread operations
        try:
            loop = asyncio.get_running_loop()
            loop.set_default_executor(_cf.ThreadPoolExecutor(max_workers=cfg.runtime.threads_max_workers, thread_name_prefix="jinx-worker"))
        except Exception:
            pass
        # Compose supervised jobs
        job_specs: list[SupervisedJob] = [
            SupervisedJob(name="input", start=lambda: start_input_task(q_in)),
            SupervisedJob(name="frame", start=lambda: asyncio.create_task(_frame_shift(q_for_frame))),
            SupervisedJob(name="priority", start=lambda: start_priority_dispatcher_task(q_in, q_proc, cfg)),
            SupervisedJob(name="embeddings", start=lambda: start_embeddings_task()),
            SupervisedJob(name="memopt", start=lambda: start_memory_optimizer_task()),
            SupervisedJob(name="proj-embed", start=lambda: start_project_embeddings_task()),
            SupervisedJob(name="autotune", start=lambda: start_autotune_task(q_in, cfg)),
            SupervisedJob(name="watchdog", start=lambda: start_watchdog_task(cfg)),
        ]
        
        # Add self-healing and health monitoring
        try:
            from jinx.micro.runtime.health_monitor import start_health_monitoring
            
            async def _start_health():
                await start_health_monitoring()
                while not jx_state.shutdown_event.is_set():
                    await asyncio.sleep(1.0)
                return None
            
            job_specs.append(
                SupervisedJob(name="health-monitor", start=lambda: asyncio.create_task(_start_health()))
            )
        except Exception:
            pass
        
        try:
            # Start optional plugins prior to supervised
            with contextlib.suppress(Exception):
                loop = asyncio.get_running_loop()
                _set_plugin_ctx(_PluginContext(loop=loop, shutdown_event=jx_state.shutdown_event, settings=cfg, publish=_publish_event))
                _reg_builtin_plugins()
                await _start_plugins()
            # Mark runtime healthy after plugins start and before entering supervisor loop
            try:
                _hs_healthy()
            except Exception:
                pass
            # small settle delay
            try:
                await asyncio.sleep(0.05)
            except Exception:
                pass
            # Run supervised set; returns when shutdown_event is set
            await run_supervisor(job_specs, jx_state.shutdown_event, cfg)
        except RecursionError:
            # Deep recursion during shutdown, force exit
            jx_state.shutdown_event.set()
            import sys
            sys.exit(1)
        finally:
            # Stop auxiliary background workers first
            try:
                await stop_error_worker()
            except Exception:
                pass
            
            try:
                await stop_memory_optimizer()
            except Exception:
                pass
            
            try:
                # Ensure embeddings services are cancelled/awaited for clean shutdown
                await stop_embeddings_task()
            except Exception:
                pass
            
            try:
                await stop_project_embeddings_task()
            except Exception:
                pass
            
            try:
                # Cancel/await self-study tasks started by ensure_runtime()
                await _stop_selfstudy()
            except Exception:
                pass
            
            try:
                # Stop optional plugins last
                await _stop_plugins()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            
            # Ensure all pending tasks are properly cleaned up (avoid deep recursion)
            try:
                loop = asyncio.get_running_loop()
                # Get only top-level tasks, not nested ones
                pending = [
                    t for t in asyncio.all_tasks(loop) 
                    if not t.done() and t != asyncio.current_task()
                ]
                
                # Cancel tasks one by one with shallow cancellation
                for task in pending:
                    try:
                        task.cancel()
                    except RecursionError:
                        # If recursion error, skip this task
                        pass
                
                # Wait with timeout to prevent hanging
                if pending:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*pending, return_exceptions=True),
                            timeout=2.0
                        )
                    except asyncio.TimeoutError:
                        # Some tasks didn't finish, that's ok
                        pass
            except RecursionError:
                # Deep recursion detected, bail out gracefully
                pass
            except Exception:
                pass
            
            # Ensure ProcessPoolExecutor is torn down to avoid atexit join hang
            with contextlib.suppress(Exception):
                _retr_pool_shutdown()
            # Finalize run recorder
            try:
                if _run_id:
                    from jinx.observability.recorder import finalize_run_record
                    finalize_run_record(_run_id, extra={"status": "shutdown"})
            except Exception:
                pass
