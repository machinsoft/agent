"""Top-level orchestrator.

This module exposes the synchronous ``main()`` function that boots the
asynchronous runtime loop via ``jinx.runtime_service.pulse_core``. Keeping this
adapter minimal ensures clean separation between synchronous CLI entrypoints
and the async runtime core.
"""

from __future__ import annotations

import asyncio
from jinx.bootstrap import load_env


def main() -> None:
    """Run the async runtime loop and block until completion.

    This function is intentionally synchronous so it can be used directly from
    standard CLI entrypoints without requiring the caller to manage an event
    loop.
    """
    def _record(op: str, **kwargs) -> None:
        try:
            from jinx.micro.runtime.crash_diagnostics import record_operation as _rec
            _rec(op, **kwargs)
        except Exception:
            pass

    def _mark(reason: str) -> None:
        try:
            from jinx.micro.runtime.crash_diagnostics import mark_normal_shutdown as _mark_normal
            _mark_normal(reason)
        except Exception:
            pass

    # Install crash diagnostics FIRST
    try:
        from jinx.micro.runtime.crash_diagnostics import install_crash_diagnostics
        install_crash_diagnostics()
        _record("startup", details={'stage': 'orchestrator'}, success=True)
    except Exception:
        pass
    
    # Install shutdown event monitor
    try:
        import jinx.state as jx_state
        import traceback as tb
        from jinx.micro.logger.debug_logger import debug_log_sync
        
        original_set = jx_state.shutdown_event.set
        
        def monitored_set():
            try:
                debug_log_sync("="*70, "SHUTDOWN")
                debug_log_sync("shutdown_event.set() called!", "SHUTDOWN")
                debug_log_sync("="*70, "SHUTDOWN")
                debug_log_sync("Call stack:", "SHUTDOWN")
                for line in tb.format_stack()[:-1]:
                    debug_log_sync(line.strip(), "SHUTDOWN")
                debug_log_sync("="*70, "SHUTDOWN")
            except Exception:
                pass
            return original_set()
        
        jx_state.shutdown_event.set = monitored_set
    except Exception:
        pass
    
    # Ensure environment variables (e.g., OPENAI_API_KEY) are loaded from .env
    load_env()
    # Ensure runtime optional deps are present before importing runtime_service
    pass

    try:
        from jinx.micro.runtime.startup_checks import run_startup_checks as _startup_checks
        _startup_checks(stage="post")
    except Exception:
        pass

    # Defer import until after dependencies are ensured to avoid early import errors
    from jinx.runtime_service import pulse_core

    try:
        _record("runtime_start", success=True)
        asyncio.run(pulse_core())
        _record("runtime_end", success=True)
        
        # Mark as normal shutdown
        _mark("normal_completion")
    except KeyboardInterrupt:
        _record("runtime_interrupted", details={'reason': 'KeyboardInterrupt'}, success=True)
        _mark("keyboard_interrupt")
        raise
    except Exception as e:
        _record("runtime_error", details={'error': str(e)}, success=False, error=str(e))
        raise
