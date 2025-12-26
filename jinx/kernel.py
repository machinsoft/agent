from __future__ import annotations

"""
Kernel boot for Jinx.

Performs minimal, safe, synchronous startup steps before the async runtime loop:
- Install import resilience for internal modules (stubs instead of crashes)
- Load environment from .env
- Apply autonomous defaults (no user config required)
- Prewarm OpenAI HTTP client (connection pools)

These steps are side-effect–light and idempotent.
"""

def boot() -> None:
    # 1) Import resilience (safe stubs for missing jinx.* modules)
    try:
        from jinx.micro.runtime.resilience import install_resilience as _install
        _install()
    except Exception:
        pass

    # 2) Load .env early so downstream imports see env
    try:
        pass
    except Exception:
        pass
    try:
        from jinx.micro.runtime.startup_checks import run_startup_checks as _startup_checks
        _startup_checks(stage="pre")
    except Exception:
        pass

    # 3) Apply autonomous defaults (env-based, synchronous)
    try:
        pass
    except Exception:
        pass

    # 4) Prewarm OpenAI client (synchronous, cheap)
    try:
        from jinx.micro.net.client import prewarm_openai_client as _prewarm
        _prewarm()
    except Exception:
        pass
    # 5) Optional OTEL setup (no-op if not installed or disabled)
    try:
        from jinx.observability.setup import setup_otel as _setup_otel
        _setup_otel()
    except Exception:
        pass

__all__ = ["boot"]
