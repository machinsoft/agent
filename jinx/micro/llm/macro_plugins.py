from __future__ import annotations

import asyncio
import importlib.util
import os
from typing import Callable, Awaitable

from .macro_registry import register_macro as _register_macro


async def _call_register(mod, reg: Callable[[str, callable], Awaitable[None]]) -> None:
    """Invoke plugin's register/setup entrypoint if present.

    Entry signature options:
    - async def register(register_macro)
    - def register(register_macro)
    - async def setup(register_macro)
    - def setup(register_macro)
    """
    fn = None
    for name in ("register", "setup"):
        if hasattr(mod, name):
            fn = getattr(mod, name)
            break
    if not fn:
        return
    try:
        if asyncio.iscoroutinefunction(fn):
            await fn(reg)
        else:
            # Run sync and ignore return value
            fn(reg)
    except Exception:
        # Best-effort: ignore plugin errors
        pass


async def load_macro_plugins() -> None:
    """Load macro provider plugins from a fixed repo-relative directory.

    Each plugin module may expose `register(register_macro)` or `setup(register_macro)` and
    should call the provided function to register macros. Errors are swallowed.
    """
    # Compute default plugin directory under repo: jinx/plugins/macros
    try:
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # .../jinx
    except Exception:
        return
    plugin_dir = os.path.join(base, "plugins", "macros")
    try:
        if not os.path.isdir(plugin_dir):
            return
        for entry in os.listdir(plugin_dir):
            if not entry.endswith(".py"):
                continue
            fp = os.path.join(plugin_dir, entry)
            name = f"jinx_plugins_macros_{os.path.splitext(entry)[0]}"
            try:
                spec = importlib.util.spec_from_file_location(name, fp)
                if not spec or not spec.loader:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                # Provide a simple register wrapper that schedules async registration if called sync
                def _reg(ns: str, handler):
                    try:
                        return asyncio.create_task(_register_macro(ns, handler))
                    except Exception:
                        return None
                await _call_register(mod, _reg)
            except Exception:
                # Ignore broken plugins
                continue
    except Exception:
        return
