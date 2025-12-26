from __future__ import annotations

import sys
import types
import os
import importlib
import asyncio
from typing import List, Optional
from jinx.micro.common.env import truthy

# Resilience layer: ensures imports of internal modules do not crash the process.
# When enabled (JINX_RESILIENCE=1), missing modules under 'jinx.' are provided
# as stub modules that safely no-op and record the missing import for later repair.

_missing: List[str] = []
_installed: bool = False
_orig_import_module = None  # type: ignore[var-annotated]
_monitor_task: asyncio.Task | None = None


def _enabled() -> bool:
    return True


def _record_missing(name: str) -> None:
    try:
        if name not in _missing:
            _missing.append(name)
    except Exception:
        pass


def _make_stub_module(fullname: str) -> types.ModuleType:
    m = types.ModuleType(fullname)
    m.__dict__["__jinx_resilient__"] = True
    m.__dict__["__missing__"] = True

    class _Dummy:
        def __getattr__(self, _name):
            return self
        def __call__(self, *a, **k):
            return None
        def __iter__(self):
            return iter(())
        def __await__(self):
            async def _noop():
                return None
            return _noop().__await__()

    _dummy = _Dummy()

    def __getattr__(name: str):  # module-level __getattr__ (PEP 562)
        return _dummy

    m.__getattr__ = __getattr__  # type: ignore[attr-defined]
    m.__all__ = []  # type: ignore[attr-defined]
    return m


class _ResilientLoader:
    def __init__(self, fullname: str) -> None:
        self.fullname = fullname

    def create_module(self, spec):  # pragma: no cover
        return None  # use default module creation

    def exec_module(self, module):  # pragma: no cover
        # Replace the created module with our stub content
        stub = _make_stub_module(self.fullname)
        module.__dict__.update(stub.__dict__)
        _record_missing(self.fullname)


class _ResilientFinder:
    def find_spec(self, fullname: str, path, target=None):  # pragma: no cover
        if not _enabled():
            return None
        # Only handle our internal namespace, avoid shadowing stdlib/site-packages
        if not fullname or not fullname.startswith("jinx."):
            return None
        # If it already exists, do nothing
        if fullname in sys.modules:
            return None
        # Let normal finders try first; we run last in sys.meta_path
        # If previous finders failed, provide a stub spec
        import importlib.machinery as _machinery
        return _machinery.ModuleSpec(fullname, _ResilientLoader(fullname))


def install_resilience() -> None:
    global _installed
    if _installed:
        return
    try:
        # append to the END so standard loaders run first
        sys.meta_path.append(_ResilientFinder())
        # Wrap importlib.import_module to stub on failing imports for jinx.*
        global _orig_import_module
        if _orig_import_module is None:
            _orig_import_module = importlib.import_module
            def _wrapped_import_module(name: str, package: Optional[str] = None):  # type: ignore[override]
                try:
                    return _orig_import_module(name, package)
                except Exception:
                    # Fallback: if jinx.* module import failed, install stub and return it
                    full = name if not package else importlib.util.resolve_name(name, package)  # type: ignore[attr-defined]
                    if isinstance(full, str) and full.startswith("jinx."):
                        mod = _make_stub_module(full)
                        sys.modules[full] = mod
                        _record_missing(full)
                        return mod
                    raise
            importlib.import_module = _wrapped_import_module  # type: ignore[assignment]
        _installed = True
    except Exception:
        _installed = True


def pending_missing() -> List[str]:
    return list(_missing)


async def schedule_repairs() -> None:
    """Submit repair tasks for any missing internal modules detected so far."""
    if not _missing:
        return
    try:
        from jinx.micro.runtime.api import submit_task as _submit
    except Exception:
        return
    for mod in list(_missing):
        try:
            await _submit("repair.import_missing", module=str(mod))
        except Exception:
            pass


def start_resilience_monitor_task(period_s: Optional[float] = None) -> asyncio.Task:
    """Start a background task that periodically invokes schedule_repairs().

    Controlled by JINX_RESILIENCE_MONITOR (default ON). Returns the task.
    """
    if not truthy("JINX_RESILIENCE_MONITOR", "1"):
        return asyncio.create_task(asyncio.sleep(0))
    global _monitor_task
    if _monitor_task and not _monitor_task.done():
        return _monitor_task
    p = 45.0 if period_s is None else float(period_s)

    async def _loop() -> None:
        while True:
            try:
                await asyncio.sleep(max(5.0, p))
                await schedule_repairs()
            except Exception:
                # never crash the monitor
                await asyncio.sleep(max(5.0, p))

    _monitor_task = asyncio.create_task(_loop(), name="resilience-monitor")
    return _monitor_task
