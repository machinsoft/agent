from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict, Iterable


def ensure_optional(packages: Iterable[str]) -> Dict[str, ModuleType]:
    """Ensure packages are importable; install on demand if missing.

    Returns a dict mapping package name to imported module.
    """
    mods: Dict[str, ModuleType] = {}
    for name in packages:
        try:
            mods[name] = import_module(name)
        except Exception:
            continue
    return mods
