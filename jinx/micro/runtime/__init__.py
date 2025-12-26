from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "start_input_task",
    "frame_shift",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "start_input_task": ("jinx.micro.runtime.input_task", "start_input_task"),
    "frame_shift": ("jinx.micro.runtime.frame_shift", "frame_shift"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    mod = importlib.import_module(mod_name)
    val = getattr(mod, attr)
    globals()[name] = val
    return val
