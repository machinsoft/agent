from __future__ import annotations

from typing import Any

__all__ = [
    "spike_exec",
    "spawn_pty_process",
    "ExecCommandSession",
    "SpawnedPty",
]


def __getattr__(name: str) -> Any:
    if name == "spike_exec":
        from .executor import spike_exec  # local import to avoid circular deps
        return spike_exec
    if name == "spawn_pty_process":
        from .pty import spawn_pty_process  # type: ignore
        return spawn_pty_process
    if name == "ExecCommandSession":
        from .pty import ExecCommandSession  # type: ignore
        return ExecCommandSession
    if name == "SpawnedPty":
        from .pty import SpawnedPty  # type: ignore
        return SpawnedPty
    raise AttributeError(name)
