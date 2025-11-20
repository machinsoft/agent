from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

__all__ = ["summarize_sandbox_policy"]


def _as_bool(v: Any) -> bool:
    try:
        return bool(v)
    except Exception:
        return False


def _as_paths(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for it in items or []:
        try:
            if isinstance(it, (str, Path)):
                out.append(str(it))
            else:
                out.append(str(it))
        except Exception:
            continue
    return out


def summarize_sandbox_policy(policy: Any) -> str:
    """Summarize sandbox policy into a compact string.

    Accepts either:
    - symbolic string: "danger-full-access" | "read-only" | "workspace-write"
    - mapping/object with fields for workspace-write:
        writable_roots: list[str]
        network_access: bool
        exclude_tmpdir_env_var: bool
        exclude_slash_tmp: bool
    Unknown structures fall back to "read-only".
    """
    # Simple symbolic modes
    if isinstance(policy, str):
        m = policy.strip().lower()
        if m in {"danger-full-access", "read-only", "workspace-write"}:
            return m
        return "read-only"

    # Mapping-based policy
    mode = None
    try:
        mode = (getattr(policy, "mode", None) or getattr(policy, "type", None) or "").lower()
    except Exception:
        mode = None
    if not mode and isinstance(policy, dict):
        mode = str(policy.get("mode") or policy.get("type") or "").lower()

    if mode == "danger-full-access":
        return "danger-full-access"
    if mode == "read-only":
        return "read-only"

    # Treat remaining as workspace-write by default
    writable_roots = []
    network = False
    exclude_tmpdir = False
    exclude_slash_tmp = False

    if isinstance(policy, dict):
        writable_roots = _as_paths(policy.get("writable_roots", []))
        network = _as_bool(policy.get("network_access"))
        exclude_tmpdir = _as_bool(policy.get("exclude_tmpdir_env_var"))
        exclude_slash_tmp = _as_bool(policy.get("exclude_slash_tmp"))
    else:
        try:
            writable_roots = _as_paths(getattr(policy, "writable_roots", []) or [])
            network = _as_bool(getattr(policy, "network_access", False))
            exclude_tmpdir = _as_bool(getattr(policy, "exclude_tmpdir_env_var", False))
            exclude_slash_tmp = _as_bool(getattr(policy, "exclude_slash_tmp", False))
        except Exception:
            pass

    writable_entries: list[str] = ["workdir"]
    if not exclude_slash_tmp:
        writable_entries.append("/tmp")
    if not exclude_tmpdir:
        writable_entries.append("$TMPDIR")
    writable_entries.extend(writable_roots)

    summary = f"workspace-write [{', '.join(writable_entries)}]"
    if network:
        summary += " (network access enabled)"
    return summary
