from __future__ import annotations

from typing import List

# Pull data directly to avoid unresolved macros and keep RT behavior
from jinx.micro.runtime.exports import collect_export, collect_export_for_group
from jinx.micro.runtime.task_ctx import get_current_group as _get_group

_PREVIEW_CHARS = 160
_RUN_TTL_MS = 120000


def _clamp(s: str, lim: int) -> str:
    s = (s or "").strip()
    if lim <= 0:
        return s
    return s[:lim]


async def patch_exports_lines() -> List[str]:
    """Smart patch dashboard lines (resolved values, group-aware).

    - Aggregates recent patch artifacts from all programs via runtime exports.
    - Computes simple size metrics from commit preview.
    - Includes current logical group to align with concurrent turns.
    """
    try:
        gid = _get_group()
    except Exception:
        gid = "main"
    # Preview clamp
    prev_chars = _PREVIEW_CHARS
    # Collect aggregated exports
    try:
        prevs = await collect_export_for_group("last_patch_preview", gid, limit=1)
    except Exception:
        prevs = []
    try:
        commits = await collect_export_for_group("last_patch_commit", gid, limit=1)
    except Exception:
        commits = []
    try:
        strategy = (await collect_export_for_group("last_patch_strategy", gid, limit=1) or [""])[0]
    except Exception:
        strategy = ""
    try:
        reason = (await collect_export_for_group("last_patch_reason", gid, limit=1) or [""])[0]
    except Exception:
        reason = ""
    prev = _clamp((prevs[0] if prevs else ""), prev_chars)
    commit = commits[0] if commits else ""
    # Simple size proxy: changed lines count
    try:
        lines_changed = commit.count("\n") if commit else 0
    except Exception:
        lines_changed = 0
    lines: List[str] = []
    lines.append(f"[patch][group={gid}] strategy={strategy or '-'} | lines={lines_changed} | reason={(reason or '-').strip()}")
    if prev:
        lines.append(f"preview: {prev}")
    # Optional short commit head
    if commit:
        _flat = commit.replace("\r", " ").replace("\n", " ")
        lines.append("commit: " + _clamp(_flat, max(80, prev_chars)))
    return lines


async def verify_exports_lines() -> List[str]:
    """Verification dashboard lines (resolved values).

    - Pulls score/reason/files directly from exports aggregator.
    - Emits a compact single-line summary plus optional files list.
    """
    try:
        gid = _get_group()
    except Exception:
        gid = "main"
    try:
        score = (await collect_export_for_group("last_verify_score", gid, limit=1) or [""])[0]
    except Exception:
        score = ""
    try:
        reason = (await collect_export_for_group("last_verify_reason", gid, limit=1) or [""])[0]
    except Exception:
        reason = ""
    try:
        files = (await collect_export_for_group("last_verify_files", gid, limit=1) or [""])[0]
    except Exception:
        files = ""
    out: List[str] = []
    score_s = (score or "").strip()
    reason_s = (reason or "").strip()
    files_s = (files or "").strip()
    base = f"[verify] score={score_s or '?'}"
    if files_s:
        base += f" | files={files_s}"
    if reason_s:
        base += f" | {reason_s}"
    out.append(base)
    return out


async def run_exports_lines(run_chars: int | None = None) -> List[str]:
    """Last run dashboard (resolved values, TTL-respecting).

    - Uses run_exports helpers directly to avoid unresolved macros.
    - Respects JINX_RUN_EXPORT_TTL_MS and preview char budget.
    """
    if run_chars is None:
        run_chars = _PREVIEW_CHARS
    ttl_ms = _RUN_TTL_MS
    from jinx.micro.exec.run_exports import read_last_stdout, read_last_stderr, read_last_status
    try:
        status = read_last_status(ttl_ms)
    except Exception:
        status = ""
    try:
        stdout = read_last_stdout(3, run_chars, ttl_ms)
    except Exception:
        stdout = ""
    try:
        stderr = read_last_stderr(2, run_chars, ttl_ms)
    except Exception:
        stderr = ""
    lines: List[str] = []
    lines.append(f"[run] status={(status or '').strip() or 'unknown'}")
    if stdout:
        lines.append(f"stdout: {stdout.strip()}")
    if stderr:
        lines.append(f"stderr: {stderr.strip()}")
    return lines


__all__ = [
    "patch_exports_lines",
    "verify_exports_lines",
    "run_exports_lines",
]
