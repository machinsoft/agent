from __future__ import annotations

import asyncio
from typing import Tuple


async def _run_git(*args: str, cwd: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        return ""
    return out.decode(errors="ignore").strip()


async def git_diff_to_remote(cwd: str) -> Tuple[str, str]:
    """Return (sha, diff) vs upstream if available; fall back to empty strings.

    - sha: current HEAD commit hash (short).
    - diff: unified diff vs upstream ("@{u}") if tracking branch exists, else empty.
    """
    sha = await _run_git("rev-parse", "--short", "HEAD", cwd=cwd)
    # Try upstream diff first
    diff = await _run_git("diff", "@{u}", cwd=cwd) or await _run_git("diff", cwd=cwd)
    return sha, diff
