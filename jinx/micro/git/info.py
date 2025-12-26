from __future__ import annotations

from typing import Tuple
from .ops import run_git_for_stdout, GitToolingError


async def git_diff_to_remote(cwd: str) -> Tuple[str, str]:
    sha = ""
    try:
        sha = await run_git_for_stdout(cwd, ["rev-parse", "--short", "HEAD"])
    except GitToolingError:
        sha = ""
    upstream = ""
    try:
        upstream = await run_git_for_stdout(cwd, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    except GitToolingError:
        upstream = ""
    diff = ""
    try:
        if upstream:
            diff = await run_git_for_stdout(cwd, ["diff", upstream])
        else:
            diff = await run_git_for_stdout(cwd, ["diff"])
    except GitToolingError:
        diff = ""
    return sha, diff
