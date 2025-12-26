from __future__ import annotations

from pathlib import Path
from typing import Optional

from .ops import (
    GitToolingError,
    ensure_git_repository,
    resolve_head,
    resolve_repository_root,
    run_git_for_stdout,
)


async def _resolve_branch_ref(repo_root: str | Path, branch: str) -> Optional[str]:
    try:
        rev = await run_git_for_stdout(repo_root, ["rev-parse", "--verify", branch])
        return rev
    except GitToolingError:
        return None


async def _resolve_upstream_if_remote_ahead(repo_root: str | Path, branch: str) -> Optional[str]:
    try:
        upstream = await run_git_for_stdout(
            repo_root,
            [
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                f"{branch}@{{upstream}}",
            ],
        )
        upstream = upstream.strip()
        if not upstream:
            return None
    except GitToolingError:
        return None

    try:
        counts = await run_git_for_stdout(
            repo_root,
            ["rev-list", "--left-right", "--count", f"{branch}...{upstream}"],
        )
    except GitToolingError:
        return None
    parts = counts.split()
    # left is local ahead, right is remote ahead
    right = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return upstream if right > 0 else None


async def merge_base_with_head(repo_path: str | Path, branch: str) -> Optional[str]:
    await ensure_git_repository(repo_path)
    repo_root = await resolve_repository_root(repo_path)
    head = await resolve_head(repo_root)
    if not head:
        return None

    branch_ref = await _resolve_branch_ref(repo_root, branch)
    if branch_ref is None:
        return None

    preferred_ref = branch_ref
    upstream = await _resolve_upstream_if_remote_ahead(repo_root, branch)
    if upstream:
        # if upstream resolves, prefer it, else fallback to branch_ref
        alt = await _resolve_branch_ref(repo_root, upstream)
        preferred_ref = alt or branch_ref

    merge_base = await run_git_for_stdout(repo_root, ["merge-base", head, preferred_ref])
    return merge_base if merge_base else None
