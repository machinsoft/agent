from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
import shutil

from .ops import (
    run_git_for_status,
    run_git_for_stdout,
    run_git_for_stdout_all,
    ensure_git_repository,
    resolve_repository_root,
    resolve_head,
    normalize_relative_path,
    apply_repo_prefix_to_force_include,
    repo_subdir,
)


DEFAULT_COMMIT_MESSAGE = "codex snapshot"


@dataclass
class GhostCommit:
    id: str
    parent: Optional[str]
    preexisting_untracked_files: list[str] = field(default_factory=list)
    preexisting_untracked_dirs: list[str] = field(default_factory=list)


@dataclass
class GhostSnapshotConfig:
    ignore_large_untracked_files: Optional[int] = 10 * 1024 * 1024
    ignore_large_untracked_dirs: Optional[int] = 200
    disable_warnings: bool = False


@dataclass
class GhostSnapshotReport:
    large_untracked_dirs: list["LargeUntrackedDir"] = field(default_factory=list)
    ignored_untracked_files: list["IgnoredUntrackedFile"] = field(default_factory=list)


@dataclass
class LargeUntrackedDir:
    path: str
    file_count: int


@dataclass
class IgnoredUntrackedFile:
    path: str
    byte_size: int


@dataclass
class CreateGhostCommitOptions:
    repo_path: Path | str
    message: Optional[str] = None
    force_include: list[Path] = field(default_factory=list)
    ghost_snapshot: GhostSnapshotConfig = field(default_factory=GhostSnapshotConfig)


@dataclass
class RestoreGhostCommitOptions:
    repo_path: Path | str
    ghost_snapshot: GhostSnapshotConfig = field(default_factory=GhostSnapshotConfig)


async def _ls_untracked(repo_root: Path | str) -> list[str]:
    out = await run_git_for_stdout(repo_root, ["ls-files", "--others", "--exclude-standard", "-z"])
    if not out:
        return []
    items = [p for p in out.split("\x00") if p]
    return items


def _collect_untracked_dirs(files: list[str]) -> list[str]:
    seen: set[str] = set()
    for f in files:
        p = Path(f)
        parent = str(p.parent) if str(p.parent) != "." else ""
        if parent:
            seen.add(parent)
    return sorted(seen)


def _with_env(base: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    merged = dict(base)
    merged.update(extra)
    return merged


async def create_ghost_commit(options: CreateGhostCommitOptions) -> GhostCommit:
    await ensure_git_repository(options.repo_path)
    repo_root = await resolve_repository_root(options.repo_path)
    parent = await resolve_head(repo_root)

    tmp = tempfile.TemporaryDirectory(prefix="codex-git-index-")
    idx = str(Path(tmp.name) / "index")
    base_env = {"GIT_INDEX_FILE": idx}

    if parent:
        await run_git_for_status(repo_root, ["read-tree", parent], base_env)

    await run_git_for_status(repo_root, ["add", "--all", "--", "."], base_env)
    if options.force_include:
        prefix = repo_subdir(Path(repo_root), Path(options.repo_path)) if options.repo_path else None
        norm = [normalize_relative_path(Path(p)) for p in options.force_include]
        forced = apply_repo_prefix_to_force_include(prefix, norm)
        args = ["add", "--force", *[str(p) for p in forced]]
        await run_git_for_status(repo_root, args, base_env)

    tree_id = await run_git_for_stdout(repo_root, ["write-tree"], base_env)

    commit_env = _with_env(base_env, {
        "GIT_AUTHOR_NAME": "Codex Snapshot",
        "GIT_AUTHOR_EMAIL": "snapshot@codex.local",
        "GIT_COMMITTER_NAME": "Codex Snapshot",
        "GIT_COMMITTER_EMAIL": "snapshot@codex.local",
    })
    msg = options.message or DEFAULT_COMMIT_MESSAGE
    args = ["commit-tree", tree_id]
    if parent:
        args.extend(["-p", parent])
    args.extend(["-m", msg])
    commit_id = await run_git_for_stdout(repo_root, args, commit_env)

    untracked = await _ls_untracked(repo_root)
    pre_files = sorted(untracked)
    pre_dirs = _collect_untracked_dirs(pre_files)

    return GhostCommit(id=commit_id, parent=parent, preexisting_untracked_files=pre_files, preexisting_untracked_dirs=pre_dirs)


async def capture_ghost_snapshot_report(options: CreateGhostCommitOptions) -> GhostSnapshotReport:
    await ensure_git_repository(options.repo_path)
    repo_root = await resolve_repository_root(options.repo_path)
    prefix = repo_subdir(Path(repo_root), Path(options.repo_path)) if options.repo_path else None
    args = ["status", "--porcelain=2", "-z", "--untracked-files=all"]
    if prefix:
        args.extend(["--", str(prefix)])
    out = await run_git_for_stdout_all(repo_root, args)  
    if not out:
        return GhostSnapshotReport()
    untracked_files: list[str] = []
    for rec in out.split("\x00"):
        if not rec:
            continue
        tag = rec[0]
        if tag in ("?", "!"):
            parts = rec.split(" ", 1)
            if len(parts) == 2 and parts[1]:
                p = parts[1].strip()
                if p:
                    if tag == "?":
                        untracked_files.append(p)
                    continue
    large_files: list[IgnoredUntrackedFile] = []
    thr_f = options.ghost_snapshot.ignore_large_untracked_files
    if thr_f and thr_f > 0:
        for p in untracked_files:
            ap = Path(repo_root).joinpath(p)
            try:
                sz = ap.stat().st_size
            except Exception:
                continue
            if sz > thr_f:
                large_files.append(IgnoredUntrackedFile(path=p, byte_size=int(sz)))
    thr_d = options.ghost_snapshot.ignore_large_untracked_dirs
    large_dirs: list[LargeUntrackedDir] = []
    if thr_d and thr_d > 0:
        counts: dict[str, int] = {}
        for p in untracked_files:
            parent = str(Path(p).parent)
            if parent and parent != ".":
                counts[parent] = counts.get(parent, 0) + 1
        for d, n in counts.items():
            if n >= thr_d:
                large_dirs.append(LargeUntrackedDir(path=d, file_count=int(n)))
        large_dirs.sort(key=lambda x: (-x.file_count, x.path))
    return GhostSnapshotReport(large_untracked_dirs=large_dirs, ignored_untracked_files=large_files)


async def restore_ghost_commit(repo_path: Path | str, commit: GhostCommit) -> None:
    await restore_ghost_commit_with_options(RestoreGhostCommitOptions(repo_path=repo_path), commit)


async def restore_ghost_commit_with_options(options: RestoreGhostCommitOptions, commit: GhostCommit) -> None:
    repo_root = await resolve_repository_root(options.repo_path)
    await restore_to_commit(repo_root, commit.id)
    cur_untracked = await _ls_untracked(repo_root)
    keep_files = set(commit.preexisting_untracked_files or [])
    keep_dirs = set(commit.preexisting_untracked_dirs or [])
    for p in cur_untracked:
        if p in keep_files:
            continue
        if any(str(Path(p)).startswith(d + "\\") or str(Path(p)).startswith(d + "/") or str(Path(p)) == d for d in keep_dirs):
            continue
        ap = Path(repo_root).joinpath(p)
        try:
            if ap.is_dir():
                shutil.rmtree(ap, ignore_errors=True)
            else:
                ap.unlink(missing_ok=True)
        except Exception:
            pass


async def restore_to_commit(repo_path: Path | str, commit_id: str) -> None:
    await run_git_for_status(repo_path, ["restore", "--source", commit_id, "--worktree", "--", "."])
