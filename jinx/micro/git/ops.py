from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


@dataclass
class GitToolingError(Exception):
    kind: str
    message: str
    command: Optional[str] = None
    status_code: Optional[int] = None
    stderr: Optional[str] = None

    def __str__(self) -> str:
        base = f"{self.kind}: {self.message}"
        if self.command:
            base += f" [cmd={self.command}]"
        if self.status_code is not None:
            base += f" [code={self.status_code}]"
        if self.stderr:
            base += f" [stderr={self.stderr}]"
        return base


def _build_command_string(args: Sequence[str]) -> str:
    if not args:
        return "git"
    return "git " + " ".join(args)


async def _run_git(dir_: Path | str, args: Iterable[str], env: Optional[dict[str, str]] = None):
    cwd = str(dir_)
    argv = [str(a) for a in args]
    cmd = _build_command_string(argv)
    merged_env = os.environ.copy()
    merged_env.setdefault("GIT_TERMINAL_PROMPT", "0")
    if env:
        merged_env.update(env)
    proc = await asyncio.create_subprocess_exec(
        "git",
        *argv,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=merged_env,
    )
    timeout_s = None
    try:
        ts = os.getenv("JINX_GIT_TIMEOUT_SECONDS", "").strip()
        if ts:
            timeout_s = float(ts)
    except Exception:
        timeout_s = None
    try:
        if timeout_s and timeout_s > 0:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        else:
            out_b, err_b = await proc.communicate()
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            await proc.communicate()
        except Exception:
            pass
        raise GitToolingError(
            kind="GitTimeout",
            message="git timed out",
            command=cmd,
        )
    return proc.returncode, out_b, err_b, cmd


async def run_git_for_status(dir_: Path | str, args: Iterable[str], env: Optional[dict[str, str]] = None) -> None:
    code, _out, err, cmd = await _run_git(dir_, args, env)
    if code != 0:
        raise GitToolingError(
            kind="GitCommand",
            message="git exited with non-zero status",
            command=cmd,
            status_code=code,
            stderr=(err.decode(errors="ignore").strip() if err else None),
        )


async def run_git_for_stdout(dir_: Path | str, args: Iterable[str], env: Optional[dict[str, str]] = None) -> str:
    code, out, err, cmd = await _run_git(dir_, args, env)
    if code != 0:
        raise GitToolingError(
            kind="GitCommand",
            message="git exited with non-zero status",
            command=cmd,
            status_code=code,
            stderr=(err.decode(errors="ignore").strip() if err else None),
        )
    return (out.decode(errors="ignore").strip() if out else "")


async def run_git_for_stdout_all(dir_: Path | str, args: Iterable[str], env: Optional[dict[str, str]] = None) -> str:
    code, out, err, cmd = await _run_git(dir_, args, env)
    if code != 0:
        raise GitToolingError(
            kind="GitCommand",
            message="git exited with non-zero status",
            command=cmd,
            status_code=code,
            stderr=(err.decode(errors="ignore").strip() if err else None),
        )
    return (out.decode(errors="ignore") if out else "")


async def ensure_git_repository(path: Path | str) -> None:
    try:
        out = await run_git_for_stdout(path, ["rev-parse", "--is-inside-work-tree"])
    except GitToolingError as e:
        if e.status_code == 128:
            raise GitToolingError("NotAGitRepository", f"{path}")
        raise
    if out.strip() != "true":
        raise GitToolingError("NotAGitRepository", f"{path}")


async def resolve_head(path: Path | str) -> Optional[str]:
    try:
        sha = await run_git_for_stdout(path, ["rev-parse", "--verify", "HEAD"])
        return sha
    except GitToolingError as e:
        if e.status_code == 128:
            return None
        raise


async def resolve_repository_root(path: Path | str) -> str:
    root = await run_git_for_stdout(path, ["rev-parse", "--show-toplevel"])
    return root


def normalize_relative_path(p: Path | str) -> Path:
    path = Path(p)
    if path.is_absolute() or (getattr(path, "drive", "") not in ("", None)):
        raise GitToolingError("NonRelativePath", str(path))
    result_parts: list[str] = []
    saw_component = False
    for part in path.parts:
        if part in ("", "."):
            saw_component = True
            continue
        if part == "..":
            saw_component = True
            if not result_parts:
                raise GitToolingError("PathEscapesRepository", str(path))
            result_parts.pop()
            continue
        saw_component = True
        result_parts.append(part)
    if not saw_component:
        raise GitToolingError("NonRelativePath", str(path))
    return Path(*result_parts)


def apply_repo_prefix_to_force_include(prefix: Optional[Path], paths: list[Path]) -> list[Path]:
    if not paths:
        return []
    if prefix is None:
        return list(paths)
    return [prefix.joinpath(p) for p in paths]


def repo_subdir(repo_root: Path | str, repo_path: Path | str) -> Optional[Path]:
    root = Path(repo_root)
    path = Path(repo_path)
    if root == path:
        return None
    try:
        rel = path.relative_to(root)
        return rel if str(rel) else None
    except Exception:
        try:
            rc = root.resolve(strict=False)
            pc = path.resolve(strict=False)
            rel = pc.relative_to(rc)
            return rel if str(rel) else None
        except Exception:
            return None
