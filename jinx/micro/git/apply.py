from __future__ import annotations

import asyncio
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .ops import run_git_for_stdout, GitToolingError, run_git_for_status


@dataclass
class ApplyGitRequest:
    cwd: Path
    diff: str
    revert: bool
    preflight: bool


@dataclass
class ApplyGitResult:
    exit_code: int
    applied_paths: list[str]
    skipped_paths: list[str]
    conflicted_paths: list[str]
    stdout: str
    stderr: str
    cmd_for_log: str


async def apply_git_patch(req: ApplyGitRequest) -> ApplyGitResult:
    git_root = await run_git_for_stdout(req.cwd, ["rev-parse", "--show-toplevel"])
    tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    try:
        tmp.write(req.diff)
        tmp.flush()
        tmp_path = Path(tmp.name)
    finally:
        tmp.close()

    if req.revert and not req.preflight:
        await stage_paths(Path(git_root), req.diff)

    cfg_parts: list[str] = []
    cfg = os.getenv("CODEX_APPLY_GIT_CFG", "").strip()
    if cfg:
        for pair in cfg.split(","):
            p = pair.strip()
            if p and "=" in p:
                cfg_parts.extend(["-c", p])

    args: list[str] = ["apply"]
    if req.preflight:
        args.append("--check")
    else:
        args.append("--3way")
    if req.revert:
        args.append("-R")
    args.append(str(tmp_path))

    cmd_for_log = _render_command_for_log(Path(git_root), cfg_parts, args)
    code, out, err = await _run_git_raw(Path(git_root), cfg_parts, args)

    applied, skipped, conflicted = parse_git_apply_output(out, err)
    applied = sorted(set(applied))
    skipped = sorted(set(skipped))
    conflicted = sorted(set(conflicted))

    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    return ApplyGitResult(
        exit_code=code,
        applied_paths=applied,
        skipped_paths=skipped,
        conflicted_paths=conflicted,
        stdout=out,
        stderr=err,
        cmd_for_log=cmd_for_log,
    )


async def _run_git_raw(cwd: Path, cfg_parts: list[str], args: list[str]) -> Tuple[int, str, str]:
    argv = ["git", *cfg_parts, *args]
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await proc.communicate()
    code = int(proc.returncode or 0)
    return code, (out_b.decode(errors="ignore") if out_b else ""), (err_b.decode(errors="ignore") if err_b else "")


def _quote_shell(s: str) -> str:
    if all(c.isalnum() or c in "-_.:/@%+" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def _render_command_for_log(cwd: Path, cfg_parts: list[str], args: list[str]) -> str:
    parts: list[str] = ["git", *(_quote_shell(p) for p in cfg_parts), *(_quote_shell(a) for a in args)]
    return f"(cd {_quote_shell(str(cwd))} && {' '.join(parts)})"


def extract_paths_from_patch(diff_text: str) -> list[str]:
    re_hdr = re.compile(r"(?m)^diff --git a/(.*?) b/(.*)$")
    paths: set[str] = set()
    for m in re_hdr.finditer(diff_text):
        a = (m.group(1) or "").strip()
        b = (m.group(2) or "").strip()
        if a and a != "/dev/null":
            paths.add(a)
        if b and b != "/dev/null":
            paths.add(b)
    return sorted(paths)


async def stage_paths(git_root: Path, diff: str) -> None:
    paths = extract_paths_from_patch(diff)
    existing: list[str] = []
    for p in paths:
        if git_root.joinpath(p).exists():
            existing.append(p)
    if not existing:
        return
    await run_git_for_status(git_root, ["add", "--", *existing])


def parse_git_apply_output(stdout: str, stderr: str) -> Tuple[list[str], list[str], list[str]]:
    combined = "\n".join(s for s in (stdout, stderr) if s)

    applied: set[str] = set()
    skipped: set[str] = set()
    conflicted: set[str] = set()
    last_seen_path: Optional[str] = None

    def add(target: set[str], raw: str) -> None:
        t = raw.strip()
        if not t:
            return
        if (t[0] in ('"', "'") and t[-1] == t[0] and len(t) >= 2):
            t = t[1:-1]
        target.add(t)

    APPLIED_CLEAN = re.compile(r"(?i)^Applied patch(?: to)?\s+(?P<path>.+?)\s+cleanly\.?$")
    APPLIED_CONFLICTS = re.compile(r"(?i)^Applied patch(?: to)?\s+(?P<path>.+?)\s+with conflicts\.?$")
    APPLYING_WITH_REJECTS = re.compile(r"(?i)^Applying patch\s+(?P<path>.+?)\s+with\s+\d+\s+rejects?\.{0,3}$")
    CHECKING_PATCH = re.compile(r"(?i)^Checking patch\s+(?P<path>.+?)\.\.\.$")
    UNMERGED_LINE = re.compile(r"(?i)^U\s+(?P<path>.+)$")
    PATCH_FAILED = re.compile(r"(?i)^error:\s+patch failed:\s+(?P<path>.+?)(?::\d+)?(?:\s|$)")
    DOES_NOT_APPLY = re.compile(r"(?i)^error:\s+(?P<path>.+?):\s+patch does not apply$")
    THREE_WAY_START = re.compile(r"(?i)^(?:Performing three-way merge|Falling back to three-way merge)\.\.\.$")
    THREE_WAY_FAILED = re.compile(r"(?i)^Failed to perform three-way merge\.\.\.$")
    FALLBACK_DIRECT = re.compile(r"(?i)^Falling back to direct application\.\.\.$")
    LACKS_BLOB = re.compile(r"(?i)^(?:error: )?repository lacks the necessary blob to (?:perform|fall back on) 3-?way merge\.?$")
    INDEX_MISMATCH = re.compile(r"(?i)^error:\s+(?P<path>.+?):\s+does not match index\b")
    NOT_IN_INDEX = re.compile(r"(?i)^error:\s+(?P<path>.+?):\s+does not exist in index\b")
    ALREADY_EXISTS_WT = re.compile(r"(?i)^error:\s+(?P<path>.+?)\s+already exists in (?:the )?working directory\b")
    FILE_EXISTS = re.compile(r"(?i)^error:\s+patch failed:\s+(?P<path>.+?)\s+File exists")
    RENAMED_DELETED = re.compile(r"(?i)^error:\s+path\s+(?P<path>.+?)\s+has been renamed\/deleted")
    CANNOT_APPLY_BINARY = re.compile(r"(?i)^error:\s+cannot apply binary patch to\s+['\"]?(?P<path>.+?)['\"]?\s+without full index line$")
    BINARY_DOES_NOT_APPLY = re.compile(r"(?i)^error:\s+binary patch does not apply to\s+['\"]?(?P<path>.+?)['\"]?$")
    BINARY_INCORRECT_RESULT = re.compile(r"(?i)^error:\s+binary patch to\s+['\"]?(?P<path>.+?)['\"]?\s+creates incorrect result\b")
    CANNOT_READ_CURRENT = re.compile(r"(?i)^error:\s+cannot read the current contents of\s+['\"]?(?P<path>.+?)['\"]?$")
    SKIPPED_PATCH = re.compile(r"(?i)^Skipped patch\s+['\"]?(?P<path>.+?)['\"]\.$")
    CANNOT_MERGE_BINARY_WARN = re.compile(r"(?i)^warning:\s*Cannot merge binary files:\s+(?P<path>.+?)\s+\(ours\s+vs\.\s+theirs\)")

    for raw_line in combined.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = CHECKING_PATCH.search(line)
        if m:
            last_seen_path = m.group("path")
            continue
        m = APPLIED_CLEAN.search(line)
        if m:
            add(applied, m.group("path"))
            p = next(iter(sorted(applied))) if applied else None
            if p:
                conflicted.discard(p)
                skipped.discard(p)
                last_seen_path = p
            continue
        m = APPLIED_CONFLICTS.search(line)
        if m:
            add(conflicted, m.group("path"))
            p = next(iter(sorted(conflicted))) if conflicted else None
            if p:
                applied.discard(p)
                skipped.discard(p)
                last_seen_path = p
            continue
        if APPLYING_WITH_REJECTS.search(line):
            if last_seen_path:
                add(conflicted, last_seen_path)
                applied.discard(last_seen_path)
                skipped.discard(last_seen_path)
            continue
        m = UNMERGED_LINE.search(line)
        if m:
            add(conflicted, m.group("path"))
            p = next(iter(sorted(conflicted))) if conflicted else None
            if p:
                applied.discard(p)
                skipped.discard(p)
                last_seen_path = p
            continue
        if PATCH_FAILED.search(line) or DOES_NOT_APPLY.search(line):
            m = PATCH_FAILED.search(line) or DOES_NOT_APPLY.search(line)
            if m:
                last_seen_path = m.group("path")
                add(skipped, last_seen_path)
            continue
        if THREE_WAY_START.search(line) or FALLBACK_DIRECT.search(line):
            continue
        if THREE_WAY_FAILED.search(line) or LACKS_BLOB.search(line):
            if last_seen_path:
                add(skipped, last_seen_path)
                applied.discard(last_seen_path)
                conflicted.discard(last_seen_path)
            continue
        m = (
            INDEX_MISMATCH.search(line)
            or NOT_IN_INDEX.search(line)
            or ALREADY_EXISTS_WT.search(line)
            or FILE_EXISTS.search(line)
            or RENAMED_DELETED.search(line)
            or CANNOT_APPLY_BINARY.search(line)
            or BINARY_DOES_NOT_APPLY.search(line)
            or BINARY_INCORRECT_RESULT.search(line)
            or CANNOT_READ_CURRENT.search(line)
            or SKIPPED_PATCH.search(line)
        )
        if m:
            add(skipped, m.group("path"))
            p_now = next(iter(sorted(skipped))) if skipped else None
            if p_now:
                applied.discard(p_now)
                conflicted.discard(p_now)
                last_seen_path = p_now
            continue
        m = CANNOT_MERGE_BINARY_WARN.search(line)
        if m:
            add(conflicted, m.group("path"))
            p = next(iter(sorted(conflicted))) if conflicted else None
            if p:
                applied.discard(p)
                skipped.discard(p)
                last_seen_path = p
            continue

    for p in list(conflicted):
        applied.discard(p)
        skipped.discard(p)
    for p in list(applied):
        skipped.discard(p)

    return list(applied), list(skipped), list(conflicted)
