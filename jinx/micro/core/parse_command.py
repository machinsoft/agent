from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional
import os

from jinx.micro.protocol.parse_command import (
    ParsedCommand,
    ParsedCommandListFiles,
    ParsedCommandRead,
    ParsedCommandSearch,
    ParsedCommandUnknown,
)


_READ_CMDS = {"cat", "type", "Get-Content"}
_LIST_CMDS = {"ls", "dir", "Get-ChildItem"}
_SEARCH_CMDS = {"rg", "ripgrep", "grep", "findstr"}


def _is_flag(s: str) -> bool:
    return s.startswith("-") or s.startswith("--") or s.startswith("/")


def parse_command(argv: List[str], cwd: Optional[str] = None) -> List[ParsedCommand]:
    """Heuristic parser to classify a shell-like command into high-level actions.

    Returns a list of ParsedCommand variants that describe user intent.
    This is a lightweight, cross-platform approximation of the Rust parser.
    """
    if not argv:
        return [ParsedCommandUnknown(cmd="")]

    cmd = argv[0]
    args = argv[1:]

    if cmd in _READ_CMDS:
        # cat <path>
        path = next((a for a in args if not _is_flag(a)), "")
        return [ParsedCommandRead(cmd=" ".join(argv), name=os.path.basename(path) if path else "", path=path)]

    if cmd in _LIST_CMDS:
        path = next((a for a in args if not _is_flag(a)), None)
        return [ParsedCommandListFiles(cmd=" ".join(argv), path=path)]

    if cmd in _SEARCH_CMDS:
        # grep/rg/findstr <query> [path]
        query: Optional[str] = None
        path: Optional[str] = None
        non_flags = [a for a in args if not _is_flag(a)]
        if non_flags:
            query = non_flags[0]
            if len(non_flags) > 1:
                path = non_flags[1]
        return [ParsedCommandSearch(cmd=" ".join(argv), query=query, path=path)]

    return [ParsedCommandUnknown(cmd=" ".join(argv))]
