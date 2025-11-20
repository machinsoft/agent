from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple
import fnmatch
import os


@dataclass
class EnvironmentVariablePattern:
    pattern: str
    case_insensitive: bool = True

    @staticmethod
    def new_case_insensitive(pat: str) -> "EnvironmentVariablePattern":
        return EnvironmentVariablePattern(pattern=pat, case_insensitive=True)

    def matches(self, name: str) -> bool:
        if self.case_insensitive:
            return fnmatch.fnmatch(name.upper(), self.pattern.upper())
        return fnmatch.fnmatch(name, self.pattern)


@dataclass
class ShellEnvironmentPolicy:
    inherit: str = "core"  # 'all' | 'none' | 'core'
    ignore_default_excludes: bool = False
    exclude: List[EnvironmentVariablePattern] = field(default_factory=list)
    include_only: List[EnvironmentVariablePattern] = field(default_factory=list)
    set_vars: Dict[str, str] = field(default_factory=dict)


_CORE_VARS = {"HOME", "LOGNAME", "PATH", "SHELL", "USER", "USERNAME", "TMPDIR", "TEMP", "TMP"}
_DEFAULT_EXCLUDES = [
    EnvironmentVariablePattern.new_case_insensitive("*KEY*"),
    EnvironmentVariablePattern.new_case_insensitive("*SECRET*"),
    EnvironmentVariablePattern.new_case_insensitive("*TOKEN*"),
]


def _iter_env() -> Iterable[Tuple[str, str]]:
    # Snapshot environment; avoid dynamic changes during iteration
    return list(os.environ.items())


def _matches_any(name: str, patterns: List[EnvironmentVariablePattern]) -> bool:
    return any(p.matches(name) for p in patterns)


def create_env(policy: ShellEnvironmentPolicy) -> Dict[str, str]:
    # Step 1: inherit strategy
    if policy.inherit == "all":
        env_map: Dict[str, str] = {k: v for k, v in _iter_env()}
    elif policy.inherit == "none":
        env_map = {}
    else:  # core
        env_map = {k: v for k, v in _iter_env() if k in _CORE_VARS}

    # Step 2: default excludes
    if not policy.ignore_default_excludes:
        env_map = {k: v for k, v in env_map.items() if not _matches_any(k, _DEFAULT_EXCLUDES)}

    # Step 3: custom excludes
    if policy.exclude:
        env_map = {k: v for k, v in env_map.items() if not _matches_any(k, policy.exclude)}

    # Step 4: overrides (set)
    if policy.set_vars:
        env_map.update(policy.set_vars)

    # Step 5: include_only filter
    if policy.include_only:
        env_map = {k: v for k, v in env_map.items() if _matches_any(k, policy.include_only)}

    return env_map
