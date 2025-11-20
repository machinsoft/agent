from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

__all__ = ["extract_powershell_command"]

_ALLOWED_FLAGS = {"-nologo", "-noprofile", "-command", "-c"}


def _is_powershell(exe: str) -> bool:
    p = exe.strip().lower()
    name = Path(p).name
    return name in {"powershell", "powershell.exe", "pwsh"}


def extract_powershell_command(command: List[str]) -> Optional[Tuple[str, str]]:
    """Extract (shell, script) when the first arg is PowerShell and -Command/-c present.

    Mirrors Codex logic: reject unknown flags before the command body.
    """
    if len(command) < 3:
        return None
    shell = command[0]
    if not _is_powershell(shell):
        return None

    i = 1
    while i + 1 < len(command):
        flag = command[i]
        flag_l = flag.lower()
        if flag_l not in _ALLOWED_FLAGS:
            return None
        if flag_l in {"-command", "-c"}:
            script = command[i + 1]
            return shell, script
        i += 1
    return None
