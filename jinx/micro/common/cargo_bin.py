from __future__ import annotations

import os
from pathlib import Path
from shutil import which
from typing import Optional


def _home_dir() -> Path:
    h = os.path.expanduser("~")
    return Path(h) if h else Path.cwd()


def _cargo_home() -> Path:
    env = os.getenv("CARGO_HOME")
    if env:
        return Path(env)
    home = _home_dir()
    return home / ".cargo"


def resolve_binary(name: str, fallback: Optional[str] = None) -> Path:
    cand = which(name)
    if cand:
        return Path(cand)
    ch = _cargo_home() / "bin" / name
    if os.name == "nt" and not ch.suffix:
        exts = os.getenv("PATHEXT", ".EXE;.BAT;.CMD").split(";")
        for ext in exts:
            p = ch.with_suffix(ext.lower())
            if p.exists():
                return p
    if ch.exists():
        return ch
    if fallback:
        cand2 = which(fallback)
        if cand2:
            return Path(cand2)
    raise FileNotFoundError(name)
