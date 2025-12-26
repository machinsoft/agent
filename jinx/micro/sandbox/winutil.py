from __future__ import annotations

import os
import subprocess
from pathlib import Path


def create_junction(src: str | Path, dst: str | Path) -> None:
    src_p = Path(src)
    dst_p = Path(dst)
    if os.name != "nt":
        raise OSError("junctions are supported on Windows only")
    if dst_p.exists() or dst_p.is_symlink():
        raise FileExistsError(str(dst_p))
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    if not src_p.exists():
        raise FileNotFoundError(str(src_p))
    subprocess.check_call(["cmd", "/c", "mklink", "/J", str(dst_p), str(src_p)])
