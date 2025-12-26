from __future__ import annotations

import os
import subprocess
from pathlib import Path


def create_symlink(src: str | Path, dst: str | Path) -> None:
    src_p = Path(src)
    dst_p = Path(dst)

    if dst_p.exists() or dst_p.is_symlink():
        try:
            if dst_p.is_symlink() and Path(os.readlink(dst_p)) == src_p:
                return
        except Exception:
            pass
        raise FileExistsError(str(dst_p))

    dst_p.parent.mkdir(parents=True, exist_ok=True)

    is_dir = False
    try:
        is_dir = src_p.is_dir()
    except Exception:
        is_dir = False

    if os.name == "nt":
        try:
            os.symlink(src_p, dst_p, target_is_directory=is_dir)
            return
        except OSError:
            try:
                if is_dir:
                    subprocess.check_call(["cmd", "/c", "mklink", "/J", str(dst_p), str(src_p)])
                else:
                    subprocess.check_call(["cmd", "/c", "mklink", str(dst_p), str(src_p)])
                return
            except Exception as e:
                raise e
    else:
        os.symlink(src_p, dst_p)
