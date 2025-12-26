from __future__ import annotations

import getpass
import subprocess
from typing import Optional


def current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return ""


def user_sid(user: Optional[str] = None) -> Optional[str]:
    try:
        args = ["whoami", "/user"] if user is None else ["wmic", "useraccount", f"where name='{user}'", "get", "sid"]
        cp = subprocess.run(args, capture_output=True, text=True, check=False)
        out = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
        sid = None
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if user is None and (" S-1-" in line):
                parts = line.split()
                sid = parts[-1] if parts else None
                break
            if user is not None and line.startswith("S-1-"):
                sid = line
                break
        return sid
    except Exception:
        return None
