from __future__ import annotations

import os
from typing import Iterable

# Try to ensure python-dotenv is available at runtime; if not, proceed with noop
dotenv = None  # type: ignore[assignment]


def load_env(paths: Iterable[str] | None = None) -> None:
    """Best-effort load of environment variables via python-dotenv.

    If python-dotenv is unavailable, this is a no-op.
    """
    def _iter_paths() -> list[str]:
        if paths:
            return [str(p) for p in paths if p]
        cands: list[str] = []
        try:
            cands.append(os.path.join(os.getcwd(), ".env"))
        except Exception:
            pass
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            cands.append(os.path.join(repo_root, ".env"))
        except Exception:
            pass
        return [p for p in cands if p and os.path.exists(p)]

    def _strip_quotes(v: str) -> str:
        t = (v or "").strip()
        if len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")):
            return t[1:-1]
        return t

    for p in _iter_paths():
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    k = (k or "").strip()
                    if not k:
                        continue
                    if os.environ.get(k):
                        continue
                    os.environ[k] = _strip_quotes(v)
        except Exception:
            pass
