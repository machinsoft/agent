from __future__ import annotations

import os
import difflib
from typing import Optional, Tuple, List

from jinx.async_utils.fs import read_text_raw, write_text
from jinx.micro.embeddings.search_cache import search_project_cached
from .utils import (
    unified_diff,
    syntax_check_enabled,
    detect_eol,
    has_trailing_newline,
    join_lines,
    normalize_indentation,
    leading_ws,
)
import ast
from jinx.micro.embeddings.project_config import resolve_project_root as _resolve_root
import asyncio
from jinx.micro.common.env import truthy


def _get_root() -> str:
    # Use the same robust resolver as embeddings pipeline to ensure consistency.
    return _resolve_root()


def _rel_to_root(path: str) -> str:
    try:
        root = _get_root()
        p = os.path.relpath(path, root)
    except Exception:
        p = path
    return p.replace("\\", "/")


async def _best_hit_in_file(path: str, query: str, rep_lines: List[str], *, topk: int, margin: int, tol: float) -> Tuple[int, int] | None:
    rel = _rel_to_root(path)
    hits = await search_project_cached(query, k=max(1, topk), max_time_ms=400)
    # Prefer hits that are in the same file
    candidates = [h for h in (hits or []) if str(h.get("file") or "").replace("\\", "/") == rel]
    if not candidates:
        return None
    # Score by fuzzy similarity between rep_lines and the window (offloaded)
    cur = await read_text_raw(path)
    lines = (cur or "").splitlines()
    def _score_candidates() -> tuple[Optional[tuple[int,int]], float]:
        best_local = None
        best_score_local = -1.0
        for h in candidates:
            ls = int(h.get("line_start") or 1)
            le = int(h.get("line_end") or ls)
            ls0 = max(1, ls - margin)
            le0 = min(len(lines), le + margin)
            win = lines[ls0 - 1 : le0]
            score = difflib.SequenceMatcher(None, "\n".join(win), "\n".join(rep_lines)).ratio()
            if score > best_score_local:
                best_score_local = score
                best_local = (ls0, le0)
        return best_local, best_score_local
    best, best_score = await asyncio.to_thread(_score_candidates)
    if best is None:
        return None
    if best_score < tol and truthy("JINX_SEMANTIC_PATCH_STRICT", "0"):
        return None
    return best


async def patch_semantic_in_file(path: str, query: str, replacement: str, *, preview: bool = False, topk: Optional[int] = None, margin: Optional[int] = None, tol: Optional[float] = None) -> Tuple[bool, str]:
    """Patch a large file by locating the best in-file window via embeddings, then replacing it.

    - Falls back to fuzzy window if embeddings are not available.
    - Preserves EOL and trailing newline; auto-normalizes indentation to the target window's left indent.
    """
    cur = await read_text_raw(path)
    if cur == "":
        return False, "file read error or empty"
    eol = detect_eol(cur)
    trailing_nl = has_trailing_newline(cur)
    lines = cur.splitlines()
    rep_lines = (replacement or "").splitlines()
    # Pick defaults
    k = int(topk) if topk is not None else 5
    mg = int(margin) if margin is not None else 6
    tl = float(tol) if tol is not None else 0.55

    # Try embedding-guided window
    loc = None
    try:
        if truthy("JINX_SEMANTIC_PATCH_ENABLE", "1") and query:
            loc = await _best_hit_in_file(path, query, rep_lines, topk=k, margin=mg, tol=tl)
    except Exception:
        loc = None

    # Fallback: fuzzy match by sliding window roughly the size of rep_lines (offloaded)
    if loc is None:
        m = max(1, len(rep_lines))
        def _best_window() -> tuple[int, float]:
            best_i_local = -1
            best_r_local = -1.0
            for i in range(0, len(lines) - m + 1):
                win = lines[i : i + m]
                r = difflib.SequenceMatcher(None, "\n".join(win), "\n".join(rep_lines)).ratio()
                if r > best_r_local:
                    best_r_local = r
                    best_i_local = i
            return best_i_local, best_r_local
        best_i, best_r = await asyncio.to_thread(_best_window)
        if best_i >= 0:
            loc = (best_i + 1, best_i + m)
        else:
            return False, "no suitable window"

    ls0, le0 = loc
    # Align indentation of replacement to the window's first line
    base_indent = leading_ws(lines[ls0 - 1])
    rep_norm = normalize_indentation(rep_lines)
    rep_aligned = [(base_indent + ln) if ln.strip() else ln for ln in rep_norm]

    out_lines = lines[: ls0 - 1] + rep_aligned + lines[le0 :]
    out = join_lines(out_lines, eol, trailing_nl)
    if preview:
        return True, unified_diff(cur, out, path=path)
    if str(path).endswith(".py") and syntax_check_enabled():
        try:
            await asyncio.to_thread(ast.parse, out or "")
        except Exception as e:
            return False, f"syntax error: {e}"
    await write_text(path, out)
    return True, unified_diff(cur, out, path=path)
