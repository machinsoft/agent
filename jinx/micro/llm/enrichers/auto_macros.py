from __future__ import annotations

from typing import List
import re
from jinx.micro.text.heuristics import is_code_like as _is_code_like

_USE_DLG = True
_USE_PROJ = True
_DLG_K = 3
_PROJ_K = 3
_USE_MEM = True
_MEM_COMP_K = 8
_MEM_EVER_K = 8
_CODE_ON = True
_CODE_TOPK = 8


async def auto_context_lines(input_text: str) -> List[str]:
    """Return fallback context macro lines when unified context is unavailable.

    Produces dialogue/project/memory automacros with dynamic K and simple heuristics.
    """
    lines: List[str] = []
    txt = (input_text or "").strip()
    codey = _is_code_like(txt)
    # feature flags
    use_dlg = _USE_DLG
    use_proj = _USE_PROJ
    dlg_k = _DLG_K
    proj_k = _PROJ_K
    # Memory automacros
    use_mem = _USE_MEM
    mem_comp_k = _MEM_COMP_K
    mem_ever_k = _MEM_EVER_K

    # Heuristic preference: project for code-like, dialogue for natural text
    if use_dlg:
        if codey and not use_proj:
            lines.append(f"Context (dialogue): {{{{m:emb:dialogue:{dlg_k}}}}}")
        elif not codey:
            lines.append(f"Context (dialogue): {{{{m:emb:dialogue:{dlg_k}}}}}")
    if use_proj:
        if codey or not use_dlg:
            lines.append(f"Context (code): {{{{m:emb:project:{proj_k}}}}}")
    if use_mem:
        # Inject routed memory (pins + graph-aligned + ranker)
        lines.append(f"Memory (routed): {{{{m:memroute:{max(mem_comp_k, mem_ever_k)}}}}}")
    return lines


async def auto_code_lines(input_text: str) -> List[str]:
    """Return code intelligence macro lines (usage/def) inferred from input.

    Gate with JINX_AUTOMACRO_CODE (default ON). Extract the most salient token
    from code-like input or from the first callable pattern in text.
    """
    on = _CODE_ON
    if not on:
        return []
    txt = (input_text or "").strip()
    token = ""
    # Prefer last identifier before '(' in a simple assignment/call line
    # e.g., "tk = brain_topk(default_topk)" -> brain_topk
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", txt)
    if m:
        token = m.group(1)
    # fallback: longest identifier-like word
    if not token:
        ids = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", txt)
        ids = sorted(ids, key=len, reverse=True)
        if ids:
            token = ids[0]
    if not token or len(token) < 3:
        return []
    topk = _CODE_TOPK
    # Build usage + def lines; def is small by default
    lines = [
        f"Code usage: {{{{m:code:usage:{token}:{topk}}}}}",
        f"Code def: {{{{m:code:def:{token}:3}}}}",
    ]
    return lines


__all__ = ["auto_context_lines", "auto_code_lines"]
