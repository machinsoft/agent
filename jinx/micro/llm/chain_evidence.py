from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from jinx.micro.embeddings.retrieval import (
    retrieve_top_k as retrieve_dialogue_top_k,
)
from jinx.micro.embeddings.project_retrieval import (
    retrieve_project_top_k as retrieve_code_top_k,
)
from .chain_utils import truthy_env


async def collect_pre_evidence(query: str) -> str:
    """Return a tiny textual evidence block from embeddings for the planner input.

    Includes minimal dialogue/code hits with truncated previews and scores.
    Controlled by env flags with tiny budgets to keep latency low.
    """
    if not truthy_env("JINX_CHAINED_PRE_EVID", "1"):
        return ""
    q = (query or "").strip()
    if not q:
        return ""
    try:
        kd = 2
        kc = 2
        td = 100
        tc = 180
        # Continuity: append small hints to stabilize retrieval (optional, RT-safe)
        try:
            from jinx.micro.conversation.cont import load_last_anchors as _load_last_anchors
            anc = await _load_last_anchors()
            sym = (anc.get("symbols") or [])[:2]
            pth = (anc.get("paths") or [])[:1]
            hints = " ".join([x for x in (sym + pth) if x])
        except Exception:
            hints = ""
        qd = (q + " " + hints).strip() if hints else q
        qc = qd
        d_task = retrieve_dialogue_top_k(qd, k=kd, max_time_ms=td)
        c_task = retrieve_code_top_k(qc, k=kc, max_time_ms=tc)
        d_res, c_res = await asyncio.gather(d_task, c_task, return_exceptions=True)
        lines: List[str] = []
        if isinstance(d_res, list) and d_res:
            lines.append("dialogue:")
            for s, src, obj in d_res[:kd]:
                try:
                    pv = str((obj.get("meta", {}) or {}).get("text_preview") or "").strip()
                    pv = (pv[:120] + "…") if len(pv) > 120 else pv
                    if pv:
                        lines.append(f"- {str(src)[:80]} ({float(s):.2f}) — {pv}")
                    else:
                        lines.append(f"- {str(src)[:80]} ({float(s):.2f})")
                except Exception:
                    pass
        if isinstance(c_res, list) and c_res:
            lines.append("code:")
            for s, file_rel, obj in c_res[:kc]:
                try:
                    pv = str((obj.get("meta", {}) or {}).get("text_preview") or "").strip()
                    pv = (pv[:140] + "…") if len(pv) > 140 else pv
                    if pv:
                        lines.append(f"- {str(file_rel)[:120]} ({float(s):.2f}) — {pv}")
                    else:
                        lines.append(f"- {str(file_rel)[:120]} ({float(s):.2f})")
                except Exception:
                    pass
        return "\n".join(lines)
    except Exception:
        return ""


async def gather_planner_evidence(subs: List[str]) -> Dict[str, Any]:
    """Collect small, structured evidence for each sub-query without large payloads.

    Returns a dict: {"queries": [{"q": str, "dialogue": [...], "code": [...]}]}
    where each hit is {"score": float, "src": str}.
    """
    out: Dict[str, Any] = {"queries": []}
    if not subs:
        return out
    # Tight budgets
    kd = 3
    kc = 3
    td = 120
    tc = 220
    # Continuity: prepare lightweight shared hints once
    try:
        from jinx.micro.conversation.cont import load_last_anchors as _load_last_anchors
        anc = await _load_last_anchors()
        sym = (anc.get("symbols") or [])[:2]
        pth = (anc.get("paths") or [])[:1]
        hints = " ".join([x for x in (sym + pth) if x])
    except Exception:
        hints = ""

    for q in subs:
        try:
            qq = (q + " " + hints).strip() if hints else q
            d_task = retrieve_dialogue_top_k(qq, k=kd, max_time_ms=td)
            c_task = retrieve_code_top_k(qq, k=kc, max_time_ms=tc)
            d_res, c_res = await asyncio.gather(d_task, c_task, return_exceptions=True)
            d_list = []
            if isinstance(d_res, list):
                for s, src, obj in d_res[:kd]:
                    d_list.append({"score": float(s), "src": str(src)})
            c_list = []
            if isinstance(c_res, list):
                for s, file_rel, obj in c_res[:kc]:
                    c_list.append({"score": float(s), "src": str(file_rel)})
            out["queries"].append({"q": q, "dialogue": d_list, "code": c_list})
        except Exception:
            out["queries"].append({"q": q, "dialogue": [], "code": []})
        await asyncio.sleep(0)
    return out
