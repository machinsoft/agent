from __future__ import annotations

import json
from typing import Any, Dict, List

from jinx.micro.llm.service import spark_openai
from .chain_utils import truthy_env, extract_tagged_block

_CHAINED_REFLECT_ON = False


async def run_reflector(user_text: str, plan: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Run a reflection pass that yields summary and next-actions in Jinx style.

    Returns: {"summary": str, "next_actions": [str, ...]}
    Gated by JINX_CHAINED_REFLECT.
    """
    if not _CHAINED_REFLECT_ON:
        return {}
    payload = {
        "user": (user_text or "")[:500],
        "plan": plan or {},
        "evidence": evidence or {},
    }
    txt = json.dumps(payload, ensure_ascii=False)
    # Continuity: add a tiny anchors block to guide reflection (language-agnostic)
    try:
        from jinx.micro.conversation.cont import load_last_anchors as _load_last_anchors
        anc = await _load_last_anchors()
    except Exception:
        anc = {}
    try:
        lines = []
        q = (anc.get("questions") or [])[:1]
        if q:
            lines.append(f"q: {q[0]}")
        sy = (anc.get("symbols") or [])[:3]
        if sy:
            lines.append("symbols: " + ", ".join(sy))
        pth = (anc.get("paths") or [])[:2]
        if pth:
            lines.append("paths: " + ", ".join(pth))
        cont_block = ("\n\n<continuity>\n" + "\n".join(lines) + "\n</continuity>") if lines else ""
    except Exception:
        cont_block = ""
    if cont_block:
        txt = txt + cont_block
    # Inject plan_mode tag so a single combined prompt can switch schemas deterministically
    is_adv = truthy_env("JINX_CHAINED_ADVISORY", "1")
    txt = txt + f"\n\n<plan_mode>{'advisory' if is_adv else 'directive'}</plan_mode>"
    out, tag = await spark_openai(txt, prompt_override="planner_advisorycombo")
    # Prefer <reflect_{key}> if present (combo prompt), otherwise fallback to <machine_{key}>
    body = extract_tagged_block(out, tag, "reflect") or extract_tagged_block(out, tag, "machine")
    # Parse lines: advisory => summary + nudge.N; otherwise summary + next.N
    summary = ""
    items: List[str] = []
    for raw in (body or "").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if k == "summary" and not summary:
            summary = v
        elif (k.startswith("nudge.") if is_adv else k.startswith("next.")) and len(items) < 5:
            if v:
                items.append(v)
    result = {"summary": summary, "next_actions": items}
    if is_adv:
        result["mode"] = "advisory"
    # Optional: extract reusable helper kernels from reflector output
    try:
        kernels_code = extract_tagged_block(out, tag, "plan_kernels")
    except Exception:
        kernels_code = ""
    if kernels_code:
        result["kernels"] = kernels_code
    return result
