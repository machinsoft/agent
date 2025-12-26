from __future__ import annotations

from typing import Any, Dict, Optional

from jinx.micro.llm.service import spark_openai
from jinx.micro.llm.chain_utils import truthy_env, extract_tagged_block, parse_planner_block, parse_reflection_block
from jinx.micro.llm.chain_evidence import collect_pre_evidence
from jinx.micro.llm.chain_trace import trace_plan
from jinx.micro.llm.chain_resilience import record_success, record_failure, save_last_plan


async def run_planner(user_text: str, *, max_subqueries: Optional[int] = None, planner_ms: Optional[int] = 400) -> Dict[str, Any]:
    """Enhanced planner with brain systems integration and ML optimization.

    Returns a dict like: {"sub_queries": [...], "note": "...", "ml_params": {...}}
    Uses brain systems for query enhancement and adaptive parameters.
    """
    txt = (user_text or "").strip()
    if not txt:
        return {"sub_queries": [], "note": "empty"}
    
    # Enhance query with brain systems BEFORE planning
    enhanced_txt = txt
    try:
        from jinx.micro.brain import expand_query
        expanded = await expand_query(txt)
        if expanded and expanded.confidence > 0.6:
            enhanced_txt = expanded.expanded
            await trace_plan({"phase": "enhanced", "confidence": expanded.confidence})
    except Exception:
        enhanced_txt = txt

    evid = await collect_pre_evidence(txt)
    planner_input = txt if not evid else (f"<user>\n{txt}\n</user>\n\n<evidence>\n{evid}\n</evidence>")
    # Continuity: inject a tiny continuity block (anchors) to stabilize planning
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
        cont_block = ("<continuity>\n" + "\n".join(lines) + "\n</continuity>") if lines else ""
    except Exception:
        cont_block = ""
    if cont_block:
        planner_input = (planner_input + "\n\n" + cont_block) if planner_input else cont_block
    # Inject plan_mode tag so a single combined prompt can switch schemas deterministically
    try:
        advisory = truthy_env("JINX_CHAINED_ADVISORY", "1")
    except Exception:
        advisory = True
    plan_mode_tag = f"<plan_mode>{'advisory' if advisory else 'directive'}</plan_mode>"
    planner_input = (planner_input + "\n\n" + plan_mode_tag) if planner_input else plan_mode_tag

    await trace_plan({"phase": "pre", "has_evidence": bool(evid)})
    try:
        # Use a single combined prompt with mode controlled by <plan_mode>
        out, tag = await spark_openai(planner_input, prompt_override="planner_advisorycombo")
    except Exception:
        await trace_plan({"phase": "plan", "error": "openai_error"})
        await record_failure("openai_error")
        return {"goal": "", "plan": [], "sub_queries": [], "risks": [], "note": "openai_error"}

    block = extract_tagged_block(out, tag, "machine")
    if not block:
        await trace_plan({"phase": "plan", "error": "no_machine_block"})
        await record_failure("no_machine_block")
        return {"goal": "", "plan": [], "sub_queries": [], "risks": [], "note": "no_machine"}

    data = parse_planner_block(block)
    # Optional reflection block if combined prompt was used
    try:
        rbody = extract_tagged_block(out, tag, "reflect")
    except Exception:
        rbody = ""
    if rbody:
        try:
            data["reflect"] = parse_reflection_block(rbody, advisory=advisory)
        except Exception:
            pass
    # Optional: extract reusable helper kernels
    try:
        kernels_code = extract_tagged_block(out, tag, "plan_kernels")
    except Exception:
        kernels_code = ""
    if kernels_code:
        data["kernels"] = kernels_code
    # Success path: record and persist last good plan
    try:
        await record_success()
        await save_last_plan(data)
    except Exception:
        pass
    await trace_plan({
        "phase": "plan",
        "subs": data.get("sub_queries", []),
        "plan_len": len(data.get("plan", [])),
        "risks_len": len(data.get("risks", [])),
    })
    return data
