from __future__ import annotations

from typing import Any, Dict, List


def truthy_env(name: str, default: str = "0") -> bool:
    v = default
    return str(v).lower() not in ("", "0", "false", "off", "no")


def extract_tagged_block(text: str, key: str, block: str) -> str:
    """Extract content between <{block}_{key}> and </{block}_{key}>. Returns empty on failure."""
    if not text or not key:
        return ""
    start_tag = f"<{block}_{key}>"
    end_tag = f"</{block}_{key}>"
    i = text.find(start_tag)
    if i == -1:
        return ""
    j = text.find(end_tag, i + len(start_tag))
    if j == -1:
        return ""
    return text[i + len(start_tag) : j].strip()


def parse_planner_block(body: str) -> Dict[str, Any]:
    """Parse line-based planner schema from a <machine_{key}> block.

    Recognized keys:
      goal:
      plan.N.step / plan.N.why / plan.N.criteria (N=1..3)
      sub.M (M=1..2)
      risk.R (R=1..3)
      note:
      cortex.* (optional persona hints)
      advisory extras (optional, advisory mode):
        advice.do.N, advice.avoid.N, clarify.N, reminder.N, assume.N, context.N
    """
    # Env-tunable limits (counts only; lengths are free-form)
    max_plan = 3
    max_subs = 2
    max_risks = 3
    # Advisory extras limits
    max_adv = 4
    max_clr = 4
    max_rem = 4
    max_ctx = 4
    max_asm = 4

    goal = ""
    note = ""
    subs: List[str] = []
    risks: List[str] = []
    plan_steps: Dict[int, Dict[str, str]] = {}
    cortex: Dict[str, str] = {}
    advice_do: List[str] = []
    advice_avoid: List[str] = []
    clarifiers: List[str] = []
    reminders: List[str] = []
    assumptions: List[str] = []
    contexts: List[str] = []
    for raw in (body or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # key: value
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if k == "goal" or k == "need":
            if not goal:
                goal = v
            continue
        if k == "note":
            if not note:
                note = v
            continue
        if k.startswith("cortex."):
            # capture optional cortex.* lines verbatim
            field = k.split(".", 1)[1] if "." in k else ""
            if field and field not in cortex:
                cortex[field] = v
            continue
        if k.startswith("sub."):
            if len(subs) < max_subs and v:
                subs.append(v)
            continue
        if k.startswith("risk."):
            if len(risks) < max_risks and v:
                risks.append(v)
            continue
        # Advisory extras
        if k.startswith("advice.do."):
            if len(advice_do) < max_adv and v:
                advice_do.append(v)
            continue
        if k.startswith("advice.avoid."):
            if len(advice_avoid) < max_adv and v:
                advice_avoid.append(v)
            continue
        if k.startswith("clarify."):
            if len(clarifiers) < max_clr and v:
                clarifiers.append(v)
            continue
        if k.startswith("reminder."):
            if len(reminders) < max_rem and v:
                reminders.append(v)
            continue
        if k.startswith("assume."):
            if len(assumptions) < max_asm and v:
                assumptions.append(v)
            continue
        if k.startswith("context."):
            if len(contexts) < max_ctx and v:
                contexts.append(v)
            continue
        if k.startswith("plan."):
            # plan.N.field
            try:
                _, idx_s, field = k.split(".", 2)
                idx = int(idx_s)
            except Exception:
                continue
            if idx < 1 or idx > max_plan:
                continue
            field = field.strip()
            bucket = plan_steps.setdefault(idx, {"step": "", "why": "", "criteria": ""})
            if field in bucket and not bucket[field]:
                if field == "criteria":
                    bucket[field] = v
                elif field == "why":
                    bucket[field] = v
                elif field == "step":
                    bucket[field] = v
    # Normalize plan order
    plan_out: List[Dict[str, str]] = []
    for n in sorted(plan_steps.keys()):
        d = plan_steps[n]
        if any(d.values()):
            plan_out.append({
                "step": d.get("step", ""),
                "why": d.get("why", ""),
                "criteria": d.get("criteria", ""),
            })
    return {
        "goal": goal,
        "plan": plan_out[:max_plan],
        "sub_queries": subs[:max_subs],
        "risks": risks[:max_risks],
        "note": note,
        "cortex": cortex,
        # advisory extras
        "advice_do": advice_do,
        "advice_avoid": advice_avoid,
        "clarifiers": clarifiers,
        "reminders": reminders,
        "assumptions": assumptions,
        "context": contexts,
    }


def parse_reflection_block(body: str, *, advisory: bool = True) -> Dict[str, Any]:
    """Parse a reflection block body into a normalized dict.

    Recognizes:
      - summary: <...>
      - nudge.N: <...> (advisory mode)
      - next.N: <...> (directive mode)
    Returns {"summary": str, "next_actions": [str, ...], optional "mode": "advisory"}.
    """
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
            continue
        if advisory:
            if k.startswith("nudge.") and v and len(items) < 8:
                items.append(v)
        else:
            if k.startswith("next.") and v and len(items) < 8:
                items.append(v)
    res: Dict[str, Any] = {"summary": summary, "next_actions": items}
    if advisory:
        res["mode"] = "advisory"
    return res
