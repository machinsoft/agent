from __future__ import annotations

from typing import Any, List, Tuple

from jinx.micro.sandbox.summary import summarize_sandbox_policy

__all__ = ["create_config_summary_entries"]


def _get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def create_config_summary_entries(config: Any) -> List[Tuple[str, str]]:
    """Build key/value summary similar to Codex `create_config_summary_entries`.

    Accepts either an object with attributes or a dict-like config.
    """
    cwd = _get(config, "cwd")
    cwd_str = str(cwd) if cwd is not None else ""
    model = str(_get(config, "model", ""))
    provider_id = str(_get(config, "model_provider_id", ""))
    approval = str(_get(config, "approval_policy", ""))

    sandbox_policy = _get(config, "sandbox_policy", {})
    sandbox = summarize_sandbox_policy(sandbox_policy)

    entries: List[Tuple[str, str]] = [
        ("workdir", cwd_str),
        ("model", model),
        ("provider", provider_id),
        ("approval", approval),
        ("sandbox", sandbox),
    ]

    wire_api = str(_get(config, "model_provider.wire_api", "")).lower()
    supports_reasoning = bool(_get(config, "model_family.supports_reasoning_summaries", False))
    if wire_api == "responses" and supports_reasoning:
        effort = _get(config, "model_reasoning_effort") or _get(
            config, "model_family.default_reasoning_effort"
        )
        reasoning_effort = str(effort) if effort is not None else "none"
        entries.append(("reasoning effort", reasoning_effort))
        entries.append(("reasoning summaries", str(_get(config, "model_reasoning_summary", ""))))

    return entries
