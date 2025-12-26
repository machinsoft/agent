from __future__ import annotations

import os
from typing import Callable, Dict

# Registry for prompt providers: name -> loader function returning the prompt string
_REGISTRY: Dict[str, Callable[[], str]] = {}


def register_prompt(name: str, loader: Callable[[], str]) -> None:
    name = name.strip().lower()
    _REGISTRY[name] = loader


def get_prompt(name: str | None = None) -> str:
    """Return the prompt text by name.

    If name is None, resolve from environment variable JINX_PROMPT (or PROMPT_NAME),
    defaulting to "burning_logic".
    """
    if not name:
        name = os.getenv("JINX_PROMPT") or os.getenv("PROMPT_NAME") or "burning_logic"
    key = name.strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown prompt '{name}'. Available: {available}")
    base = _REGISTRY[key]()
    try:
        include = (os.getenv("JINX_INCLUDE_SYSTEM_DESC", "1").strip().lower() in {"1","true","yes","on"})
    except Exception:
        include = True
    if not include:
        return base
    try:
        from .system_desc import get_system_description
        return base + "\n\n" + get_system_description()
    except Exception:
        return base


# Import built-ins to register them
from . import burning_logic  # noqa: F401
from . import chaos_bloom  # noqa: F401
from . import jinxed_blueprint  # noqa: F401
from . import memory_optimizer  # noqa: F401
from . import burning_logic_recovery  # noqa: F401
from . import planner_minjson  # noqa: F401
from . import planner_reflectjson  # noqa: F401
from . import planner_advisorycombo  # noqa: F401
from . import planner_synthesize  # noqa: F401
from . import planner_refine_embed  # noqa: F401
from . import architect_api  # noqa: F401
from . import consensus_alt  # noqa: F401
from . import consensus_judge  # noqa: F401
from . import cross_rerank  # noqa: F401
from . import state_compiler  # noqa: F401
from . import skill_acquire_spec  # noqa: F401
from . import reprogram_adversarial  # noqa: F401
from . import repair_suggest  # noqa: F401
from . import repair_stub  # noqa: F401
from . import code_audit  # noqa: F401
from . import multi_task  # noqa: F401


# --- Health check utilities ---
def list_prompts() -> list[str]:
    return sorted(_REGISTRY.keys())


def _schema_requirements() -> Dict[str, list[str]]:
    return {
        # planners/architects
        "planner_synthesize": ["goal", "embed_context", "topk", "example_json", "risk_text"],
        "planner_refine_embed": ["goal", "top_files_csv", "current_plan_json"],
        "architect_api": ["shape", "project_name", "candidate_resources_json", "request"],
        # runtime/state
        "state_compiler": ["board_json", "last_query", "memory_snippets", "evergreen"],
        # skills/repair/test
        "skill_acquire_spec": ["query", "suggested_path"],
        "repair_suggest": [
            "error_type", "error_message", "file_path", "line_number",
            "error_line", "context_before_text", "context_after_text", "containing_scope",
        ],
        "repair_stub": ["module"],
        "reprogram_adversarial": ["goal"],
        # consensus/rerank
        "consensus_judge": ["a", "b"],
        "cross_rerank": ["query", "candidate"],
        # persona/blueprints (no required slots)
        "burning_logic": [],
        "burning_logic_recovery": [],
        "planner_minjson": [],
        "planner_reflectjson": [],
        "planner_advisorycombo": [],
        "chaos_bloom": [],
        "jinxed_blueprint": [],
        "memory_optimizer": [],
    }


def validate_prompt_format(name: str, strict: bool = False) -> tuple[bool, str]:
    """Validate that the registered prompt can be formatted with required keys.

    Returns (ok, msg). In strict mode raises on failure.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        msg = f"missing prompt: {name}"
        if strict:
            raise KeyError(msg)
        return False, msg
    tmpl = _REGISTRY[key]()
    reqs = _schema_requirements().get(key, [])
    if not reqs:
        return True, "ok"
    # Build dummy kwargs
    data = {k: "X" for k in reqs}
    try:
        _ = tmpl.format(**data)
        return True, "ok"
    except Exception as e:
        msg = f"format error in '{name}': {e}"
        if strict:
            raise ValueError(msg)
        return False, msg


def validate_all_prompts(strict: bool = False) -> list[str]:
    """Validate all known prompts with their required placeholders.

    Returns a list of issues; empty when all good. In strict mode, raises on first failure.
    """
    issues: list[str] = []
    for nm in sorted(_schema_requirements().keys()):
        ok, msg = validate_prompt_format(nm, strict=strict)
        if not ok:
            issues.append(msg)
    return issues


# Optional auto-validation on import (env-gated)
try:
    import os as _os
    if (_os.getenv("JINX_PROMPTS_VALIDATE") or "").strip().lower() in {"1", "true", "on", "yes"}:
        _issues = validate_all_prompts(strict=False)
        # Best-effort warning to stderr to avoid crashing in RT
        if _issues:
            try:
                import sys as _sys
                _sys.stderr.write("[prompts] validation issues: " + "; ".join(_issues) + "\n")
            except Exception:
                pass
except Exception:
    pass


def render_prompt(name: str, /, **kwargs) -> str:
    """Return a formatted prompt string with optional ASCII enforcement.

    Reads JINX_PROMPTS_ASCII_ENFORCE to strip non-ASCII characters when set.
    Raises KeyError/ValueError on missing prompt or format failure.
    """
    tmpl = get_prompt(name)
    try:
        out = tmpl.format(**kwargs)
    except Exception as e:
        raise ValueError(f"prompt format error for '{name}': {e}")
    try:
        import os as _os
        if (_os.getenv("JINX_PROMPTS_ASCII_ENFORCE") or "").strip().lower() in {"1","true","on","yes"}:
            out = out.encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    return out
