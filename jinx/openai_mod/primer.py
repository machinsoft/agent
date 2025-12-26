from __future__ import annotations

from jinx.config import jinx_tag
from jinx.prompts import get_prompt


async def build_header_and_tag(
    prompt_override: str | None = None,
    *,
    # Modular prompt flags
    has_code_task: bool = True,
    has_complex_reasoning: bool = False,
    has_embeddings: bool = False,
    is_error_recovery: bool = False,
    task_count: int = 0,
) -> tuple[str, str]:
    """Build instruction header and return it with a code tag identifier.

    Returns (header_plus_prompt, code_tag_id).
    If ``prompt_override`` is provided, that prompt name is used instead of the global default.
    Uses modular prompt system to minimize token usage.
    """
    fid, _ = jinx_tag()
    from jinx.config import neon_stat, PROMPT_NAME

    chaos = neon_stat()
    header = (
        f"\npulse: 1\nkey: {fid}\nos: {chaos['os']}\narch: {chaos['arch']}\nhost: {chaos['host']}\nuser: {chaos['user']}\n"
    )
    
    active_name = (prompt_override or PROMPT_NAME)
    
    # Use modular prompt for burning_logic
    if active_name == "burning_logic":
        from jinx.prompts.burning_logic import build_modular_prompt
        prompt = build_modular_prompt(
            has_code_task=has_code_task,
            has_complex_reasoning=has_complex_reasoning,
            has_embeddings=has_embeddings,
            is_error_recovery=is_error_recovery,
            task_count=task_count,
        )
    else:
        # Fallback to registered prompts
        prompt = get_prompt(active_name)
    
    # Fill template variables like {key}
    prompt = prompt.format(key=fid)
    return header + prompt, fid


async def build_modular_header(
    *,
    has_code_task: bool = False,
    has_complex_reasoning: bool = False,
    has_runtime_usage: bool = False,
    has_file_edits: bool = False,
    has_embeddings: bool = False,
    is_error_recovery: bool = False,
    is_architecture_mode: bool = False,
    task_count: int = 0,
) -> tuple[str, str]:
    """Build minimal modular header based on task requirements."""
    fid, _ = jinx_tag()
    from jinx.config import neon_stat
    
    chaos = neon_stat()
    header = f"pulse: 1\nkey: {fid}\nos: {chaos['os']}\n"
    
    from jinx.prompts.burning_logic import build_modular_prompt
    prompt = build_modular_prompt(
        has_code_task=has_code_task,
        has_complex_reasoning=has_complex_reasoning,
        has_runtime_usage=has_runtime_usage,
        has_file_edits=has_file_edits,
        has_embeddings=has_embeddings,
        is_error_recovery=is_error_recovery,
        is_architecture_mode=is_architecture_mode,
        task_count=task_count,
    ).format(key=fid)
    
    return header + prompt, fid
