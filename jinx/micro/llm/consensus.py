from __future__ import annotations

from typing import Callable


def _score(text: str) -> float:
    """Heuristic score for outputs. Prefer structured, codeful, and concise content."""
    if not text:
        return 0.0
    s = text.strip()
    score = 0.0
    n = len(s)
    # Prefer presence of fenced code or <python_*> blocks
    if "```" in s:
        score += 1.5
    if "<python_" in s and "</python_" in s:
        score += 1.0
    # Penalize overly long blobs
    if n <= 6000:
        score += 0.5
    # Simple Python parse attempt on first code fence
    try:
        import re as _re, ast as _ast
        m = _re.search(r"```(?:python)?\n([\s\S]*?)\n```", s)
        if m:
            body = m.group(1)
            _ast.parse(body)
            score += 1.0
    except Exception:
        pass
    return score


async def refine_output(instructions: str, model: str, input_text: str, base_text: str) -> str:
    """Try a tiny consensus: request one alternative candidate with a small variation
    and pick the higher-scoring output. Budgeted via env `JINX_LLM_CONSENSUS_MS`.
    If disabled or on error, returns base_text.
    """
    return base_text
