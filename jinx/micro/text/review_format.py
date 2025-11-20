from __future__ import annotations

from typing import Any, Iterable, List, Optional

__all__ = [
    "format_review_findings_block",
]


def _get(obj: Any, *path: str) -> Any:
    cur = obj
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return cur


def _format_location(item: Any) -> str:
    p = _get(item, "code_location", "absolute_file_path")
    start = _get(item, "code_location", "line_range", "start")
    end = _get(item, "code_location", "line_range", "end")
    return f"{p}:{start}-{end}"


def format_review_findings_block(findings: Iterable[Any], selection: Optional[Iterable[bool]] = None) -> str:
    lines: List[str] = []
    items = list(findings or [])
    lines.append("")
    # Header
    if len(items) > 1:
        lines.append("Full review comments:")
    else:
        lines.append("Review comment:")

    flags = list(selection) if selection is not None else None

    for idx, item in enumerate(items):
        lines.append("")
        title = _get(item, "title") or ""
        location = _format_location(item)
        if flags is not None:
            checked = flags[idx] if idx < len(flags) else True
            marker = "[x]" if checked else "[ ]"
            lines.append(f"- {marker} {title} — {location}")
        else:
            lines.append(f"- {title} — {location}")

        body = _get(item, "body") or ""
        for body_line in str(body).splitlines():
            lines.append(f"  {body_line}")

    return "\n".join(lines)
