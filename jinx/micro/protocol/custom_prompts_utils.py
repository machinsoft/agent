from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple
from pathlib import Path

from .custom_prompts import CustomPrompt


def default_prompts_dir() -> Optional[Path]:
    """Return default prompts dir.

    No external envs: assume "./prompts" relative to current working dir.
    Returns None if the path does not exist.
    """
    p = Path.cwd() / "prompts"
    return p if p.exists() else None


def _parse_frontmatter(content: str) -> Tuple[Optional[str], Optional[str], str]:
    lines = content.splitlines(keepends=True)
    if not lines:
        return None, None, ""
    if lines[0].strip() != "---":
        return None, None, content

    desc: Optional[str] = None
    hint: Optional[str] = None
    consumed = len(lines[0])

    for seg in lines[1:]:
        trimmed = seg.strip()
        consumed += len(seg)
        if trimmed == "---":
            body = content[consumed:] if consumed < len(content) else ""
            return desc, hint, body
        if not trimmed or trimmed.startswith('#'):
            continue
        if ":" in trimmed:
            k, v = trimmed.split(":", 1)
            key = k.strip().lower()
            val = v.strip()
            if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                val = val[1:-1]
            if key == "description":
                desc = val
            elif key in ("argument-hint", "argument_hint"):
                hint = val
    # Unterminated frontmatter → treat as body
    return None, None, content


async def discover_prompts_in(dir_path: Path) -> List[CustomPrompt]:
    return await discover_prompts_in_excluding(dir_path, exclude=set())


async def discover_prompts_in_excluding(dir_path: Path, *, exclude: Set[str]) -> List[CustomPrompt]:
    out: List[CustomPrompt] = []
    try:
        entries = list(dir_path.iterdir())
    except Exception:
        return out

    for entry in entries:
        try:
            if not entry.is_file():
                continue
            if entry.suffix.lower() != ".md":
                continue
            name = entry.stem
            if name in exclude:
                continue
            try:
                content = entry.read_text(encoding="utf-8")
            except Exception:
                # Skip non-UTF8
                continue
            desc, hint, body = _parse_frontmatter(content)
            out.append(
                CustomPrompt(
                    name=name,
                    path=str(entry),
                    content=body,
                    description=desc,
                    argument_hint=hint,
                )
            )
        except Exception:
            continue

    out.sort(key=lambda p: p.name)
    return out
