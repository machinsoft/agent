from __future__ import annotations

import re
import secrets
from typing import Tuple, Optional

_GROUP_RE = re.compile(r"^\s*\[#group:([A-Za-z0-9_\-:.]{1,64})\]\s*")
_ID_RE = re.compile(r"^\s*\[#id:([A-Za-z0-9_\-:.]{1,128})\]\s*")


def _gen_id() -> str:
    # 12 urlsafe chars ~ 72 bits entropy, trimmed of '-' and '_' for tag safety
    tok = secrets.token_urlsafe(9)
    return re.sub(r"[^A-Za-z0-9]", "", tok)


def split_tags(s: str) -> tuple[Optional[str], Optional[str], str]:
    """Extract (id, group, body) from a tagged message string.
    Tags are only recognized at the start in the order: optional [#group:..], optional [#id:..].
    """
    if not s:
        return None, None, ""
    rest = s.lstrip()
    group: Optional[str] = None
    m = _GROUP_RE.match(rest)
    if m:
        group = (m.group(1) or "").strip() or None
        rest = rest[m.end():]
    msg_id: Optional[str] = None
    m2 = _ID_RE.match(rest)
    if m2:
        msg_id = (m2.group(1) or "").strip() or None
        rest = rest[m2.end():]
    return msg_id, group, rest.lstrip()


def inject_message_id(s: str, msg_id: str) -> str:
    """Insert [#id:..] after an optional leading [#group:..] tag, else at start."""
    if not s:
        return f"[#id:{msg_id}]"
    rest = s
    m = _GROUP_RE.match(rest)
    if m:
        # Insert after the group tag
        head = rest[: m.end()]
        tail = rest[m.end():]
        return f"{head}[#id:{msg_id}] {tail.lstrip()}"
    return f"[#id:{msg_id}] {rest.lstrip()}"


def wrap_with_tags(group: str, msg_id: str, body: str) -> str:
    body = (body or "").lstrip()
    return f"[#group:{group}][#id:{msg_id}] {body}"


def ensure_message_id(s: str) -> str:
    """Ensure the string carries a message ID tag.
    Preserves an existing group tag and inserts the id after it.
    """
    mid, _g, _body = split_tags(s)
    if mid:
        return s
    return inject_message_id(s, _gen_id())


def child_id(parent_id: str | None, index: int) -> str:
    base = parent_id or _gen_id()
    return f"{base}:{index}"
