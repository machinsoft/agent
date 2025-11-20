"""Backend client data models for Codex Cloud Tasks responses.

This module defines light-weight dataclasses and helpers to parse backend
responses used by the Codex client. Kept minimal and self-contained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StructuredContent:
    content_type: Optional[str] = None
    text: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any] | None) -> "StructuredContent | None":
        if not d or not isinstance(d, dict):
            return None
        return StructuredContent(
            content_type=d.get("content_type"),
            text=d.get("text"),
        )


@dataclass
class ContentFragment:
    # One of: structured(text) or raw text
    structured: Optional[StructuredContent] = None
    text: Optional[str] = None

    @staticmethod
    def from_json(v: Any) -> "ContentFragment":
        if isinstance(v, str):
            return ContentFragment(text=v)
        if isinstance(v, dict):
            return ContentFragment(structured=StructuredContent.from_dict(v))
        return ContentFragment()

    def text_value(self) -> Optional[str]:
        if self.text is not None:
            t = self.text.strip()
            return t or None
        if self.structured and (self.structured.content_type or "").lower() == "text":
            t = (self.structured.text or "").strip()
            return t or None
        return None


@dataclass
class DiffPayload:
    diff: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any] | None) -> "DiffPayload | None":
        if not d or not isinstance(d, dict):
            return None
        return DiffPayload(diff=d.get("diff"))


@dataclass
class TurnItem:
    kind: str = ""
    role: Optional[str] = None
    content: List[ContentFragment] = field(default_factory=list)
    diff: Optional[str] = None
    output_diff: Optional[DiffPayload] = None

    @staticmethod
    def from_dict(d: Dict[str, Any] | None) -> "TurnItem":
        if not d or not isinstance(d, dict):
            return TurnItem()
        content = d.get("content")
        fragments: List[ContentFragment] = []
        if isinstance(content, list):
            fragments = [ContentFragment.from_json(x) for x in content]
        return TurnItem(
            kind=str(d.get("type", "")),
            role=d.get("role"),
            content=fragments,
            diff=d.get("diff"),
            output_diff=DiffPayload.from_dict(d.get("output_diff")),
        )

    def text_values(self) -> List[str]:
        out: List[str] = []
        for f in self.content:
            tv = f.text_value()
            if tv:
                out.append(tv)
        return out

    def diff_text(self) -> Optional[str]:
        if self.kind == "output_diff":
            if self.diff:
                d = self.diff.strip()
                if d:
                    return d
        elif self.kind == "pr" and self.output_diff and self.output_diff.diff:
            d = (self.output_diff.diff or "").strip()
            if d:
                return d
        return None


@dataclass
class WorklogContent:
    parts: List[ContentFragment] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any] | None) -> "WorklogContent | None":
        if not d or not isinstance(d, dict):
            return None
        parts_raw = d.get("parts")
        parts: List[ContentFragment] = []
        if isinstance(parts_raw, list):
            parts = [ContentFragment.from_json(x) for x in parts_raw]
        return WorklogContent(parts=parts)


@dataclass
class Author:
    role: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any] | None) -> "Author | None":
        if not d or not isinstance(d, dict):
            return None
        return Author(role=d.get("role"))


@dataclass
class WorklogMessage:
    author: Optional[Author] = None
    content: Optional[WorklogContent] = None

    @staticmethod
    def from_dict(d: Dict[str, Any] | None) -> "WorklogMessage":
        if not d or not isinstance(d, dict):
            return WorklogMessage()
        return WorklogMessage(
            author=Author.from_dict(d.get("author")),
            content=WorklogContent.from_dict(d.get("content")),
        )

    def is_assistant(self) -> bool:
        role = (self.author.role if self.author else None) or ""
        return role.lower() == "assistant"

    def text_values(self) -> List[str]:
        content = self.content
        if not content:
            return []
        out: List[str] = []
        for part in content.parts:
            tv = part.text_value()
            if tv:
                out.append(tv)
        return out


@dataclass
class Worklog:
    messages: List[WorklogMessage] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any] | None) -> "Worklog | None":
        if not d or not isinstance(d, dict):
            return None
        msgs_raw = d.get("messages")
        msgs: List[WorklogMessage] = []
        if isinstance(msgs_raw, list):
            msgs = [WorklogMessage.from_dict(x) for x in msgs_raw]
        return Worklog(messages=msgs)


@dataclass
class TurnError:
    code: Optional[str] = None
    message: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any] | None) -> "TurnError | None":
        if not d or not isinstance(d, dict):
            return None
        return TurnError(code=d.get("code"), message=d.get("message"))

    def summary(self) -> Optional[str]:
        code = (self.code or "").strip()
        msg = (self.message or "").strip()
        if not code and not msg:
            return None
        if code and msg:
            return f"{code}: {msg}"
        return code or msg


@dataclass
class Turn:
    id: Optional[str] = None
    attempt_placement: Optional[int] = None
    turn_status: Optional[str] = None
    sibling_turn_ids: List[str] = field(default_factory=list)
    input_items: List[TurnItem] = field(default_factory=list)
    output_items: List[TurnItem] = field(default_factory=list)
    worklog: Optional[Worklog] = None
    error: Optional[TurnError] = None

    @staticmethod
    def from_dict(d: Dict[str, Any] | None) -> "Turn | None":
        if not d or not isinstance(d, dict):
            return None
        def parse_items(key: str) -> List[TurnItem]:
            raw = d.get(key)
            if isinstance(raw, list):
                return [TurnItem.from_dict(x) for x in raw]
            return []
        return Turn(
            id=d.get("id"),
            attempt_placement=d.get("attempt_placement"),
            turn_status=d.get("turn_status"),
            sibling_turn_ids=[str(x) for x in (d.get("sibling_turn_ids") or []) if isinstance(x, str)],
            input_items=parse_items("input_items"),
            output_items=parse_items("output_items"),
            worklog=Worklog.from_dict(d.get("worklog")),
            error=TurnError.from_dict(d.get("error")),
        )

    def unified_diff(self) -> Optional[str]:
        for item in self.output_items:
            d = item.diff_text()
            if d:
                return d
        return None

    def message_texts(self) -> List[str]:
        out: List[str] = []
        for item in self.output_items:
            if item.kind == "message":
                out.extend(item.text_values())
        if self.worklog:
            for msg in self.worklog.messages:
                if msg.is_assistant():
                    out.extend(msg.text_values())
        return out

    def user_prompt(self) -> Optional[str]:
        parts: List[str] = []
        for item in self.input_items:
            if item.kind == "message" and ((item.role or "user").lower() == "user"):
                parts.extend(item.text_values())
        if not parts:
            return None
        return "\n\n".join(parts)

    def error_summary(self) -> Optional[str]:
        return self.error.summary() if self.error else None


@dataclass
class CodeTaskDetailsResponse:
    current_user_turn: Optional[Turn] = None
    current_assistant_turn: Optional[Turn] = None
    current_diff_task_turn: Optional[Turn] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CodeTaskDetailsResponse":
        return CodeTaskDetailsResponse(
            current_user_turn=Turn.from_dict(d.get("current_user_turn")),
            current_assistant_turn=Turn.from_dict(d.get("current_assistant_turn")),
            current_diff_task_turn=Turn.from_dict(d.get("current_diff_task_turn")),
        )

    # Ext methods
    def unified_diff(self) -> Optional[str]:
        for turn in (self.current_diff_task_turn, self.current_assistant_turn):
            if turn:
                d = turn.unified_diff()
                if d:
                    return d
        return None

    def assistant_text_messages(self) -> List[str]:
        out: List[str] = []
        for turn in (self.current_diff_task_turn, self.current_assistant_turn):
            if turn:
                out.extend(turn.message_texts())
        return out

    def user_text_prompt(self) -> Optional[str]:
        return self.current_user_turn.user_prompt() if self.current_user_turn else None

    def assistant_error_message(self) -> Optional[str]:
        return self.current_assistant_turn.error_summary() if self.current_assistant_turn else None


@dataclass
class RateLimitWindow:
    used_percent: Optional[float] = None
    window_minutes: Optional[int] = None
    resets_at: Optional[int] = None


@dataclass
class CreditsSnapshot:
    has_credits: Optional[bool] = None
    unlimited: Optional[bool] = None
    balance: Optional[float] = None


@dataclass
class RateLimitSnapshot:
    primary: Optional[RateLimitWindow] = None
    secondary: Optional[RateLimitWindow] = None
    credits: Optional[CreditsSnapshot] = None


@dataclass
class TurnAttemptsSiblingTurnsResponse:
    sibling_turns: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TurnAttemptsSiblingTurnsResponse":
        raw = d.get("sibling_turns")
        if isinstance(raw, list):
            return TurnAttemptsSiblingTurnsResponse(sibling_turns=[x for x in raw if isinstance(x, dict)])
        return TurnAttemptsSiblingTurnsResponse()
