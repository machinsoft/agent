from __future__ import annotations

from typing import List, Optional

from .items import (
    AgentMessageContentText,
    AgentMessageItem,
    ReasoningItem,
    TurnItem,
    UserMessageItem,
    WebSearchItem,
)
from .models import (
    ContentItemInputImage,
    ContentItemInputText,
    ContentItemOutputText,
    ReasoningItemContent,
    ReasoningItemContentReasoningText,
    ReasoningItemReasoningSummary,
    ResponseItem,
    ResponseItemMessage,
    ResponseItemReasoning,
    ResponseItemWebSearchCall,
    WebSearchActionSearch,
)
from .v2 import UserInput


def _is_session_prefix(text: str) -> bool:
    t = text.lstrip().lower()
    return t.startswith("<environment_context>")


def _is_user_shell_command_text(text: str) -> bool:
    return "<user_shell_command>" in text


def _parse_user_message(message: List[object]) -> Optional[UserMessageItem]:
    content: List[UserInput] = []
    for c in message:
        if isinstance(c, ContentItemInputText):
            if _is_session_prefix(c.text) or _is_user_shell_command_text(c.text):
                return None
            content.append(UserInput.text_msg(c.text))
        elif isinstance(c, ContentItemInputImage):
            content.append(UserInput.image_url(c.image_url))
        elif isinstance(c, ContentItemOutputText):
            if _is_session_prefix(c.text):
                return None
            # ignore output text in user message
        else:
            # unknown content item -> ignore
            pass
    return UserMessageItem.new(content)


def _parse_agent_message(id_opt: Optional[str], message: List[object]) -> AgentMessageItem:
    content: List[AgentMessageContentText] = []
    for c in message:
        if isinstance(c, ContentItemOutputText):
            content.append(AgentMessageContentText(text=c.text))
        else:
            # ignore unexpected content in agent message
            pass
    return AgentMessageItem(id=id_opt or __import__("uuid").uuid4().hex, content=content)


def parse_turn_item(item: ResponseItem) -> Optional[TurnItem]:
    if isinstance(item, ResponseItemMessage):
        if item.role == "user":
            um = _parse_user_message(item.content)
            return None if um is None else um
        if item.role == "assistant":
            return _parse_agent_message(item.id, item.content)
        return None

    if isinstance(item, ResponseItemReasoning):
        summary_text: List[str] = []
        for s in item.summary:
            if isinstance(s, ReasoningItemReasoningSummary):
                # only variant is summary_text
                summary_text.append(s.text)
        raw_content: List[str] = []
        for entry in (item.content or []):
            if isinstance(entry, ReasoningItemContentReasoningText):
                raw_content.append(entry.text)
            elif isinstance(entry, ReasoningItemContent):
                # includes plain text variant
                raw_content.append(entry.text)  # type: ignore[attr-defined]
        return ReasoningItem(id=item.id, summary_text=summary_text, raw_content=raw_content)

    if isinstance(item, ResponseItemWebSearchCall):
        if isinstance(item.action, WebSearchActionSearch):
            return WebSearchItem(id=item.id or "", query=item.action.query)
        return None

    return None
