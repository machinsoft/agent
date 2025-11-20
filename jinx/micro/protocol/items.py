from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal
import uuid

from .v2 import UserInput


# --- Agent message content ---
@dataclass
class AgentMessageContentText:
    type: Literal["text"] = "text"
    text: str = ""


AgentMessageContent = AgentMessageContentText


# --- Items ---
@dataclass
class UserMessageItem:
    id: str
    content: List[UserInput]

    @staticmethod
    def new(content: List[UserInput]) -> "UserMessageItem":
        return UserMessageItem(id=str(uuid.uuid4()), content=list(content))

    def message(self) -> str:
        parts: List[str] = []
        for c in self.content:
            if getattr(c, "type", None) == "text" and getattr(c, "text", ""):
                parts.append(c.text or "")
        return "".join(parts)

    def image_urls(self) -> List[str]:
        urls: List[str] = []
        for c in self.content:
            if getattr(c, "type", None) == "image" and getattr(c, "url", ""):
                urls.append(c.url or "")
        return urls


@dataclass
class AgentMessageItem:
    id: str
    content: List[AgentMessageContent]

    @staticmethod
    def new(content: List[AgentMessageContent]) -> "AgentMessageItem":
        return AgentMessageItem(id=str(uuid.uuid4()), content=list(content))


@dataclass
class ReasoningItem:
    id: str
    summary_text: List[str] = field(default_factory=list)
    raw_content: List[str] = field(default_factory=list)


@dataclass
class WebSearchItem:
    id: str
    query: str


TurnItem = UserMessageItem | AgentMessageItem | ReasoningItem | WebSearchItem
