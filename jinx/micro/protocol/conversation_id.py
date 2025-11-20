from __future__ import annotations

from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class ConversationId:
    value: str

    @staticmethod
    def new() -> "ConversationId":
        # Use UUID v4 for portability; v7 not guaranteed in stdlib
        return ConversationId(str(uuid.uuid4()))

    @staticmethod
    def from_string(s: str) -> "ConversationId":
        # Validate format; raises ValueError if invalid
        _ = uuid.UUID(s)
        return ConversationId(s)

    def __str__(self) -> str:
        return self.value
