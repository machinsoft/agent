from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HistoryEntry:
    conversation_id: str
    ts: int
    text: str
