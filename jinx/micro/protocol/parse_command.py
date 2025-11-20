from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ParsedCommandRead:
    type: Literal["read"] = "read"
    cmd: str = ""
    name: str = ""
    path: str = ""


@dataclass
class ParsedCommandListFiles:
    type: Literal["listFiles"] = "listFiles"
    cmd: str = ""
    path: Optional[str] = None


@dataclass
class ParsedCommandSearch:
    type: Literal["search"] = "search"
    cmd: str = ""
    query: Optional[str] = None
    path: Optional[str] = None


@dataclass
class ParsedCommandUnknown:
    type: Literal["unknown"] = "unknown"
    cmd: str = ""


ParsedCommand = (
    ParsedCommandRead | ParsedCommandListFiles | ParsedCommandSearch | ParsedCommandUnknown
)
