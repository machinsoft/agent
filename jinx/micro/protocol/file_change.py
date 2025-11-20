from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class FileChangeAdd:
    type: Literal["add"] = "add"
    content: str = ""


@dataclass
class FileChangeDelete:
    type: Literal["delete"] = "delete"
    content: str = ""


@dataclass
class FileChangeUpdate:
    type: Literal["update"] = "update"
    unified_diff: str = ""
    move_path: Optional[str] = None


FileChange = FileChangeAdd | FileChangeDelete | FileChangeUpdate
