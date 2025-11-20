from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

PROMPTS_CMD_PREFIX = "prompts"


@dataclass
class CustomPrompt:
    name: str
    path: str
    content: str
    description: Optional[str] = None
    argument_hint: Optional[str] = None
