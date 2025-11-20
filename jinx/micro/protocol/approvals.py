from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal

from .file_change import FileChange
from .parse_command import (
    ParsedCommand,
    ParsedCommandRead,
    ParsedCommandListFiles,
    ParsedCommandSearch,
    ParsedCommandUnknown,
)
from .v2 import SandboxCommandAssessment


SandboxRiskLevel = Literal["low", "medium", "high"]


@dataclass
class ExecApprovalRequestEvent:
    call_id: str
    turn_id: str
    command: List[str]
    cwd: str
    reason: Optional[str]
    risk: Optional[SandboxCommandAssessment]
    parsed_cmd: List[ParsedCommand]


@dataclass
class ApplyPatchApprovalRequestEvent:
    call_id: str
    turn_id: str
    changes: Dict[str, FileChange]
    reason: Optional[str] = None
    grant_root: Optional[str] = None
