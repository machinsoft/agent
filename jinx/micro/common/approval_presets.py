from __future__ import annotations

from dataclasses import dataclass
from typing import List

__all__ = ["ApprovalPreset", "builtin_approval_presets"]


@dataclass(frozen=True)
class ApprovalPreset:
    id: str
    label: str
    description: str
    approval: str  # 'unless_trusted' | 'on_failure' | 'on_request' | 'never'
    sandbox: str   # 'read-only' | 'workspace-write' | 'danger-full-access'


def builtin_approval_presets() -> List[ApprovalPreset]:
    return [
        ApprovalPreset(
            id="read-only",
            label="Read Only",
            description="Requires approval to edit files and run commands.",
            approval="on_request",
            sandbox="read-only",
        ),
        ApprovalPreset(
            id="auto",
            label="Agent",
            description="Read and edit files, and run commands.",
            approval="on_request",
            sandbox="workspace-write",
        ),
        ApprovalPreset(
            id="full-access",
            label="Agent (full access)",
            description=(
                "Can edit files outside this workspace and run commands with network access. "
                "Exercise caution when using."
            ),
            approval="never",
            sandbox="danger-full-access",
        ),
    ]
