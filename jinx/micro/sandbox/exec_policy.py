from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal

ApprovalState = Literal["skip", "needsApproval", "forbidden"]


@dataclass
class ApprovalRequirement:
    state: ApprovalState
    reason: Optional[str] = None


class Policy:
    """Lightweight placeholder for exec policy. Always permissive by default."""

    @staticmethod
    def empty() -> "Policy":
        return Policy()

    def check(self, _command: List[str]) -> Optional[ApprovalRequirement]:
        return None


async def exec_policy_for(_features: object, _codex_home: str) -> Policy:
    """Return an in-memory policy object (placeholder)."""
    return Policy.empty()


def create_approval_requirement_for_command(
    _policy: Policy,
    command: List[str],
    approval_policy: str,
    _sandbox_policy: object,
    _sandbox_permissions: object = None,
) -> ApprovalRequirement:
    """Heuristic approval gate tuned for autonomous operation.

    Rules (minimal):
    - If approval_policy == 'never' → skip approval.
    - Otherwise, allow benign read/list/search commands; require approval for high‑risk verbs.
    """
    ap = (approval_policy or "").lower()
    if ap == "never":
        return ApprovalRequirement("skip")

    # Very rough heuristic on the first token
    cmd = (command[0] if command else "").lower()
    benign = {"cat", "type", "more", "less", "rg", "grep", "findstr", "dir", "ls"}
    risky = {"rm", "del", "rmdir", "mv", "move", "chmod", "chown", "mkfs", "diskpart"}

    if cmd in benign:
        return ApprovalRequirement("skip")
    if cmd in risky:
        return ApprovalRequirement("needsApproval", reason="potentially destructive command")

    # Default permissive
    return ApprovalRequirement("skip")
