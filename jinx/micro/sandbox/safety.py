from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

from jinx.micro.protocol.file_change import FileChange, FileChangeAdd, FileChangeDelete, FileChangeUpdate
from jinx.micro.protocol.v2 import SandboxPolicy, SandboxPolicyReadOnly, SandboxPolicyDangerFullAccess, SandboxPolicyWorkspaceWrite

SafetyCheck = Literal["autoApprove", "askUser", "reject"]


@dataclass
class ApplyPatchAction:
    changes: Dict[str, FileChange]

    def is_empty(self) -> bool:
        return not self.changes


def _is_path_writable(abs_path: Path, sandbox_policy: SandboxPolicy, cwd: Path) -> bool:
    if isinstance(sandbox_policy, SandboxPolicyDangerFullAccess):
        return True
    if isinstance(sandbox_policy, SandboxPolicyReadOnly):
        return False
    if isinstance(sandbox_policy, SandboxPolicyWorkspaceWrite):
        roots = list(getattr(sandbox_policy, "writable_roots", []) or [])
        # If no roots specified, treat cwd as the workspace root
        candidates = [cwd] + [Path(r) if not Path(r).is_absolute() else Path(r) for r in roots]
        try:
            abs_path = abs_path.resolve()
        except Exception:
            pass
        for root in candidates:
            try:
                root_abs = (root if root.is_absolute() else (cwd / root)).resolve()
            except Exception:
                root_abs = (cwd / root)
            try:
                if abs_path == root_abs or str(abs_path).startswith(str(root_abs) + "\\") or str(abs_path).startswith(str(root_abs) + "/"):
                    return True
            except Exception:
                continue
    return False


def _write_patch_constrained_to_writable_paths(action: ApplyPatchAction, sandbox_policy: SandboxPolicy, cwd: Path) -> bool:
    for path_str, change in action.changes.items():
        p = Path(path_str)
        abs_p = p if p.is_absolute() else (cwd / p)
        if isinstance(change, (FileChangeAdd, FileChangeDelete)):
            if not _is_path_writable(abs_p, sandbox_policy, cwd):
                return False
        elif isinstance(change, FileChangeUpdate):
            if not _is_path_writable(abs_p, sandbox_policy, cwd):
                return False
            if change.move_path:
                dest = Path(change.move_path)
                dest_abs = dest if dest.is_absolute() else (cwd / dest)
                if not _is_path_writable(dest_abs, sandbox_policy, cwd):
                    return False
    return True


def assess_patch_safety(action: ApplyPatchAction, approval_policy: str, sandbox_policy: SandboxPolicy, cwd: str) -> tuple[SafetyCheck, dict]:
    """Assess whether a patch can be auto‑approved under the current sandbox.

    Returns (decision, meta) where decision is one of: "autoApprove" | "askUser" | "reject".
    """
    if action.is_empty():
        return "reject", {"reason": "empty patch"}

    policy = (approval_policy or "").strip()
    cwd_path = Path(cwd)

    # Allow auto-approve if writes are constrained to writable roots or approval is on failure
    if _write_patch_constrained_to_writable_paths(action, sandbox_policy, cwd_path) or policy == "onFailure":
        if isinstance(sandbox_policy, SandboxPolicyDangerFullAccess):
            return "autoApprove", {"sandboxType": "none", "userApproved": False}
        return "autoApprove", {"sandboxType": "platform", "userApproved": False}

    if policy == "never":
        return "reject", {"reason": "writing outside workspace rejected by approval settings"}

    return "askUser", {}
