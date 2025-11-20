from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal

from jinx.micro.protocol.config_types import SandboxMode
from jinx.micro.protocol.v2 import SandboxPolicy
from jinx.micro.protocol.models import (
    ContentItemInputText,
    ResponseItem,
    ResponseItemMessage,
)

ENVIRONMENT_CONTEXT_OPEN_TAG = "<environment_context>"
ENVIRONMENT_CONTEXT_CLOSE_TAG = "</environment_context>"

NetworkAccess = Literal["restricted", "enabled"]


@dataclass
class EnvironmentContext:
    cwd: Optional[str]
    approval_policy: Optional[str]
    sandbox_mode: Optional[SandboxMode]
    network_access: Optional[NetworkAccess]
    writable_roots: Optional[List[str]]
    shell: Optional[str] = None

    @staticmethod
    def new(
        cwd: Optional[str],
        approval_policy: Optional[str],
        sandbox_policy: Optional[SandboxPolicy],
        shell: Optional[str] = None,
    ) -> "EnvironmentContext":
        if sandbox_policy and getattr(sandbox_policy, "type", None) == "dangerFullAccess":
            sandbox_mode: Optional[SandboxMode] = "danger-full-access"
            network: Optional[NetworkAccess] = "enabled"
            roots: Optional[List[str]] = None
        elif sandbox_policy and getattr(sandbox_policy, "type", None) == "readOnly":
            sandbox_mode = "read-only"
            network = "restricted"
            roots = None
        elif sandbox_policy and getattr(sandbox_policy, "type", None) == "workspaceWrite":
            sandbox_mode = "workspace-write"
            network = "enabled" if bool(getattr(sandbox_policy, "network_access", False)) else "restricted"
            roots = list(getattr(sandbox_policy, "writable_roots", []) or [])
        else:
            sandbox_mode = None
            network = None
            roots = None
        return EnvironmentContext(
            cwd=cwd,
            approval_policy=approval_policy,
            sandbox_mode=sandbox_mode,
            network_access=network,
            writable_roots=roots,
            shell=shell,
        )

    def serialize_to_xml(self) -> str:
        lines: List[str] = [ENVIRONMENT_CONTEXT_OPEN_TAG]
        if self.cwd:
            lines.append(f"  <cwd>{self.cwd}</cwd>")
        if self.approval_policy:
            lines.append(f"  <approval_policy>{self.approval_policy}</approval_policy>")
        if self.sandbox_mode:
            lines.append(f"  <sandbox_mode>{self.sandbox_mode}</sandbox_mode>")
        if self.network_access:
            lines.append(f"  <network_access>{self.network_access}</network_access>")
        if self.writable_roots:
            lines.append("  <writable_roots>")
            for root in self.writable_roots:
                lines.append(f"    <root>{root}</root>")
            lines.append("  </writable_roots>")
        if self.shell:
            lines.append(f"  <shell>{self.shell}</shell>")
        lines.append(ENVIRONMENT_CONTEXT_CLOSE_TAG)
        return "\n".join(lines)

    def as_response_item(self) -> ResponseItem:
        return ResponseItemMessage(
            role="user",
            content=[ContentItemInputText(text=self.serialize_to_xml())],
        )
