from __future__ import annotations

from typing import Literal

# Mirrors codex-rs protocol config enums as string literal types.
# Keep values lowercase/kebab-case to match wire formats.

ReasoningEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]

ReasoningSummary = Literal[
    "auto",
    "concise",
    "detailed",
    "none",
]

Verbosity = Literal["low", "medium", "high"]

# Note: This is the legacy kebab-case variant used in some v1-style payloads.
SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]

ForcedLoginMethod = Literal["chatgpt", "api"]

TrustLevel = Literal["trusted", "untrusted"]
