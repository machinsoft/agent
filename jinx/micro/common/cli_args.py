from __future__ import annotations

__all__ = [
    "parse_approval_mode",
    "parse_sandbox_mode",
]


_APPROVAL_MAP = {
    "untrusted": "unless_trusted",
    "on-failure": "on_failure",
    "onrequest": "on_request",
    "on-request": "on_request",
    "never": "never",
}


def parse_approval_mode(value: str) -> str:
    """Map CLI approval mode (kebab-case) to canonical policy string.

    Returns one of: 'unless_trusted' | 'on_failure' | 'on_request' | 'never'
    """
    key = (value or "").strip().lower()
    if key in _APPROVAL_MAP:
        return _APPROVAL_MAP[key]
    # Default: conservative ask policy
    return "unless_trusted"


_SANDBOX_MAP = {
    "read-only": "read-only",
    "workspace-write": "workspace-write",
    "danger-full-access": "danger-full-access",
}


def parse_sandbox_mode(value: str) -> str:
    """Map CLI sandbox mode to canonical string.

    Returns one of: 'read-only' | 'workspace-write' | 'danger-full-access'
    """
    key = (value or "").strip().lower()
    return _SANDBOX_MAP.get(key, "read-only")
