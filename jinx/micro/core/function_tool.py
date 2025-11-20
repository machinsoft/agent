from __future__ import annotations

__all__ = [
    "FunctionCallError",
    "RespondToModel",
    "Denied",
    "MissingLocalShellCallId",
    "Fatal",
]


class FunctionCallError(Exception):
    """Base error for function tool invocations."""


class RespondToModel(FunctionCallError):
    def __init__(self, message: str):
        super().__init__(message)


class Denied(FunctionCallError):
    def __init__(self, message: str):
        super().__init__(message)


class MissingLocalShellCallId(FunctionCallError):
    def __init__(self):
        super().__init__("LocalShellCall without call_id or id")


class Fatal(FunctionCallError):
    def __init__(self, message: str):
        super().__init__(f"Fatal error: {message}")
