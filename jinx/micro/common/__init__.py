from __future__ import annotations
 
# Lightweight re-exports for common micro-modules
import importlib
from typing import Any
 
__all__ = [
    "format_elapsed",
    "format_duration",
    "format_env_display",
    "create_config_summary_entries",
    "parse_approval_mode",
    "parse_sandbox_mode",
    "builtin_approval_presets",
    "ApprovalPreset",
    "parse_overrides",
    "apply_on_value",
    "UserNotifier",
    "UserNotification",
    "TokenData",
    "IdTokenInfo",
    "parse_id_token",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "format_elapsed": ("jinx.micro.common.elapsed", "format_elapsed"),
    "format_duration": ("jinx.micro.common.elapsed", "format_duration"),
    "format_env_display": ("jinx.micro.common.format_env_display", "format_env_display"),
    "create_config_summary_entries": ("jinx.micro.common.config_summary", "create_config_summary_entries"),
    "parse_approval_mode": ("jinx.micro.common.cli_args", "parse_approval_mode"),
    "parse_sandbox_mode": ("jinx.micro.common.cli_args", "parse_sandbox_mode"),
    "builtin_approval_presets": ("jinx.micro.common.approval_presets", "builtin_approval_presets"),
    "ApprovalPreset": ("jinx.micro.common.approval_presets", "ApprovalPreset"),
    "parse_overrides": ("jinx.micro.common.config_override", "parse_overrides"),
    "apply_on_value": ("jinx.micro.common.config_override", "apply_on_value"),
    "UserNotifier": ("jinx.micro.common.user_notification", "UserNotifier"),
    "UserNotification": ("jinx.micro.common.user_notification", "UserNotification"),
    "TokenData": ("jinx.micro.common.token_data", "TokenData"),
    "IdTokenInfo": ("jinx.micro.common.token_data", "IdTokenInfo"),
    "parse_id_token": ("jinx.micro.common.token_data", "parse_id_token"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    mod = importlib.import_module(mod_name)
    val = getattr(mod, attr)
    globals()[name] = val
    return val
