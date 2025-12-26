from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "get_default_model_for_oss_provider",
    "ensure_oss_provider_ready",
]


def _norm(pid: str | None) -> str:
    if not pid:
        return ""
    return str(pid).strip().lower()


def get_default_model_for_oss_provider(provider_id: str) -> Optional[str]:
    # Local/OSS providers are not supported; OpenAI API only.
    _ = _norm(provider_id)
    return None


async def ensure_oss_provider_ready(provider_id: str, config: Any | None = None) -> None:
    """Best-effort readiness hook.

    For Jinx, readiness is delegated to runtime configuration; this is a no-op
    that completes successfully to avoid blocking startup.
    """
    _ = (_norm(provider_id), config)
    return None
