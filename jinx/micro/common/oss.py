from __future__ import annotations

import os
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
    """Return default model name for known OSS providers using env overrides.

    Environment overrides:
      - JINX_OSS_LMSTUDIO_DEFAULT_MODEL
      - JINX_OSS_OLLAMA_DEFAULT_MODEL
    Unknown providers -> None.
    """
    pid = _norm(provider_id)
    if pid in ("lmstudio", "lmstudio-oss"):
        return os.getenv("JINX_OSS_LMSTUDIO_DEFAULT_MODEL")
    if pid in ("ollama", "ollama-oss"):
        return os.getenv("JINX_OSS_OLLAMA_DEFAULT_MODEL")
    return None


async def ensure_oss_provider_ready(provider_id: str, config: Any | None = None) -> None:
    """Best-effort readiness hook.

    For Jinx, readiness is delegated to runtime configuration; this is a no-op
    that completes successfully to avoid blocking startup.
    """
    _ = (_norm(provider_id), config)
    return None
