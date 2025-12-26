from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "ModelInfo",
    "get_model_info",
]


@dataclass(frozen=True)
class ModelInfo:
    context_window: int
    max_output_tokens: int
    auto_compact_token_limit: Optional[int]


def _auto_compact_default(context_window: int) -> int:
    return (int(context_window) * 9) // 10


def _mk(context_window: int, max_output_tokens: int) -> ModelInfo:
    return ModelInfo(
        context_window=int(context_window),
        max_output_tokens=int(max_output_tokens),
        auto_compact_token_limit=_auto_compact_default(int(context_window)),
    )


def get_model_info(model_slug: str) -> Optional[ModelInfo]:
    slug = (model_slug or "").strip()

    if slug == "o3":
        return _mk(200_000, 100_000)

    if slug == "o4-mini":
        return _mk(200_000, 100_000)

    if slug == "codex-mini-latest":
        return _mk(200_000, 100_000)

    if slug in ("gpt-4.1", "gpt-4.1-2025-04-14"):
        return _mk(1_047_576, 32_768)

    if slug in ("gpt-4o", "gpt-4o-2024-08-06"):
        return _mk(128_000, 16_384)

    if slug == "gpt-4o-2024-05-13":
        return _mk(128_000, 4_096)

    if slug == "gpt-4o-2024-11-20":
        return _mk(128_000, 16_384)

    if slug == "gpt-3.5-turbo":
        return _mk(16_385, 4_096)

    if slug.startswith("gpt-5-codex") or slug.startswith("gpt-5.1-codex") or slug.startswith("gpt-5.1-codex-max"):
        return _mk(272_000, 128_000)

    if slug.startswith("gpt-5"):
        return _mk(272_000, 128_000)

    if slug.startswith("codex-"):
        return _mk(272_000, 128_000)

    return None
