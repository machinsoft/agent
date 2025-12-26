from __future__ import annotations

import asyncio
import threading
from typing import Any, Iterable, Optional

from jinx.bootstrap import ensure_optional
from jinx.micro.common.cache import LruCache

mods = ensure_optional(["tiktoken"])  # installs if missing
_tiktoken = mods["tiktoken"]


class _FallbackEncoding:
    def encode(self, text: str, allowed_special=()):
        # Simple, deterministic fallback: unicode codepoints.
        return [ord(ch) for ch in (text or "")]

    def decode(self, tokens: list[int]):
        try:
            return "".join(chr(int(t)) for t in tokens)
        except Exception:
            return ""


def _tiktoken_missing() -> bool:
    try:
        return bool(getattr(_tiktoken, "__jinx_optional_missing__", False))
    except Exception:
        return True

__all__ = [
    "EncodingKind",
    "Tokenizer",
    "warm_model_cache",
]


class EncodingKind:
    O200K_BASE = "o200k_base"
    CL100K_BASE = "cl100k_base"


# Cache encodings by model name
_MODEL_CACHE: LruCache[str, Any] = LruCache(64)


def _encoding_for_kind(kind: str):
    if _tiktoken_missing():
        return _FallbackEncoding()
    name = EncodingKind.O200K_BASE if kind == EncodingKind.O200K_BASE else EncodingKind.CL100K_BASE
    return _tiktoken.get_encoding(name)


def _encoding_for_model(model: str):
    if _tiktoken_missing():
        return _FallbackEncoding()
    try:
        return _tiktoken.encoding_for_model(model)
    except Exception:
        return _tiktoken.get_encoding(EncodingKind.O200K_BASE)


class Tokenizer:
    def __init__(self, enc: Any) -> None:
        self._enc = enc

    @classmethod
    def new(cls, kind: str) -> "Tokenizer":
        enc = _encoding_for_kind(kind)
        return cls(enc)

    @classmethod
    def try_default(cls) -> "Tokenizer":
        return cls.new(EncodingKind.O200K_BASE)

    @classmethod
    def for_model(cls, model: str) -> "Tokenizer":
        enc = _MODEL_CACHE.get_or_insert_with(model, lambda: _encoding_for_model(model))
        return cls(enc)

    def encode(self, text: str, with_special_tokens: bool) -> list[int]:
        if with_special_tokens:
            ids = self._enc.encode(text, allowed_special="all")
        else:
            ids = self._enc.encode(text, allowed_special=())
        return [int(t) for t in ids]

    def count(self, text: str) -> int:
        try:
            return int(len(self._enc.encode(text, allowed_special=())))
        except Exception:
            # Conservative fallback
            return len(text)

    def decode(self, tokens: Iterable[int]) -> str:
        try:
            return self._enc.decode([int(t) for t in tokens])
        except Exception as e:  # decode errors propagate as runtime error
            raise RuntimeError("decode error") from e


def warm_model_cache(model: str) -> None:
    """Pre-warm model encoding cache in background (best-effort)."""

    def _load() -> None:
        _MODEL_CACHE.get_or_insert_with(model, lambda: _encoding_for_model(model))

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(asyncio.to_thread(_load))
            return
    except Exception:
        pass
    threading.Thread(target=_load, daemon=True).start()
