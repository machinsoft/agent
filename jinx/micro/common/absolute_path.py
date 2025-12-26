from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Optional, Union

_PathLike = Union[str, Path]

_BASE_PATH: ContextVar[Optional[Path]] = ContextVar("JINX_ABSOLUTE_PATH_BASE", default=None)


class AbsolutePath:
    def __init__(self, path: _PathLike) -> None:
        p = Path(path)
        if not p.is_absolute():
            raise ValueError("AbsolutePath requires an absolute path")
        self._p = p.resolve(strict=False)

    @staticmethod
    def resolve_against_base(path: _PathLike, base_path: _PathLike) -> "AbsolutePath":
        p = Path(path)
        base = Path(base_path)
        if not p.is_absolute():
            p = base.joinpath(p)
        return AbsolutePath(p)

    @staticmethod
    def from_absolute_path(path: _PathLike) -> "AbsolutePath":
        p = Path(path)
        if not p.is_absolute():
            raise ValueError("from_absolute_path expects an absolute path")
        return AbsolutePath(p)

    @staticmethod
    def current_dir() -> "AbsolutePath":
        return AbsolutePath(Path.cwd())

    def join(self, path: _PathLike) -> "AbsolutePath":
        return AbsolutePath.resolve_against_base(path, self._p)

    def parent(self) -> Optional["AbsolutePath"]:
        par = self._p.parent
        if par == self._p:
            return None
        return AbsolutePath.from_absolute_path(par)

    def as_path(self) -> Path:
        return self._p

    def to_path(self) -> Path:
        return Path(self._p)

    def to_string(self) -> str:
        return str(self._p)

    def __fspath__(self) -> str:
        return str(self._p)

    def __str__(self) -> str:
        return str(self._p)

    def __repr__(self) -> str:
        return f"AbsolutePath({self._p!s})"


def deserialize_absolute_path(value: _PathLike) -> AbsolutePath:
    p = Path(value)
    base = _BASE_PATH.get()
    if base is not None and not p.is_absolute():
        return AbsolutePath.resolve_against_base(p, base)
    if p.is_absolute():
        return AbsolutePath.from_absolute_path(p)
    raise ValueError("AbsolutePath deserialized without a base path")


@contextmanager
def absolute_path_base(base_path: _PathLike) -> Iterator[None]:
    token = _BASE_PATH.set(Path(base_path))
    try:
        yield None
    finally:
        _BASE_PATH.reset(token)
