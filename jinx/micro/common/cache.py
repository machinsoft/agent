from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import RLock
from typing import Callable, Generic, Hashable, Iterator, MutableMapping, Optional, Tuple, TypeVar

__all__ = [
    "LruCache",
    "sha1_digest",
]

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def sha1_digest(data: bytes) -> bytes:
    """Return 20-byte SHA-1 digest for ``data``.

    Mirrors Rust's [u8; 20] output semantics.
    """
    h = hashlib.sha1()
    h.update(data)
    return h.digest()  # 20 bytes


class LruCache(Generic[K, V]):
    """Minimal LRU cache with thread-safe operations.

    - Fixed "capacity" (>0). Oldest entries evicted on insert overflow.
    - All operations are O(1) on average using OrderedDict.
    - Methods return plain Python values; callers retain ownership.
    """

    __slots__ = ("_cap", "_data", "_lock")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._cap: int = int(capacity)
        self._data: "OrderedDict[K, V]" = OrderedDict()
        self._lock = RLock()

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key, last=True)
            return self._data[key]

    def insert(self, key: K, value: V) -> Optional[V]:
        with self._lock:
            old = self._data.pop(key, None)
            self._data[key] = value
            if len(self._data) > self._cap:
                self._data.popitem(last=False)  # evict LRU
            return old

    def remove(self, key: K) -> Optional[V]:
        with self._lock:
            return self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def get_or_insert_with(self, key: K, factory: Callable[[], V]) -> V:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key, last=True)
                return self._data[key]
            v = factory()
            self._data[key] = v
            if len(self._data) > self._cap:
                self._data.popitem(last=False)
            return v

    def get_or_try_insert_with(self, key: K, factory: Callable[[], V]) -> V:
        # Same as get_or_insert_with; exceptions bubble through
        return self.get_or_insert_with(key, factory)

    def with_mut(self, callback: Callable[[MutableMapping[K, V]], V]) -> V:
        with self._lock:
            return callback(self._data)

    def __len__(self) -> int:  # pragma: no cover
        with self._lock:
            return len(self._data)

    def __contains__(self, key: object) -> bool:  # pragma: no cover
        with self._lock:
            return key in self._data
