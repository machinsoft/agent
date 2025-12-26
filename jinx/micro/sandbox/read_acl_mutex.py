from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Optional

_NAME = "Local\\CodexSandboxReadAcl"
_MUTEX_ALL_ACCESS = 0x001F0001


class ReadAclMutexGuard:
    def __init__(self, handle: int) -> None:
        self._h = wintypes.HANDLE(handle)

    def release(self) -> None:
        if getattr(self, "_h", None) and self._h.value:
            ctypes.windll.kernel32.ReleaseMutex(self._h)
            ctypes.windll.kernel32.CloseHandle(self._h)
            self._h = wintypes.HANDLE(0)

    def __enter__(self) -> "ReadAclMutexGuard":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def _wstr(s: str):
    return ctypes.c_wchar_p(s)


def read_acl_mutex_exists() -> bool:
    h = ctypes.windll.kernel32.OpenMutexW(_MUTEX_ALL_ACCESS, 0, _wstr(_NAME))
    if not h:
        err = ctypes.GetLastError()
        if err == 2:
            return False
        return False
    ctypes.windll.kernel32.CloseHandle(h)
    return True


def acquire_read_acl_mutex() -> Optional[ReadAclMutexGuard]:
    h = ctypes.windll.kernel32.CreateMutexW(None, 1, _wstr(_NAME))
    if not h:
        return None
    err = ctypes.GetLastError()
    if err == 183:
        ctypes.windll.kernel32.CloseHandle(h)
        return None
    return ReadAclMutexGuard(h)
