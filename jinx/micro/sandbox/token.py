from __future__ import annotations

import os
import ctypes
from ctypes import wintypes
from typing import Optional


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def integrity_level() -> Optional[int]:
    try:
        advapi = ctypes.windll.advapi32
        kernel = ctypes.windll.kernel32
        h = wintypes.HANDLE()
        TOKEN_QUERY = 0x0008
        if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(h)):
            return None
        try:
            TokenIntegrityLevel = 25
            size = wintypes.DWORD(0)
            advapi.GetTokenInformation(h, TokenIntegrityLevel, None, 0, ctypes.byref(size))
            buf = ctypes.create_string_buffer(size.value)
            if not advapi.GetTokenInformation(h, TokenIntegrityLevel, buf, size, ctypes.byref(size)):
                return None
            class SID_AND_ATTRIBUTES(ctypes.Structure):
                _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]
            sia = SID_AND_ATTRIBUTES.from_buffer(buf)
            sub_auth_count = ctypes.cast(sia.Sid, ctypes.POINTER(ctypes.c_ubyte))[1]
            sub_auth = ctypes.cast(sia.Sid, ctypes.POINTER(ctypes.c_ulong))[sub_auth_count]
            return int(sub_auth)
        finally:
            kernel.CloseHandle(h)
    except Exception:
        return None
