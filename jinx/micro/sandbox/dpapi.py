from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Optional


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.c_void_p)]


def _to_blob(data: bytes) -> tuple[_DATA_BLOB, ctypes.Array]:
    buf = (ctypes.c_ubyte * len(data))()
    if data:
        ctypes.memmove(buf, data, len(data))
    blob = _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.c_void_p))
    return blob, buf


def protect(data: bytes, *, entropy: Optional[bytes] = None, machine_scope: bool = False) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    in_blob, in_buf = _to_blob(bytes(data))
    out_blob = _DATA_BLOB()
    if entropy is not None:
        ent_blob, ent_buf = _to_blob(entropy)
        ent_ptr = ctypes.byref(ent_blob)
    else:
        ent_ptr = None
    flags = 0x01
    if machine_scope:
        flags |= 0x04
    ok = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        None,
        ent_ptr,
        None,
        None,
        flags,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise OSError(ctypes.GetLastError())
    try:
        out = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return out
    finally:
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)


def unprotect(blob: bytes, *, entropy: Optional[bytes] = None) -> bytes:
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError("blob must be bytes")
    in_blob, in_buf = _to_blob(bytes(blob))
    out_blob = _DATA_BLOB()
    if entropy is not None:
        ent_blob, ent_buf = _to_blob(entropy)
        ent_ptr = ctypes.byref(ent_blob)
    else:
        ent_ptr = None
    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        ent_ptr,
        None,
        None,
        0x01,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise OSError(ctypes.GetLastError())
    try:
        out = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return out
    finally:
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)
