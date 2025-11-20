from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from jinx.bootstrap import package
from jinx.micro.common.cache import LruCache, sha1_digest
try:
    from PIL import Image  # type: ignore
except Exception:
    # Install Pillow to provide the PIL module, then import
    package("Pillow")
    from PIL import Image  # type: ignore

# Bounds mirror Rust constants
MAX_WIDTH: int = 2048
MAX_HEIGHT: int = 768


@dataclass
class EncodedImage:
    bytes: bytes
    mime: str
    width: int
    height: int

    def into_data_url(self) -> str:
        encoded = base64.b64encode(self.bytes).decode("ascii")
        return f"data:{self.mime};base64,{encoded}"


# Content-addressable image cache by SHA-1 digest
_IMAGE_CACHE: LruCache[bytes, EncodedImage] = LruCache(32)


def _format_to_mime(fmt: Optional[str]) -> str:
    if (fmt or "").upper() == "JPEG":
        return "image/jpeg"
    return "image/png"


def _encode_image(img, preferred: Optional[str]) -> Tuple[bytes, str]:
    target = (preferred or "PNG").upper()
    if target not in ("PNG", "JPEG", "JPG"):
        target = "PNG"
    mime = _format_to_mime("JPEG" if target in ("JPEG", "JPG") else "PNG")

    buf = io.BytesIO()
    if target in ("JPEG", "JPG"):
        # Convert to RGB for JPEG (no alpha)
        enc = img.convert("RGB")
        enc.save(buf, format="JPEG", quality=85, optimize=True)
    else:
        # PNG preferred to preserve lossless content
        enc = img.convert("RGBA")
        enc.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), mime


def _read_file_bytes(path: Path) -> bytes:
    # Blocking read; call async wrapper for event-loop friendly usage
    return path.read_bytes()


def load_and_resize_to_fit(path: str | Path) -> EncodedImage:
    p = Path(path)
    data = _read_file_bytes(p)
    key = sha1_digest(data)

    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        return cached

    # Detect format via Pillow
    with Image.open(io.BytesIO(data)) as img:
        fmt = (img.format or "").upper() or None
        width, height = img.size

        if width <= MAX_WIDTH and height <= MAX_HEIGHT and fmt in ("PNG", "JPEG", "JPG"):
            result = EncodedImage(bytes=data, mime=_format_to_mime(fmt), width=width, height=height)
            _IMAGE_CACHE.insert(key, result)
            return result

        # Resize and/or re-encode
        target_fmt = fmt or "PNG"
        resized = img.copy()
        resized.thumbnail((MAX_WIDTH, MAX_HEIGHT))
        out_bytes, mime = _encode_image(resized, preferred=target_fmt)
        result = EncodedImage(bytes=out_bytes, mime=mime, width=resized.width, height=resized.height)
        _IMAGE_CACHE.insert(key, result)
        return result


async def load_and_resize_to_fit_async(path: str | Path) -> EncodedImage:
    import asyncio

    return await asyncio.to_thread(load_and_resize_to_fit, path)
