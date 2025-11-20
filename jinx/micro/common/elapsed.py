from __future__ import annotations

import time
from datetime import timedelta
from typing import Union

__all__ = [
    "format_elapsed",
    "format_duration",
]


def _format_elapsed_millis(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    if ms < 60_000:
        return f"{ms / 1000.0:.2f}s"
    minutes = ms // 60_000
    seconds = (ms % 60_000) // 1000
    return f"{minutes}m {seconds:02d}s"


def format_duration(duration: Union[float, timedelta]) -> str:
    """Render duration as compact human string.

    - <1s => "{ms}ms"
    - <60s => "{sec:.2f}s"
    - >=60s => "{min}m {sec:02d}s"
    Accepts seconds (float) or timedelta.
    """
    if isinstance(duration, timedelta):
        ms = int(duration.total_seconds() * 1000)
    else:
        ms = int(float(duration) * 1000)
    return _format_elapsed_millis(ms)


def format_elapsed(start_time: float) -> str:
    """Format time elapsed since ``start_time`` (perf_counter origin)"""
    elapsed_s = time.perf_counter() - float(start_time)
    return format_duration(elapsed_s)
