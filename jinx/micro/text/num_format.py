from __future__ import annotations

from math import isfinite


def format_with_separators(n: int) -> str:
    """Format integer with digit group separators (en-US style)."""
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _format_scaled(n: int, scale: int, frac_digits: int) -> str:
    # Round to requested fractional digits
    value = n / float(scale)
    if not isfinite(value):
        value = 0.0
    fmt = f"{{:.{frac_digits}f}}"
    return fmt.format(round(value, frac_digits))


def format_si_suffix(n: int) -> str:
    """Format token counts with K/M/G suffixes to ~3 significant figures.

    Examples:
    - 999 -> "999"
    - 1200 -> "1.20K"
    - 123456789 -> "123M"
    """
    n = max(0, int(n))
    if n < 1000:
        return format_with_separators(n)

    UNITS = [(1_000, "K"), (1_000_000, "M"), (1_000_000_000, "G")]
    f = float(n)
    for scale, suffix in UNITS:
        if round(100.0 * f / scale) < 1000:
            return f"{_format_scaled(n, scale, 2)}{suffix}"
        elif round(10.0 * f / scale) < 1000:
            return f"{_format_scaled(n, scale, 1)}{suffix}"
        elif round(f / scale) < 1000:
            return f"{_format_scaled(n, scale, 0)}{suffix}"

    # Above 1000G: keep whole‑G precision.
    from math import floor

    g_val = int(round(f / 1e9))
    return f"{format_with_separators(g_val)}G"
