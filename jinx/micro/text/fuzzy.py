from __future__ import annotations

from typing import List, Optional, Tuple

__all__ = [
    "fuzzy_match",
    "fuzzy_indices",
]


def _casefold_chars_with_map(s: str) -> Tuple[list[str], list[int]]:
    lowered: list[str] = []
    mapping: list[int] = []
    for i, ch in enumerate(s):
        for lc in ch.casefold():  # robust Unicode case fold
            lowered.append(lc)
            mapping.append(i)
    return lowered, mapping


def fuzzy_match(haystack: str, needle: str) -> Optional[Tuple[List[int], int]]:
    """Case-insensitive subsequence match.

    Returns (sorted_unique_indices_in_original_string, score) where lower score is better.
    Strong prefix matches receive a large negative bonus (-100).
    """
    if needle == "":
        return ([], 2**31 - 1)  # i32::MAX analogue

    lowered_chars, lowered_to_orig = _casefold_chars_with_map(haystack)
    lowered_needle = list(needle.casefold())

    result_orig_indices: list[int] = []
    last_lower_pos: Optional[int] = None
    cur = 0
    for nc in lowered_needle:
        found_at: Optional[int] = None
        while cur < len(lowered_chars):
            if lowered_chars[cur] == nc:
                found_at = cur
                cur += 1
                break
            cur += 1
        if found_at is None:
            return None
        pos = found_at
        result_orig_indices.append(lowered_to_orig[pos])
        last_lower_pos = pos

    if not result_orig_indices:
        first_lower_pos = 0
    else:
        target_orig = result_orig_indices[0]
        try:
            first_lower_pos = lowered_to_orig.index(target_orig)
        except ValueError:
            first_lower_pos = 0

    last_lower_pos = first_lower_pos if last_lower_pos is None else last_lower_pos
    window = (last_lower_pos - first_lower_pos + 1) - len(lowered_needle)
    score = window if window > 0 else 0
    if first_lower_pos == 0:
        score -= 100

    result_orig_indices = sorted(set(result_orig_indices))
    return result_orig_indices, int(score)


def fuzzy_indices(haystack: str, needle: str) -> Optional[List[int]]:
    m = fuzzy_match(haystack, needle)
    if m is None:
        return None
    idx, _ = m
    return idx
