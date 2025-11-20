from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union

__all__ = [
    "TruncationPolicy",
    "approx_token_count",
    "formatted_truncate_text",
    "truncate_text",
    "truncate_with_token_budget",
]

APPROX_BYTES_PER_TOKEN: int = 4


@dataclass(frozen=True)
class TruncationPolicy:
    kind: str  # "bytes" | "tokens"
    budget: int

    @staticmethod
    def Bytes(n: int) -> "TruncationPolicy":
        return TruncationPolicy("bytes", max(0, int(n)))

    @staticmethod
    def Tokens(n: int) -> "TruncationPolicy":
        return TruncationPolicy("tokens", max(0, int(n)))

    def mul(self, multiplier: float) -> "TruncationPolicy":
        from math import ceil

        return TruncationPolicy(self.kind, int(ceil(self.budget * float(multiplier))))

    def token_budget(self) -> int:
        if self.kind == "tokens":
            return self.budget
        # bytes -> approximate tokens
        return int(approx_tokens_from_byte_count(self.budget))

    def byte_budget(self) -> int:
        if self.kind == "bytes":
            return self.budget
        return approx_bytes_for_tokens(self.budget)


def approx_token_count(text: str) -> int:
    n = len(text.encode("utf-8"))
    return (n + (APPROX_BYTES_PER_TOKEN - 1)) // APPROX_BYTES_PER_TOKEN


def approx_bytes_for_tokens(tokens: int) -> int:
    return max(0, int(tokens)) * APPROX_BYTES_PER_TOKEN


def approx_tokens_from_byte_count(bytes_n: int) -> int:
    b = max(0, int(bytes_n))
    return (b + (APPROX_BYTES_PER_TOKEN - 1)) // APPROX_BYTES_PER_TOKEN


def _split_budget(budget: int) -> Tuple[int, int]:
    left = budget // 2
    return left, budget - left


def _split_string(s: str, beginning_bytes: int, end_bytes: int) -> Tuple[int, str, str]:
    if not s:
        return 0, "", ""

    b = s.encode("utf-8")
    total_len = len(b)
    tail_start_target = max(0, total_len - max(0, int(end_bytes)))

    prefix_end_bytes = 0
    suffix_start_bytes = total_len
    removed_chars = 0
    suffix_started = False

    cum = 0
    for ch in s:
        chb = ch.encode("utf-8")
        char_end = cum + len(chb)
        if char_end <= beginning_bytes:
            prefix_end_bytes = char_end
            cum = char_end
            continue
        if cum >= tail_start_target:
            if not suffix_started:
                suffix_start_bytes = cum
                suffix_started = True
            cum = char_end
            continue
        removed_chars += 1
        cum = char_end

    if suffix_start_bytes < prefix_end_bytes:
        suffix_start_bytes = prefix_end_bytes

    before = b[:prefix_end_bytes].decode("utf-8", errors="strict")
    after = b[suffix_start_bytes:].decode("utf-8", errors="strict")
    return removed_chars, before, after


def _format_truncation_marker(policy: TruncationPolicy, removed_count: int) -> str:
    if policy.kind == "tokens":
        return f"…{removed_count} tokens truncated…"
    return f"…{removed_count} chars truncated…"


def _removed_units_for_source(policy: TruncationPolicy, removed_bytes: int, removed_chars: int) -> int:
    if policy.kind == "tokens":
        return approx_tokens_from_byte_count(removed_bytes)
    return max(0, int(removed_chars))


def _assemble_truncated_output(prefix: str, suffix: str, marker: str) -> str:
    return f"{prefix}{marker}{suffix}"


def truncate_with_byte_estimate(s: str, policy: TruncationPolicy) -> str:
    if not s:
        return ""
    max_bytes = policy.byte_budget()
    if max_bytes == 0:
        marker = _format_truncation_marker(policy, _removed_units_for_source(policy, len(s.encode('utf-8')), len(s)))
        return marker
    if len(s.encode("utf-8")) <= max_bytes:
        return s

    left_budget, right_budget = _split_budget(max_bytes)
    removed_chars, left, right = _split_string(s, left_budget, right_budget)

    total_bytes = len(s.encode("utf-8"))
    removed = total_bytes - max_bytes
    marker = _format_truncation_marker(policy, _removed_units_for_source(policy, removed, removed_chars))
    return _assemble_truncated_output(left, right, marker)


def truncate_with_token_budget(s: str, policy: TruncationPolicy) -> Tuple[str, Union[int, None]]:
    if not s:
        return "", None
    max_tokens = policy.token_budget()
    byte_len = len(s.encode("utf-8"))
    if max_tokens > 0 and byte_len <= approx_bytes_for_tokens(max_tokens):
        return s, None

    truncated = truncate_with_byte_estimate(s, policy)
    approx_total = approx_token_count(s)
    if truncated == s:
        return truncated, None
    return truncated, approx_total


def truncate_text(content: str, policy: TruncationPolicy) -> str:
    if policy.kind == "bytes":
        return truncate_with_byte_estimate(content, policy)
    else:
        truncated, _ = truncate_with_token_budget(content, policy)
        return truncated


def formatted_truncate_text(content: str, policy: TruncationPolicy) -> str:
    if len(content.encode("utf-8")) <= policy.byte_budget():
        return content
    total_lines = content.count("\n") + 1 if content else 0
    result = truncate_text(content, policy)
    return f"Total output lines: {total_lines}\n\n{result}"
