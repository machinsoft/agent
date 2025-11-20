from __future__ import annotations

# Re-export structural helpers for downstream modules
from .structural import (
    NAME,
    NAME_RE,
    is_camel_case,
    is_pathlike,
    match_paren,
    match_bracket,
    split_top_args,
)

# UTF-8 byte-boundary helpers
from .utf8_bytes import (
    take_bytes_at_char_boundary,
    take_last_bytes_at_char_boundary,
)

# ANSI escape handling
from .ansi_escape import (
    expand_tabs,
    ansi_escape,
    ansi_escape_line,
)

# Tokenizer utilities (tiktoken-backed)
from .tokenizer import (
    EncodingKind,
    Tokenizer,
    warm_model_cache,
)

# Truncation utilities
from .truncate import (
    TruncationPolicy,
    approx_token_count,
    formatted_truncate_text,
    truncate_text,
    truncate_with_token_budget,
)

# Fuzzy matching utilities
from .fuzzy import (
    fuzzy_match,
    fuzzy_indices,
)

# Review formatting
from .review_format import (
    format_review_findings_block,
)
