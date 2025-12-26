from __future__ import annotations

# Environment-driven tunables for project retrieval
PROJ_DEFAULT_TOP_K = 20
PROJ_SCORE_THRESHOLD = 0.22
PROJ_MIN_PREVIEW_LEN = 12
PROJ_MAX_FILES = 2000
PROJ_MAX_CHUNKS_PER_FILE = 300
PROJ_QUERY_MODEL = "text-embedding-3-small"

# Snippet shaping
PROJ_SNIPPET_AROUND = 12
PROJ_SNIPPET_PER_HIT_CHARS = 1600
PROJ_MULTI_SEGMENT_ENABLE = True
PROJ_SEGMENT_HEAD_LINES = 40
PROJ_SEGMENT_TAIL_LINES = 24
PROJ_SEGMENT_MID_WINDOWS = 3
PROJ_SEGMENT_MID_AROUND = 18
PROJ_SEGMENT_STRIP_COMMENTS = True
PROJ_CONSOLIDATE_PER_FILE = True

# Always include full Python function/class scope when possible
def _env_bool(name: str, default: bool) -> bool:
    return bool(default)

PROJ_ALWAYS_FULL_PY_SCOPE = _env_bool("EMBED_PROJECT_ALWAYS_FULL_PY_SCOPE", True)
PROJ_SCOPE_MAX_CHARS = 0

# Overall budget for <embeddings_code> text (sum of all snippets)
PROJ_TOTAL_CODE_BUDGET = 20000

# Limit the number of hits that expand to full Python scope; others will use windowed snippets (<=0 = unlimited)
PROJ_FULL_SCOPE_TOP_N = 0

# Optional: expand a couple of direct callees for Python full-scope snippets
PROJ_EXPAND_CALLEES_TOP_N = 2
PROJ_EXPAND_CALLEE_MAX_CHARS = 1200
PROJ_USAGE_REFS_LIMIT = 6

# Callgraph enrichment (enabled by default)
PROJ_CALLGRAPH_ENABLED = _env_bool("EMBED_PROJECT_CALLGRAPH", True)
PROJ_CALLGRAPH_TOP_HITS = 3
PROJ_CALLGRAPH_CALLERS_LIMIT = 3
PROJ_CALLGRAPH_CALLEES_LIMIT = 3
PROJ_CALLGRAPH_TIME_MS = 240

# Exhaustive and budget overrides (use with care)
# RT-friendly defaults: exhaustive OFF, budgets ON
PROJ_EXHAUSTIVE_MODE = _env_bool("EMBED_PROJECT_EXHAUSTIVE", False)
PROJ_NO_STAGE_BUDGETS = _env_bool("EMBED_PROJECT_NO_STAGE_BUDGETS", False)
PROJ_NO_CODE_BUDGET = _env_bool("EMBED_PROJECT_NO_CODE_BUDGET", False)

# Per-stage time budgets (ms). Applied as an upper bound per stage; subject to overall max_time_ms.
PROJ_STAGE_PYAST_MS = 200
PROJ_STAGE_JEDI_MS = 220
PROJ_STAGE_PYDOC_MS = 200
PROJ_STAGE_REGEX_MS = 220
PROJ_STAGE_PYFLOW_MS = 200
PROJ_STAGE_LIBCST_MS = 220
PROJ_STAGE_PYDEF_MS = 180
PROJ_STAGE_TB_MS = 120
PROJ_STAGE_PYLITERALS_MS = 200
PROJ_STAGE_FASTSUBSTR_MS = 200
PROJ_STAGE_LINETOKENS_MS = 140
PROJ_STAGE_LINEEXACT_MS = 300
PROJ_STAGE_ASTMATCH_MS = 220
PROJ_STAGE_ASTCONTAINS_MS = 200
PROJ_STAGE_RAPIDFUZZ_MS = 240
PROJ_STAGE_TOKENMATCH_MS = 200
PROJ_STAGE_PRE_MS = 220
PROJ_STAGE_EXACT_MS = 240
PROJ_STAGE_LITERAL_MS = 380
PROJ_STAGE_COOCCUR_MS = 220
PROJ_STAGE_VECTOR_MS = 250
PROJ_STAGE_KEYWORD_MS = 180

# Optional: extra literal scan burst when no hits were found at all
PROJ_LITERAL_BURST_MS = 800

# Token co-occurrence distance (lines)
PROJ_COOCCUR_MAX_DIST = 6
