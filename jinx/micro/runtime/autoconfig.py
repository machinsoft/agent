"""Autonomous Configuration - fully automatic settings without user intervention.

Jinx configures herself:
- Detects available resources
- Enables all brain systems (27)
- Activates AI editor capabilities
- Configures memory systems (4)
- Sets optimal RT parameters
- No manual .env editing needed
"""

from __future__ import annotations

import os
from typing import Any, Dict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore


_FALSE = {"", "0", "false", "off", "no"}


def _is_on(val: str | None, default: str = "1") -> bool:
    v = (val if val is not None else default)
    return (v.strip().lower() not in _FALSE)


def _set_default(name: str, value: str) -> None:
    if os.getenv(name) in (None, ""):
        os.environ[name] = value


def _detect_system_capabilities() -> Dict[str, Any]:
    """Detect system capabilities for optimal auto-configuration."""
    if not PSUTIL_AVAILABLE or psutil is None:
        # Fallback to safe defaults if psutil not available
        return {
            'cpu_count': 4,
            'memory_gb': 8,
            'high_performance': False,
            'medium_performance': True,
            'low_performance': False
        }
    
    try:
        cpu_count = psutil.cpu_count(logical=True) or 4
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        return {
            'cpu_count': cpu_count,
            'memory_gb': memory_gb,
            'high_performance': cpu_count >= 8 and memory_gb >= 16,
            'medium_performance': cpu_count >= 4 and memory_gb >= 8,
            'low_performance': cpu_count < 4 or memory_gb < 8
        }
    except Exception:
        # Fallback to safe defaults
        return {
            'cpu_count': 4,
            'memory_gb': 8,
            'high_performance': False,
            'medium_performance': True,
            'low_performance': False
        }


def apply_auto_defaults(settings: Any | None = None) -> None:
    """Apply fully autonomous configuration - NO USER INPUT NEEDED.

    Jinx automatically:
    - Detects hardware capabilities
    - Enables all brain systems (27)
    - Activates AI editor
    - Configures memory systems (4)
    - Sets optimal concurrency
    - Tunes RT parameters
    """
    # Global auto-mode: ON by default, user can disable
    auto_on = _is_on(os.getenv("JINX_AUTO_MODE"), default="1")
    if not auto_on:
        return
    
    # Detect system capabilities
    caps = _detect_system_capabilities()
    
    # =================================================================
    # CORE SETTINGS - Always enabled for autonomous operation
    # =================================================================
    
    # Model auto-select (use best available)
    if not os.getenv("OPENAI_MODEL"):
        _set_default("OPENAI_MODEL", "gpt-4o")  # Best model by default
    
    # Pulse and timeout (autonomous operation)
    _set_default("PULSE", "500")  # Increased for longer sessions
    _set_default("TIMEOUT", "1000")
    _set_default("JINX_PULSE_HARD_SHUTDOWN", "0")  # Disable hard shutdown
    
    # Crash diagnostics (always enabled)
    _set_default("JINX_CRASH_DIAGNOSTICS", "1")
    
    # Self-healing system (AI-powered auto-repair)
    _set_default("JINX_SELF_HEALING", "1")
    
    # ML System (complete ML infrastructure)
    _set_default("JINX_ML_SYSTEM", "1")
    
    # Dynamic configuration plugin (AI-powered auto-tuning)
    _set_default("JINX_DYNAMIC_CONFIG", "1")

    # Automated actions (self-executing code modifications)
    _set_default("JINX_AUTO_ACTION", "1")

    # File preview for resolved resources (helps LLM without extra user input)
    _set_default("JINX_FILE_PREVIEW", "1")

    # Resource locator concurrency (modest default, RT-safe)
    _set_default("JINX_LOCATOR_CONC", "3")
    
    # Context retrieval timeouts (increased for reliability)
    _set_default("JINX_STAGE_BASECTX_MS", "500")  # Base context: 500ms
    _set_default("JINX_STAGE_PROJCTX_MS", "5000")  # Project context: 5s (embeddings exhaustive)
    _set_default("JINX_STAGE_MEMCTX_MS", "500")  # Memory context: 500ms
    
    # =================================================================
    # EMBEDDINGS SYSTEM - МАКСИМАЛЬНАЯ МОЩНОСТЬ
    # =================================================================
    
    # Project embeddings (emb/files) - полная активация
    _set_default("EMBED_PROJECT_ENABLE", "1")
    _set_default("EMBED_PROJECT_TOP_K", "50")  # Увеличено с 20 до 50 результатов
    _set_default("EMBED_PROJECT_EXHAUSTIVE", "1")  # EXHAUSTIVE MODE - все stages!
    _set_default("EMBED_PROJECT_NO_STAGE_BUDGETS", "0")  # Keep budgets for RT
    _set_default("EMBED_PROJECT_TOTAL_CODE_BUDGET", "50000")  # Увеличено с 20KB до 50KB
    
    # Unified context budget (used in LLM service)
    _set_default("EMBED_UNIFIED_MAX_TIME_MS", "3000")  # 3 секунды на unified context
    
    # Score threshold (lower = more results)
    _set_default("EMBED_PROJECT_SCORE_THRESHOLD", "0.15")  # Снижено с 0.22 до 0.15
    
    # Callgraph enrichment (relationships)
    _set_default("EMBED_PROJECT_CALLGRAPH", "1")
    _set_default("EMBED_PROJECT_CALLGRAPH_TOP_HITS", "5")  # Увеличено с 3 до 5
    _set_default("EMBED_PROJECT_CALLGRAPH_CALLERS_LIMIT", "5")  # Увеличено
    _set_default("EMBED_PROJECT_CALLGRAPH_CALLEES_LIMIT", "5")  # Увеличено
    
    # Full Python scope extraction
    _set_default("EMBED_PROJECT_ALWAYS_FULL_PY_SCOPE", "1")
    _set_default("EMBED_PROJECT_FULL_SCOPE_TOP_N", "10")  # Top 10 get full scope
    
    # Snippet sizing
    _set_default("EMBED_PROJECT_SNIPPET_PER_HIT_CHARS", "2000")  # Увеличено с 1600
    
    # =================================================================
    # BRAIN SYSTEMS (27 SYSTEMS) - All enabled automatically
    # =================================================================
    
    # Core brain systems
    _set_default("JINX_BRAIN_ENABLE", "1")
    _set_default("EMBED_BRAIN_ENABLE", "1")
    _set_default("JINX_BRAIN_AUTO_INIT", "1")
    
    # Adaptive retrieval
    _set_default("JINX_BRAIN_ADAPTIVE_RETRIEVAL", "1")
    _set_default("JINX_BRAIN_UCB1", "1")
    
    # Threshold learning (Bayesian)
    _set_default("JINX_BRAIN_THRESHOLD_LEARNING", "1")
    _set_default("JINX_BRAIN_THOMPSON_SAMPLING", "1")
    
    # Query classification & intent
    _set_default("JINX_BRAIN_QUERY_CLASSIFICATION", "1")
    _set_default("JINX_BRAIN_INTENT_LEARNING", "1")
    
    # Context optimization (Q-learning)
    _set_default("JINX_BRAIN_CONTEXT_OPTIMIZATION", "1")
    _set_default("JINX_BRAIN_Q_LEARNING", "1")
    
    # Semantic routing
    _set_default("JINX_BRAIN_SEMANTIC_ROUTING", "1")
    
    # Intelligent planning (RL)
    _set_default("JINX_BRAIN_INTELLIGENT_PLANNING", "1")
    
    # Cache optimization
    _set_default("JINX_BRAIN_CACHE_OPTIMIZATION", "1")
    
    # Outcome tracking
    _set_default("JINX_BRAIN_OUTCOME_TRACKING", "1")
    
    # Pattern recognition
    _set_default("JINX_BRAIN_PATTERN_RECOGNITION", "1")
    
    # Meta-cognitive system
    _set_default("JINX_BRAIN_META_COGNITIVE", "1")
    
    # Goal-driven system
    _set_default("JINX_BRAIN_GOAL_DRIVEN", "1")
    
    # Ensemble decision maker
    _set_default("JINX_BRAIN_ENSEMBLE", "1")
    
    # Knowledge graph
    _set_default("JINX_BRAIN_KNOWLEDGE_GRAPH", "1")
    
    # Query expander
    _set_default("JINX_BRAIN_QUERY_EXPANDER", "1")
    
    # Auto-tuner
    _set_default("JINX_BRAIN_AUTO_TUNER", "1")
    
    # Performance monitor
    _set_default("JINX_BRAIN_PERFORMANCE_MONITOR", "1")
    
    # =================================================================
    # MEMORY SYSTEMS (4 SYSTEMS) - All enabled automatically
    # =================================================================
    
    # Working memory
    _set_default("JINX_MEMORY_WORKING", "1")
    _set_default("JINX_MEMORY_WORKING_SIZE", "7")  # Miller's 7±2
    
    # Episodic memory
    _set_default("JINX_MEMORY_EPISODIC", "1")
    _set_default("JINX_MEMORY_EPISODIC_SEARCH", "1")
    
    # Semantic memory
    _set_default("JINX_MEMORY_SEMANTIC", "1")
    
    # Jinx memory (permanent)
    _set_default("JINX_PERSIST_MEMORY", "1")
    _set_default("JINX_MEMORY_DIR", ".jinx/memory")
    
    # Memory integration hub
    _set_default("JINX_MEMORY_INTEGRATION", "1")
    _set_default("JINX_MEMORY_CONSOLIDATION", "1")
    
    # =================================================================
    # AI CODE EDITOR - All capabilities enabled
    # =================================================================
    
    # Core editor
    _set_default("JINX_AI_EDITOR_ENABLE", "1")
    
    # Real-time analysis
    _set_default("JINX_AI_ANALYZER_ENABLE", "1")
    _set_default("JINX_AI_DIAGNOSTICS", "1")
    
    # Intelligent completions
    _set_default("JINX_AI_COMPLETIONS", "1")
    _set_default("JINX_AI_SUGGESTIONS", "1")
    
    # Semantic patching
    _set_default("JINX_SEMANTIC_PATCH_ENABLE", "1")
    _set_default("JINX_SEMANTIC_PATCH_ML", "1")
    
    # AST transforms (LibCST)
    _set_default("JINX_AST_TRANSFORMS", "1")
    _set_default("JINX_LIBCST_ENABLE", "1")
    
    # Code smell detection
    _set_default("JINX_CODE_SMELL_DETECTION", "1")
    
    # Pattern learning
    _set_default("JINX_PATTERN_LEARNING", "1")
    
    # =================================================================
    # CONCURRENT PROCESSING - Handled by AutoBrain adaptively
    # =================================================================
    
    # Initialize AutoBrain with system-appropriate baselines
    try:
        from jinx.micro.runtime.autobrain_config import _get_param_state, _save_param_state
        
        if caps['high_performance']:
            # High-end system: start with higher baselines
            for name, val in [("frame_max_conc", 4), ("group_max_conc", 3), ("sandbox_conc", 3)]:
                p = _get_param_state(name, "global")
                if p.samples < 5:  # Only set if not enough learning data
                    p.current = val
                    _save_param_state(p, "global")
        elif caps['medium_performance']:
            # Medium system: balanced baselines
            for name, val in [("frame_max_conc", 3), ("group_max_conc", 2), ("sandbox_conc", 2)]:
                p = _get_param_state(name, "global")
                if p.samples < 5:
                    p.current = val
                    _save_param_state(p, "global")
        else:
            # Low-end system: conservative baselines (AutoBrain will adapt up if needed)
            for name, val in [("frame_max_conc", 2), ("group_max_conc", 2), ("sandbox_conc", 1)]:
                p = _get_param_state(name, "global")
                if p.samples < 5:
                    p.current = val
                    _save_param_state(p, "global")
    except Exception:
        pass  # AutoBrain not available yet, will use defaults
    
    # Context continuity
    _set_default("JINX_CONTEXT_CONTINUITY", "1")
    _set_default("JINX_CONTEXT_HISTORY", "20")
    
    # =================================================================
    # EMBEDDINGS & RETRIEVAL - Automatically enabled
    # =================================================================
    
    _set_default("EMBED_PROJECT_ENABLE", "1")
    _set_default("EMBED_PROJECT_ROOT", ".")
    _set_default("JINX_EMBED_MEMORY_CTX", "1")
    
    # Smart cache
    _set_default("JINX_SMART_CACHE_ENABLE", "1")
    _set_default("JINX_SMART_CACHE_MB", "100")
    _set_default("JINX_SMART_CACHE_ENTRIES", "1000")
    
    # =================================================================
    # CHAINED REASONING - Fully autonomous, AutoBrain adaptive
    # =================================================================
    
    _set_default("JINX_CHAINED_REASONING", "1")
    _set_default("JINX_CHAINED_REFLECT", "1")
    _set_default("JINX_CHAINED_ADVISORY", "1")
    
    # Initialize AutoBrain timeouts based on system capabilities
    try:
        from jinx.micro.runtime.autobrain_config import _get_param_state, _save_param_state
        
        if caps['high_performance']:
            for name, val in [("stage_basectx_ms", 600), ("stage_projctx_ms", 6000)]:
                p = _get_param_state(name, "global")
                if p.samples < 5:
                    p.current = val
                    _save_param_state(p, "global")
        else:
            for name, val in [("stage_basectx_ms", 400), ("stage_projctx_ms", 4000)]:
                p = _get_param_state(name, "global")
                if p.samples < 5:
                    p.current = val
                    _save_param_state(p, "global")
    except Exception:
        pass
    
    # =================================================================
    # UI & UX - Optimal defaults
    # =================================================================
    
    _set_default("JINX_SPINNER_ENABLE", "1")
    _set_default("JINX_SPINNER_MODE", "toolbar")
    _set_default("JINX_SPINNER_MIN_UPDATE_MS", "160")
    _set_default("JINX_SPINNER_REDRAW_ONLY_ON_CHANGE", "1")
    
    _set_default("JINX_INCLUDE_SYSTEM_DESC", "1")
    _set_default("JINX_LOCALE", "en")
    
    # =================================================================
    # RUNTIME & PERFORMANCE - Real-time optimized
    # =================================================================
    
    # Priority queue
    _set_default("JINX_RUNTIME_USE_PRIORITY_QUEUE", "1")
    
    # Streaming
    _set_default("JINX_LLM_STREAM_FASTPATH", "1")
    
    # Cooperative yielding
    _set_default("JINX_COOP_YIELD", "1")
    
    # Context compaction
    _set_default("JINX_CTX_COMPACT", "1")
    _set_default("JINX_CTX_COMPACT_ORCH", "1")
    
    # RT budgets
    _set_default("EMBED_SLICE_MS", "12")
    _set_default("JINX_LOCATOR_VEC_MS", "120")

    # ================================
    # ADVANCED LLM CONSENSUS / CONTEXT
    # ================================
    _set_default("JINX_LLM_CONSENSUS", "1")
    _set_default("JINX_LLM_CONSENSUS_MS", "500")
    _set_default("JINX_LLM_CONSENSUS_K", "3")
    _set_default("JINX_LLM_CONSENSUS_JUDGE", "1")
    _set_default("JINX_LLM_CONSENSUS_JUDGE_MS", "450")
    _set_default("JINX_CODEGRAPH_CTX", "1")

    # ================================
    # AUTOPATCH (RT-friendly budgets)
    # ================================
    _set_default("JINX_AUTOPATCH_MAX_MS", "900")
    _set_default("JINX_AUTOPATCH_PREVIEW_CONC", "4")
    _set_default("JINX_AUTOPATCH_SEARCH_TOPK", "4")
    # Respect budgets by default under RT constraints
    _set_default("JINX_AUTOPATCH_NO_BUDGETS", "0")
    _set_default("JINX_PATCH_CONTEXT_TOL", "0.72")
    _set_default("JINX_AUTOPATCH_BANDIT_HALF_SEC", "1800")

    # ================================
    # BRAIN EXPANSION FOR SEARCH
    # ================================
    _set_default("EMBED_BRAIN_ENABLE", "1")
    _set_default("EMBED_BRAIN_TOP_K", "10")
    _set_default("EMBED_BRAIN_EXPAND_MAX_TOKENS", "6")

    # ================================
    # CENTRAL CPU POOL & ADMISSION CAPS
    # ================================
    # 0 => auto based on cpu_count()
    _set_default("JINX_CPU_WORKERS", "0")
    # Admission control caps (can be overridden by env)
    _set_default("JINX_ADM_GRAPH_CONC", "1")
    _set_default("JINX_ADM_PATCH_CONC", "2")
    _set_default("JINX_ADM_LLM_CONC", "2")
    _set_default("JINX_ADM_TURN_CONC", "4")

    # ================================
    # SELF-REPROGRAM PLANNER (embeddings integration)
    # ================================
    _set_default("JINX_PLAN_TOPK", "8")
    _set_default("JINX_PLAN_EMBED_MS", "600")
    _set_default("JINX_PLAN_REFINE_MS", "500")

    # ================================
    # CODEMODS (enable rope if available)
    # ================================
    try:
        import importlib as _il
        have_rope = True
        try:
            _il.import_module("rope.base.project")
        except Exception:
            have_rope = False
        _set_default("JINX_CODEMOD_ROPE", "1" if have_rope else "0")
    except Exception:
        _set_default("JINX_CODEMOD_ROPE", "0")

    # ================================
    # SELF-REPROGRAM TESTS
    # ================================
    _set_default("JINX_REPROGRAM_TESTS", "1")

    # ================================
    # OBSERVABILITY (enable if installed)
    # ================================
    try:
        import importlib as _il
        _il.import_module("opentelemetry")
        _set_default("JINX_OTEL_SETUP", "1")
        _set_default("JINX_OTEL_EXPORTER", "console")
    except Exception:
        _set_default("JINX_OTEL_SETUP", "0")
    
    # =================================================================
    # INTELLIGENT PATCHER - All features enabled
    # =================================================================
    
    _set_default("JINX_PATCH_AUTOCOMMIT", "1")
    _set_default("JINX_PATCH_CHECK_SYNTAX", "1")
    _set_default("JINX_PATCH_AUTO_INDENT", "1")
    
    # =================================================================
    # VALIDATORS - Enabled for safety
    # =================================================================
    
    _set_default("JINX_VALIDATORS_ENABLE", "1")
    
    # =================================================================
    # AUTO MACROS - All enabled
    # =================================================================
    
    _set_default("JINX_AUTOMACROS", "1")
    _set_default("JINX_AUTOMACRO_DIALOGUE", "1")
    _set_default("JINX_AUTOMACRO_PROJECT", "1")
    _set_default("JINX_AUTOMACRO_CODE", "1")
    
    # =================================================================
    # CONTINUITY & STATE - Enabled
    # =================================================================
    
    _set_default("JINX_CONTINUITY_ENABLE", "1")
    _set_default("JINX_STATEFRAME_ENABLE", "1")
    
    # =================================================================
    # PERSISTENCE - Brain & memory
    # =================================================================
    
    _set_default("JINX_CHAINED_PERSIST_BRAIN", "1")
    _set_default("JINX_BRAIN_DIR", ".jinx/brain")
    
    # =================================================================
    # MULTI-SPLIT & PREFETCH - Enabled
    # =================================================================
    
    _set_default("JINX_MULTI_SPLIT_ENABLE", "1")
    _set_default("JINX_MULTI_SPLIT_MAX", "6")
    _set_default("JINX_PREFETCH_BROKER_CONC", "3")
    
    # =================================================================
    # VERIFICATION - Auto-run
    # =================================================================
    
    _set_default("JINX_VERIFY_AUTORUN", "1")
    _set_default("JINX_AUTOMACRO_VERIFY_EXPORTS", "1")

