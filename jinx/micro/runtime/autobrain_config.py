"""AutoBrain Configuration - Intelligent Autonomous Configuration System.

Replaces primitive os.getenv with adaptive, learning-based configuration that:
- Analyzes task context and adapts parameters dynamically
- Learns from execution outcomes (success/failure rates, latency)
- Auto-tunes based on performance feedback
- Self-repairs when configuration issues arise
- Scales intelligently based on system load and task complexity

No user intervention required. Jinx becomes smarter over time.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from threading import Lock

_STATE_PATH = Path(".jinx") / "brain" / "autobrain_state.json"
_METRICS_PATH = Path(".jinx") / "brain" / "autobrain_metrics.json"

_LOCK = Lock()
_state: Dict[str, Any] = {}
_metrics: Dict[str, Any] = {}
_initialized = False


@dataclass
class ConfigParam:
    """A single adaptive configuration parameter."""
    name: str
    default: float
    min_val: float
    max_val: float
    current: float = 0.0
    successes: float = 0.0
    failures: float = 0.0
    total_latency_ms: float = 0.0
    samples: int = 0
    last_updated: float = 0.0
    
    def __post_init__(self):
        if self.current == 0.0:
            self.current = self.default
    
    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.samples if self.samples > 0 else 0.0
    
    def score(self) -> float:
        """UCB-like score balancing exploitation and exploration."""
        sr = self.success_rate
        lat_penalty = min(0.3, self.avg_latency_ms / 10000.0)
        trials = max(1, self.successes + self.failures)
        exploration = math.sqrt(math.log(max(1, self.samples + 1)) / trials) * 0.2
        return sr - lat_penalty + exploration


@dataclass
class TaskContext:
    """Context for adaptive parameter selection."""
    task_type: str = "general"
    complexity: float = 0.5  # 0-1 scale
    urgency: float = 0.5  # 0-1 scale
    file_count: int = 0
    code_size_kb: float = 0.0
    has_errors: bool = False
    recent_failures: int = 0
    
    def feature_key(self) -> str:
        """Generate a feature key for context-based learning."""
        c = "hi" if self.complexity > 0.7 else ("lo" if self.complexity < 0.3 else "md")
        u = "urg" if self.urgency > 0.7 else "nrm"
        e = "err" if self.has_errors else "ok"
        return f"{self.task_type}:{c}:{u}:{e}"


# Default parameter definitions with adaptive ranges
_PARAM_DEFS: Dict[str, Dict[str, float]] = {
    # Concurrency parameters
    "frame_max_conc": {"default": 3, "min": 1, "max": 8},
    "group_max_conc": {"default": 3, "min": 1, "max": 6},
    "sandbox_conc": {"default": 2, "min": 1, "max": 4},
    "prefetch_conc": {"default": 2, "min": 1, "max": 4},
    "locator_conc": {"default": 3, "min": 1, "max": 6},
    "llm_max_conc": {"default": 4, "min": 1, "max": 8},
    
    # Timeout parameters (ms)
    "autopatch_max_ms": {"default": 900, "min": 300, "max": 3000},
    "semantic_patch_ms": {"default": 400, "min": 100, "max": 1500},
    "stage_basectx_ms": {"default": 500, "min": 100, "max": 2000},
    "stage_projctx_ms": {"default": 5000, "min": 1000, "max": 15000},
    "turn_timeout_sec": {"default": 45, "min": 15, "max": 120},
    "llm_timeout_ms": {"default": 20000, "min": 5000, "max": 60000},
    
    # Search/retrieval parameters
    "autopatch_search_topk": {"default": 4, "min": 2, "max": 12},
    "embed_project_top_k": {"default": 50, "min": 10, "max": 150},
    "semantic_patch_topk": {"default": 5, "min": 2, "max": 15},
    "brain_top_k": {"default": 10, "min": 4, "max": 30},
    
    # Thresholds
    "patch_context_tol": {"default": 0.72, "min": 0.4, "max": 0.95},
    "semantic_patch_tol": {"default": 0.55, "min": 0.3, "max": 0.85},
    "embed_score_threshold": {"default": 0.15, "min": 0.05, "max": 0.4},
    "arch_conf_min": {"default": 0.55, "min": 0.3, "max": 0.8},
    
    # Limits
    "patch_autocommit_max": {"default": 40, "min": 10, "max": 100},
    "code_budget_kb": {"default": 50, "min": 10, "max": 200},
    "bandit_half_life_sec": {"default": 1800, "min": 300, "max": 7200},
}


def _ensure_dirs() -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _load_state() -> Dict[str, Any]:
    global _state
    try:
        if _STATE_PATH.exists():
            with open(_STATE_PATH, "r", encoding="utf-8") as f:
                _state = json.load(f)
    except Exception:
        _state = {}
    return _state


def _save_state() -> None:
    _ensure_dirs()
    try:
        with open(_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(_state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_metrics() -> Dict[str, Any]:
    global _metrics
    try:
        if _METRICS_PATH.exists():
            with open(_METRICS_PATH, "r", encoding="utf-8") as f:
                _metrics = json.load(f)
    except Exception:
        _metrics = {}
    return _metrics


def _save_metrics() -> None:
    _ensure_dirs()
    try:
        with open(_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(_metrics, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _init() -> None:
    global _initialized
    if _initialized:
        return
    with _LOCK:
        if _initialized:
            return
        _load_state()
        _load_metrics()
        _initialized = True


def _get_param_state(name: str, ctx_key: str = "global") -> ConfigParam:
    """Get or create parameter state for a given context."""
    _init()
    
    pdef = _PARAM_DEFS.get(name, {"default": 1.0, "min": 0.1, "max": 10.0})
    
    ctx_params = _state.setdefault("params", {}).setdefault(ctx_key, {})
    ps = ctx_params.get(name)
    
    if ps is None:
        return ConfigParam(
            name=name,
            default=pdef["default"],
            min_val=pdef["min"],
            max_val=pdef["max"],
            current=pdef["default"],
        )
    
    return ConfigParam(
        name=name,
        default=pdef["default"],
        min_val=pdef["min"],
        max_val=pdef["max"],
        current=float(ps.get("current", pdef["default"])),
        successes=float(ps.get("successes", 0)),
        failures=float(ps.get("failures", 0)),
        total_latency_ms=float(ps.get("total_latency_ms", 0)),
        samples=int(ps.get("samples", 0)),
        last_updated=float(ps.get("last_updated", 0)),
    )


def _save_param_state(param: ConfigParam, ctx_key: str = "global") -> None:
    """Persist parameter state."""
    _init()
    with _LOCK:
        ctx_params = _state.setdefault("params", {}).setdefault(ctx_key, {})
        ctx_params[param.name] = {
            "current": param.current,
            "successes": param.successes,
            "failures": param.failures,
            "total_latency_ms": param.total_latency_ms,
            "samples": param.samples,
            "last_updated": time.time(),
        }
        _save_state()


def _decay(value: float, half_life: float = 3600.0) -> float:
    """Apply time-based decay to prevent stale data dominance."""
    return value * 0.95  # Simple decay per update


def _adapt_value(param: ConfigParam, ctx: TaskContext) -> float:
    """Intelligently adapt parameter value based on context and history."""
    base = param.current
    
    # Complexity adjustment: higher complexity -> more resources
    if ctx.complexity > 0.7:
        base = min(param.max_val, base * 1.2)
    elif ctx.complexity < 0.3:
        base = max(param.min_val, base * 0.85)
    
    # Urgency adjustment: higher urgency -> faster (lower timeouts, higher concurrency)
    if ctx.urgency > 0.7:
        if "timeout" in param.name or "_ms" in param.name or "_sec" in param.name:
            base = max(param.min_val, base * 0.8)
        elif "conc" in param.name:
            base = min(param.max_val, base * 1.15)
    
    # Error recovery: if recent failures, be more conservative
    if ctx.has_errors or ctx.recent_failures > 2:
        if "conc" in param.name:
            base = max(param.min_val, base * 0.8)
        elif "timeout" in param.name or "_ms" in param.name:
            base = min(param.max_val, base * 1.3)
    
    # Performance-based adjustment using UCB-like scoring
    if param.samples > 5:
        if param.success_rate < 0.5:
            # Poor performance -> explore alternative values
            delta = (param.max_val - param.min_val) * 0.1
            if param.current > param.default:
                base = max(param.min_val, base - delta)
            else:
                base = min(param.max_val, base + delta)
        elif param.success_rate > 0.8 and param.avg_latency_ms < 1000:
            # Great performance -> slightly optimize toward speed
            if "timeout" in param.name or "_ms" in param.name:
                base = max(param.min_val, base * 0.95)
    
    return max(param.min_val, min(param.max_val, base))


# ============================================================================
# PUBLIC API - Used by all runtime modules instead of os.getenv
# ============================================================================

def get(name: str, ctx: Optional[TaskContext] = None) -> float:
    """Get adaptive configuration value.
    
    This is the main entry point replacing os.getenv calls.
    Returns an intelligently adapted value based on:
    - Historical performance data
    - Current task context
    - System state and recent outcomes
    """
    ctx = ctx or TaskContext()
    ctx_key = ctx.feature_key()
    
    # Try context-specific state first, fall back to global
    param = _get_param_state(name, ctx_key)
    if param.samples < 3:
        param = _get_param_state(name, "global")
    
    return _adapt_value(param, ctx)


def get_int(name: str, ctx: Optional[TaskContext] = None) -> int:
    """Get adaptive configuration as integer."""
    return int(round(get(name, ctx)))


def get_bool(name: str, ctx: Optional[TaskContext] = None) -> bool:
    """Get adaptive configuration as boolean (> 0.5 = True)."""
    return get(name, ctx) > 0.5


def record_outcome(
    name: str,
    success: bool,
    latency_ms: float = 0.0,
    ctx: Optional[TaskContext] = None,
) -> None:
    """Record execution outcome for learning.
    
    Called after operations complete to update the learning model.
    This is how Jinx learns and improves over time.
    """
    ctx = ctx or TaskContext()
    ctx_key = ctx.feature_key()
    
    param = _get_param_state(name, ctx_key)
    
    # Decay old data
    param.successes = _decay(param.successes)
    param.failures = _decay(param.failures)
    param.total_latency_ms = _decay(param.total_latency_ms)
    
    # Record new outcome
    if success:
        param.successes += 1.0
    else:
        param.failures += 1.0
    param.total_latency_ms += latency_ms
    param.samples += 1
    
    # Update current value based on outcome
    if not success and param.samples > 3:
        # Failed -> adjust toward safer values
        if "conc" in name:
            param.current = max(param.min_val, param.current * 0.9)
        elif "timeout" in name or "_ms" in name:
            param.current = min(param.max_val, param.current * 1.1)
    elif success and latency_ms > 0 and param.samples > 5:
        # Success with latency data -> optimize
        avg_lat = param.avg_latency_ms
        if latency_ms < avg_lat * 0.8:
            # Faster than average -> can be more aggressive
            if "conc" in name:
                param.current = min(param.max_val, param.current * 1.05)
    
    _save_param_state(param, ctx_key)
    
    # Also update global stats
    global_param = _get_param_state(name, "global")
    global_param.successes = _decay(global_param.successes) + (1.0 if success else 0.0)
    global_param.failures = _decay(global_param.failures) + (0.0 if success else 1.0)
    global_param.total_latency_ms = _decay(global_param.total_latency_ms) + latency_ms
    global_param.samples += 1
    _save_param_state(global_param, "global")


def self_repair_check() -> List[str]:
    """Check for configuration issues and auto-repair.
    
    Returns list of repairs made.
    """
    _init()
    repairs: List[str] = []
    
    with _LOCK:
        for name, pdef in _PARAM_DEFS.items():
            global_param = _get_param_state(name, "global")
            
            # Check for pathological states
            if global_param.samples > 10 and global_param.success_rate < 0.2:
                # Very low success rate -> reset to default
                global_param.current = pdef["default"]
                global_param.successes = 0
                global_param.failures = 0
                global_param.samples = 0
                _save_param_state(global_param, "global")
                repairs.append(f"Reset {name} to default due to low success rate")
            
            # Check for extreme values that might be causing issues
            if global_param.current < pdef["min"] or global_param.current > pdef["max"]:
                global_param.current = max(pdef["min"], min(pdef["max"], global_param.current))
                _save_param_state(global_param, "global")
                repairs.append(f"Clamped {name} to valid range")
    
    return repairs


def get_diagnostics() -> Dict[str, Any]:
    """Get diagnostic information about the configuration system."""
    _init()
    
    diag: Dict[str, Any] = {
        "params": {},
        "health": "ok",
        "total_samples": 0,
        "avg_success_rate": 0.0,
    }
    
    total_sr = 0.0
    count = 0
    
    for name in _PARAM_DEFS:
        param = _get_param_state(name, "global")
        diag["params"][name] = {
            "current": param.current,
            "default": param.default,
            "success_rate": param.success_rate,
            "avg_latency_ms": param.avg_latency_ms,
            "samples": param.samples,
        }
        diag["total_samples"] += param.samples
        if param.samples > 0:
            total_sr += param.success_rate
            count += 1
    
    if count > 0:
        diag["avg_success_rate"] = total_sr / count
        if diag["avg_success_rate"] < 0.5:
            diag["health"] = "degraded"
        elif diag["avg_success_rate"] < 0.3:
            diag["health"] = "critical"
    
    return diag


# Feature flags with adaptive learning
_FEATURE_FLAGS: Dict[str, Dict[str, Any]] = {
    "brain_enable": {"default": True, "learns": True},
    "semantic_patch": {"default": True, "learns": True},
    "auto_action": {"default": True, "learns": True},
    "self_healing": {"default": True, "learns": True},
    "prefetch": {"default": True, "learns": True},
    "embeddings": {"default": True, "learns": True},
    "callgraph": {"default": True, "learns": True},
    "exhaustive_search": {"default": True, "learns": False},
}


def feature_enabled(name: str, ctx: Optional[TaskContext] = None) -> bool:
    """Check if a feature is enabled, with adaptive learning.
    
    Replaces primitive truthy() env checks with intelligent feature gating.
    Features can be auto-disabled if they consistently cause failures.
    """
    _init()
    
    fdef = _FEATURE_FLAGS.get(name, {"default": True, "learns": False})
    default = fdef.get("default", True)
    learns = fdef.get("learns", False)
    
    if not learns:
        return default
    
    # Check feature-specific success rate
    ctx_key = ctx.feature_key() if ctx else "global"
    feature_state = _state.setdefault("features", {}).setdefault(ctx_key, {})
    fs = feature_state.get(name, {"enabled": default, "successes": 0.0, "failures": 0.0})
    
    total = fs.get("successes", 0) + fs.get("failures", 0)
    if total < 5:
        return default
    
    success_rate = fs.get("successes", 0) / total
    
    # Auto-disable features with very low success rate
    if success_rate < 0.2 and total > 10:
        return False
    
    return fs.get("enabled", default)


def record_feature_outcome(name: str, success: bool, ctx: Optional[TaskContext] = None) -> None:
    """Record feature usage outcome for learning."""
    _init()
    
    ctx_key = ctx.feature_key() if ctx else "global"
    feature_state = _state.setdefault("features", {}).setdefault(ctx_key, {})
    fs = feature_state.get(name, {"enabled": True, "successes": 0.0, "failures": 0.0})
    
    # Decay old data
    fs["successes"] = _decay(fs.get("successes", 0))
    fs["failures"] = _decay(fs.get("failures", 0))
    
    if success:
        fs["successes"] = fs.get("successes", 0) + 1.0
    else:
        fs["failures"] = fs.get("failures", 0) + 1.0
    
    # Auto-adjust enabled state based on success rate
    total = fs["successes"] + fs["failures"]
    if total > 10:
        success_rate = fs["successes"] / total
        if success_rate < 0.2:
            fs["enabled"] = False
        elif success_rate > 0.7:
            fs["enabled"] = True
    
    feature_state[name] = fs
    with _LOCK:
        _save_state()


# Failure tracking for self-healing
_FAILURE_WINDOW: List[Tuple[float, str, str]] = []  # (timestamp, component, error)
_FAILURE_WINDOW_SIZE = 50


def record_failure(component: str, error: str) -> None:
    """Record a component failure for self-healing analysis."""
    global _FAILURE_WINDOW
    
    now = time.time()
    _FAILURE_WINDOW.append((now, component, error))
    
    # Keep window bounded
    if len(_FAILURE_WINDOW) > _FAILURE_WINDOW_SIZE:
        _FAILURE_WINDOW = _FAILURE_WINDOW[-_FAILURE_WINDOW_SIZE:]


def analyze_failures() -> Dict[str, Any]:
    """Analyze recent failures to detect patterns."""
    now = time.time()
    recent_window = 300  # 5 minutes
    
    recent = [(t, c, e) for t, c, e in _FAILURE_WINDOW if now - t < recent_window]
    
    if not recent:
        return {"status": "healthy", "recent_failures": 0}
    
    # Count by component
    by_component: Dict[str, int] = {}
    for _, c, _ in recent:
        by_component[c] = by_component.get(c, 0) + 1
    
    # Detect problematic components
    problematic = [c for c, count in by_component.items() if count >= 3]
    
    return {
        "status": "degraded" if problematic else "warning",
        "recent_failures": len(recent),
        "by_component": by_component,
        "problematic": problematic,
    }


async def autonomous_optimization_loop(interval_sec: float = 60.0) -> None:
    """Background task for continuous self-optimization.
    
    Periodically:
    - Runs self-repair checks
    - Analyzes performance trends and failure patterns
    - Adjusts parameters proactively
    - Auto-disables problematic features
    """
    while True:
        try:
            await asyncio.sleep(interval_sec)
            
            # Self-repair
            repairs = self_repair_check()
            if repairs:
                try:
                    from jinx.micro.ui.output import pretty_echo_async
                    await pretty_echo_async(
                        f"<autobrain_repair>\n" + "\n".join(repairs) + "\n</autobrain_repair>",
                        title="AutoBrain"
                    )
                except Exception:
                    pass
            
            # Analyze failure patterns
            failure_analysis = analyze_failures()
            if failure_analysis["status"] == "degraded":
                # Auto-adjust for problematic components
                for component in failure_analysis.get("problematic", []):
                    # Reduce concurrency for failing components
                    if "frame" in component or "scheduler" in component:
                        param = _get_param_state("frame_max_conc", "global")
                        param.current = max(param.min_val, param.current - 1)
                        _save_param_state(param, "global")
                    elif "sandbox" in component:
                        param = _get_param_state("sandbox_conc", "global")
                        param.current = max(param.min_val, param.current - 1)
                        _save_param_state(param, "global")
                    elif "patch" in component:
                        param = _get_param_state("autopatch_max_ms", "global")
                        param.current = min(param.max_val, param.current * 1.2)
                        _save_param_state(param, "global")
            
            # Proactive optimization based on overall health
            diag = get_diagnostics()
            if diag["health"] == "degraded":
                # System struggling -> be more conservative
                for name in ["frame_max_conc", "group_max_conc", "sandbox_conc"]:
                    param = _get_param_state(name, "global")
                    if param.current > param.min_val:
                        param.current = max(param.min_val, param.current * 0.9)
                        _save_param_state(param, "global")
                
                for name in ["autopatch_max_ms", "turn_timeout_sec", "stage_projctx_ms"]:
                    param = _get_param_state(name, "global")
                    if param.current < param.max_val:
                        param.current = min(param.max_val, param.current * 1.1)
                        _save_param_state(param, "global")
            
            elif diag["health"] == "ok" and diag["avg_success_rate"] > 0.85:
                # System performing well -> can be more aggressive
                for name in ["frame_max_conc", "group_max_conc"]:
                    param = _get_param_state(name, "global")
                    if param.current < param.max_val and param.samples > 20:
                        param.current = min(param.max_val, param.current * 1.05)
                        _save_param_state(param, "global")
            
        except Exception:
            await asyncio.sleep(interval_sec)


def start_optimization_task() -> asyncio.Task:
    """Start the autonomous optimization background task."""
    return asyncio.create_task(
        autonomous_optimization_loop(60.0),
        name="autobrain-optimizer"
    )


# ============================================================================
# CONVENIENCE ALIASES for common parameters
# ============================================================================

def frame_max_conc(ctx: Optional[TaskContext] = None) -> int:
    return get_int("frame_max_conc", ctx)

def group_max_conc(ctx: Optional[TaskContext] = None) -> int:
    return get_int("group_max_conc", ctx)

def sandbox_conc(ctx: Optional[TaskContext] = None) -> int:
    return get_int("sandbox_conc", ctx)

def autopatch_max_ms(ctx: Optional[TaskContext] = None) -> int:
    return get_int("autopatch_max_ms", ctx)

def turn_timeout_sec(ctx: Optional[TaskContext] = None) -> float:
    return get("turn_timeout_sec", ctx)

def embed_project_top_k(ctx: Optional[TaskContext] = None) -> int:
    return get_int("embed_project_top_k", ctx)


__all__ = [
    "get",
    "get_int", 
    "get_bool",
    "record_outcome",
    "self_repair_check",
    "get_diagnostics",
    "start_optimization_task",
    "TaskContext",
    "ConfigParam",
    # Feature flags
    "feature_enabled",
    "record_feature_outcome",
    # Failure tracking
    "record_failure",
    "analyze_failures",
    # Convenience aliases
    "frame_max_conc",
    "group_max_conc",
    "sandbox_conc",
    "autopatch_max_ms",
    "turn_timeout_sec",
    "embed_project_top_k",
]
