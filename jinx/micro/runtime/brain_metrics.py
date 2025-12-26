"""Brain Metrics - Performance tracking and insights for Jinx autonomous brain.

Tracks:
- Task completion rates and latencies
- Goal achievement progress
- Learning effectiveness
- System health over time
- Resource usage patterns
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock

_METRICS_DIR = Path(".jinx") / "metrics"
_METRICS_FILE = _METRICS_DIR / "brain_metrics.json"
_HISTORY_FILE = _METRICS_DIR / "metrics_history.json"

_LOCK = Lock()

# Rolling windows for real-time metrics
_TASK_LATENCIES: deque = deque(maxlen=100)
_SUCCESS_WINDOW: deque = deque(maxlen=50)
_ERROR_WINDOW: deque = deque(maxlen=50)


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics snapshot."""
    timestamp: float = field(default_factory=time.time)
    
    # Task metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_task_latency_ms: float = 0.0
    
    # Goal metrics
    goals_active: int = 0
    goals_achieved: int = 0
    goal_progress: float = 0.0
    
    # Learning metrics
    learnings_total: int = 0
    high_confidence_learnings: int = 0
    
    # Health metrics
    health_status: str = "ok"
    autobrain_success_rate: float = 1.0
    
    # Resource metrics
    llm_calls: int = 0
    self_modifications: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricsSnapshot":
        return cls(**d)


@dataclass
class PerformanceInsight:
    """An insight derived from metrics analysis."""
    category: str  # "performance", "health", "learning", "optimization"
    severity: str  # "info", "warning", "critical"
    message: str
    recommendation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# Current metrics
_current_metrics = MetricsSnapshot()
_metrics_history: List[MetricsSnapshot] = []
_insights: List[PerformanceInsight] = []


def _ensure_dirs() -> None:
    try:
        _METRICS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _load_metrics() -> None:
    global _current_metrics, _metrics_history
    _ensure_dirs()
    
    try:
        if _METRICS_FILE.exists():
            with open(_METRICS_FILE, "r", encoding="utf-8") as f:
                _current_metrics = MetricsSnapshot.from_dict(json.load(f))
    except Exception:
        pass
    
    try:
        if _HISTORY_FILE.exists():
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _metrics_history = [MetricsSnapshot.from_dict(d) for d in data[-100:]]
    except Exception:
        pass


def _save_metrics() -> None:
    _ensure_dirs()
    
    try:
        with open(_METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(_current_metrics.to_dict(), f, indent=2)
    except Exception:
        pass
    
    try:
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in _metrics_history[-100:]], f, indent=2)
    except Exception:
        pass


def record_task_completion(success: bool, latency_ms: float) -> None:
    """Record a task completion."""
    global _current_metrics
    
    with _LOCK:
        if success:
            _current_metrics.tasks_completed += 1
            _SUCCESS_WINDOW.append(1)
        else:
            _current_metrics.tasks_failed += 1
            _SUCCESS_WINDOW.append(0)
        
        _TASK_LATENCIES.append(latency_ms)
        
        if _TASK_LATENCIES:
            _current_metrics.avg_task_latency_ms = sum(_TASK_LATENCIES) / len(_TASK_LATENCIES)
        
        _save_metrics()


def record_error(error_type: str) -> None:
    """Record an error occurrence."""
    with _LOCK:
        _ERROR_WINDOW.append((time.time(), error_type))


def update_from_brain() -> MetricsSnapshot:
    """Update metrics from brain systems."""
    global _current_metrics
    
    _load_metrics()
    
    # Get AutoBrain status
    try:
        from jinx.micro.runtime.autobrain_config import get_diagnostics
        diag = get_diagnostics()
        _current_metrics.autobrain_success_rate = diag.get("avg_success_rate", 1.0)
        _current_metrics.health_status = diag.get("health", "ok")
    except Exception:
        pass
    
    # Get Arch Memory status
    try:
        from jinx.micro.runtime.arch_memory import get_memory_summary
        mem = get_memory_summary()
        tasks = mem.get("tasks", {})
        _current_metrics.tasks_completed = tasks.get("completed", 0)
        _current_metrics.tasks_failed = tasks.get("failed", 0)
    except Exception:
        pass
    
    # Get Evolution status
    try:
        from jinx.micro.runtime.self_evolution import get_evolution_summary
        evo = get_evolution_summary()
        goals = evo.get("goals", {})
        _current_metrics.goals_active = goals.get("active", 0)
        _current_metrics.goals_achieved = goals.get("achieved", 0)
        
        learnings = evo.get("learnings", {})
        _current_metrics.learnings_total = learnings.get("total", 0)
        _current_metrics.high_confidence_learnings = learnings.get("high_confidence", 0)
        
        metrics = evo.get("metrics", {})
        _current_metrics.llm_calls = metrics.get("llm_calls", 0)
        _current_metrics.self_modifications = metrics.get("self_modifications", 0)
    except Exception:
        pass
    
    _current_metrics.timestamp = time.time()
    
    with _LOCK:
        _save_metrics()
    
    return _current_metrics


def take_snapshot() -> MetricsSnapshot:
    """Take a metrics snapshot and add to history."""
    snapshot = update_from_brain()
    
    with _LOCK:
        _metrics_history.append(snapshot)
        if len(_metrics_history) > 100:
            _metrics_history[:] = _metrics_history[-100:]
        _save_metrics()
    
    return snapshot


def get_current_metrics() -> MetricsSnapshot:
    """Get current metrics."""
    return update_from_brain()


def get_metrics_history(limit: int = 20) -> List[MetricsSnapshot]:
    """Get recent metrics history."""
    _load_metrics()
    return _metrics_history[-limit:]


def analyze_performance() -> List[PerformanceInsight]:
    """Analyze metrics and generate insights."""
    insights = []
    metrics = update_from_brain()
    
    # Check success rate
    if _SUCCESS_WINDOW:
        success_rate = sum(_SUCCESS_WINDOW) / len(_SUCCESS_WINDOW)
        if success_rate < 0.7:
            insights.append(PerformanceInsight(
                category="performance",
                severity="warning",
                message=f"Task success rate is low: {success_rate:.0%}",
                recommendation="Review recent failures and adjust strategies",
            ))
    
    # Check latency
    if metrics.avg_task_latency_ms > 30000:
        insights.append(PerformanceInsight(
            category="performance",
            severity="warning",
            message=f"High average task latency: {metrics.avg_task_latency_ms:.0f}ms",
            recommendation="Consider optimizing task execution or reducing complexity",
        ))
    
    # Check health
    if metrics.health_status != "ok":
        insights.append(PerformanceInsight(
            category="health",
            severity="critical" if metrics.health_status == "critical" else "warning",
            message=f"Brain health status: {metrics.health_status}",
            recommendation="Run self-repair check",
        ))
    
    # Check learning effectiveness
    if metrics.learnings_total > 10 and metrics.high_confidence_learnings == 0:
        insights.append(PerformanceInsight(
            category="learning",
            severity="info",
            message="No high-confidence learnings yet",
            recommendation="More successful outcomes needed to reinforce learnings",
        ))
    
    # Check goal progress
    if metrics.goals_active > 5 and metrics.goals_achieved == 0:
        insights.append(PerformanceInsight(
            category="performance",
            severity="info",
            message=f"{metrics.goals_active} active goals, none achieved",
            recommendation="Focus on completing existing goals before adding new ones",
        ))
    
    # Check error rate
    recent_errors = [e for t, e in _ERROR_WINDOW if time.time() - t < 300]
    if len(recent_errors) > 5:
        insights.append(PerformanceInsight(
            category="health",
            severity="warning",
            message=f"{len(recent_errors)} errors in last 5 minutes",
            recommendation="Investigate error patterns",
        ))
    
    return insights


def get_dashboard() -> Dict[str, Any]:
    """Get comprehensive dashboard data."""
    metrics = update_from_brain()
    insights = analyze_performance()
    
    # Calculate trends from history
    history = get_metrics_history(10)
    
    success_trend = "stable"
    if len(history) >= 2:
        recent = history[-1].autobrain_success_rate
        older = history[0].autobrain_success_rate
        if recent > older + 0.1:
            success_trend = "improving"
        elif recent < older - 0.1:
            success_trend = "declining"
    
    return {
        "current": metrics.to_dict(),
        "insights": [
            {
                "category": i.category,
                "severity": i.severity,
                "message": i.message,
                "recommendation": i.recommendation,
            }
            for i in insights
        ],
        "trends": {
            "success_rate": success_trend,
            "tasks_per_hour": len([m for m in history if time.time() - m.timestamp < 3600]),
        },
        "summary": {
            "health": metrics.health_status,
            "success_rate": f"{metrics.autobrain_success_rate:.0%}",
            "tasks": f"{metrics.tasks_completed} completed, {metrics.tasks_failed} failed",
            "goals": f"{metrics.goals_active} active, {metrics.goals_achieved} achieved",
            "learnings": metrics.learnings_total,
        },
    }


def format_dashboard() -> str:
    """Format dashboard as human-readable string."""
    data = get_dashboard()
    metrics = data["current"]
    summary = data["summary"]
    insights = data["insights"]
    
    lines = [
        "╔═══════════════════════════════════════════════════════════════╗",
        "║                 JINX BRAIN DASHBOARD                          ║",
        "╠═══════════════════════════════════════════════════════════════╣",
        f"║  Health: {summary['health']:<12}  Success Rate: {summary['success_rate']:<8}       ║",
        f"║  Tasks: {summary['tasks']:<30}               ║",
        f"║  Goals: {summary['goals']:<30}               ║",
        f"║  Learnings: {summary['learnings']:<8}  LLM Calls: {metrics['llm_calls']:<8}         ║",
        "╠═══════════════════════════════════════════════════════════════╣",
    ]
    
    if insights:
        lines.append("║  INSIGHTS:                                                    ║")
        for insight in insights[:3]:
            severity_icon = "⚠" if insight["severity"] == "warning" else "ℹ" if insight["severity"] == "info" else "❌"
            lines.append(f"║  {severity_icon} {insight['message'][:55]:<55} ║")
    else:
        lines.append("║  No issues detected - system operating normally              ║")
    
    lines.append("╚═══════════════════════════════════════════════════════════════╝")
    
    return "\n".join(lines)


__all__ = [
    "record_task_completion",
    "record_error",
    "update_from_brain",
    "take_snapshot",
    "get_current_metrics",
    "get_metrics_history",
    "analyze_performance",
    "get_dashboard",
    "format_dashboard",
    "MetricsSnapshot",
    "PerformanceInsight",
]
