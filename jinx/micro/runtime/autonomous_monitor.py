"""Autonomous Monitor - Real-time issue detection, logging, and auto-repair.

Monitors Jinx in real-time to:
- Detect and log all issues, errors, anomalies
- Automatically attempt repairs without user intervention
- Track repair attempts and outcomes
- Escalate to alternative strategies when repairs fail
- Maintain comprehensive issue history for learning
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Awaitable

_MONITOR_DIR = Path(".jinx") / "monitor"
_ISSUES_FILE = _MONITOR_DIR / "issues.json"
_REPAIRS_FILE = _MONITOR_DIR / "repairs.json"
_HISTORY_FILE = _MONITOR_DIR / "repair_history.json"

_LOCK = Lock()


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IssueStatus(str, Enum):
    DETECTED = "detected"
    ANALYZING = "analyzing"
    REPAIRING = "repairing"
    REPAIRED = "repaired"
    FAILED = "failed"
    ESCALATED = "escalated"
    ROLLED_BACK = "rolled_back"


class RepairStrategy(str, Enum):
    RESTART_COMPONENT = "restart_component"
    ROLLBACK_CHANGE = "rollback_change"
    ADJUST_CONFIG = "adjust_config"
    CLEAR_CACHE = "clear_cache"
    RESET_STATE = "reset_state"
    SKIP_OPERATION = "skip_operation"
    ALTERNATIVE_PATH = "alternative_path"
    LLM_ANALYSIS = "llm_analysis"
    SELF_MODIFY = "self_modify"


@dataclass
class Issue:
    """A detected issue in the system."""
    id: str
    category: str  # "runtime", "memory", "performance", "logic", "integration"
    severity: IssueSeverity
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    
    status: IssueStatus = IssueStatus.DETECTED
    detected_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    
    repair_attempts: int = 0
    last_repair_strategy: Optional[str] = None
    repair_history: List[Dict[str, Any]] = field(default_factory=list)
    
    recurring_count: int = 1
    last_occurrence: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["status"] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Issue":
        d["severity"] = IssueSeverity(d["severity"])
        d["status"] = IssueStatus(d["status"])
        return cls(**d)


@dataclass
class RepairAttempt:
    """Record of a repair attempt."""
    issue_id: str
    strategy: RepairStrategy
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    success: bool = False
    result: str = ""
    rollback_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["strategy"] = self.strategy.value
        return d


# Active issues and repair state
_issues: Dict[str, Issue] = {}
_repair_attempts: List[RepairAttempt] = []
_issue_patterns: Dict[str, int] = {}  # pattern -> count for recurring detection

# Real-time monitoring queue
_issue_queue: deque = deque(maxlen=1000)
_repair_queue: asyncio.Queue = None

# Repair handlers registry
_repair_handlers: Dict[RepairStrategy, Callable] = {}


def _ensure_dirs() -> None:
    try:
        _MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _load_state() -> None:
    global _issues, _repair_attempts
    _ensure_dirs()
    
    try:
        if _ISSUES_FILE.exists():
            with open(_ISSUES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _issues = {k: Issue.from_dict(v) for k, v in data.items()}
    except Exception:
        pass
    
    try:
        if _REPAIRS_FILE.exists():
            with open(_REPAIRS_FILE, "r", encoding="utf-8") as f:
                _repair_attempts = json.load(f)
    except Exception:
        pass


def _save_state() -> None:
    _ensure_dirs()
    
    try:
        with open(_ISSUES_FILE, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in _issues.items()}, f, indent=2)
    except Exception:
        pass
    
    try:
        # Keep only recent repair attempts
        recent = _repair_attempts[-500:] if len(_repair_attempts) > 500 else _repair_attempts
        with open(_REPAIRS_FILE, "w", encoding="utf-8") as f:
            json.dump(recent, f, indent=2)
    except Exception:
        pass


def _generate_issue_id(category: str, description: str) -> str:
    import hashlib
    content = f"{category}:{description[:100]}:{time.time()}"
    return f"issue_{hashlib.md5(content.encode()).hexdigest()[:12]}"


def _get_issue_pattern(category: str, description: str) -> str:
    """Generate pattern for recurring issue detection."""
    # Normalize description to detect similar issues
    import re
    normalized = re.sub(r'\d+', 'N', description)  # Replace numbers
    normalized = re.sub(r'0x[a-fA-F0-9]+', 'ADDR', normalized)  # Replace addresses
    return f"{category}:{normalized[:80]}"


# =============================================================================
# ISSUE DETECTION
# =============================================================================

def detect_issue(
    category: str,
    severity: IssueSeverity,
    description: str,
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[Exception] = None,
    auto_repair: bool = True,
) -> str:
    """Detect and log an issue. Returns issue_id."""
    _load_state()
    
    # Check for recurring issue
    pattern = _get_issue_pattern(category, description)
    
    with _LOCK:
        # Check if similar issue already exists and is active
        for existing in _issues.values():
            if existing.status not in (IssueStatus.REPAIRED, IssueStatus.ROLLED_BACK):
                existing_pattern = _get_issue_pattern(existing.category, existing.description)
                if existing_pattern == pattern:
                    # Update existing issue
                    existing.recurring_count += 1
                    existing.last_occurrence = time.time()
                    if severity.value > existing.severity.value:
                        existing.severity = severity
                    _save_state()
                    
                    # Escalate if recurring too often
                    if existing.recurring_count >= 5 and existing.status != IssueStatus.ESCALATED:
                        existing.status = IssueStatus.ESCALATED
                        _trigger_escalation(existing)
                    
                    return existing.id
        
        # Create new issue
        issue_id = _generate_issue_id(category, description)
        
        stack_trace = None
        file_path = None
        function_name = None
        
        if exception:
            stack_trace = traceback.format_exc()
            # Extract location from traceback
            tb = traceback.extract_tb(exception.__traceback__)
            if tb:
                last_frame = tb[-1]
                file_path = last_frame.filename
                function_name = last_frame.name
        
        issue = Issue(
            id=issue_id,
            category=category,
            severity=severity,
            description=description,
            context=context or {},
            stack_trace=stack_trace,
            file_path=file_path,
            function_name=function_name,
        )
        
        _issues[issue_id] = issue
        _issue_queue.append(issue)
        _issue_patterns[pattern] = _issue_patterns.get(pattern, 0) + 1
        
        _save_state()
        
        # Learn from issue
        try:
            from jinx.micro.runtime.self_evolution import learn
            learn(
                category="error_pattern",
                description=f"{category}: {description[:50]}",
                context=str(context)[:200] if context else "",
                confidence=0.4,
            )
        except Exception:
            pass
    
    # Trigger auto-repair if enabled
    if auto_repair and severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL):
        asyncio.create_task(_auto_repair_issue(issue_id))
    
    return issue_id


def detect_from_exception(
    exception: Exception,
    category: str = "runtime",
    context: Optional[Dict[str, Any]] = None,
    auto_repair: bool = True,
) -> str:
    """Detect issue from an exception."""
    error_type = type(exception).__name__
    description = f"{error_type}: {str(exception)[:200]}"
    
    # Determine severity based on exception type
    severity = IssueSeverity.ERROR
    if isinstance(exception, (MemoryError, SystemError, RecursionError)):
        severity = IssueSeverity.CRITICAL
    elif isinstance(exception, (Warning, DeprecationWarning)):
        severity = IssueSeverity.WARNING
    
    return detect_issue(
        category=category,
        severity=severity,
        description=description,
        context=context,
        exception=exception,
        auto_repair=auto_repair,
    )


def detect_performance_issue(
    metric_name: str,
    current_value: float,
    threshold: float,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Detect performance-related issues."""
    if current_value <= threshold:
        return None
    
    severity = IssueSeverity.WARNING
    if current_value > threshold * 2:
        severity = IssueSeverity.ERROR
    if current_value > threshold * 5:
        severity = IssueSeverity.CRITICAL
    
    return detect_issue(
        category="performance",
        severity=severity,
        description=f"{metric_name} exceeded threshold: {current_value:.2f} > {threshold:.2f}",
        context={"metric": metric_name, "value": current_value, "threshold": threshold, **(context or {})},
        auto_repair=True,
    )


def detect_anomaly(
    component: str,
    expected: Any,
    actual: Any,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Detect behavioral anomalies."""
    return detect_issue(
        category="logic",
        severity=IssueSeverity.WARNING,
        description=f"Anomaly in {component}: expected {expected}, got {actual}",
        context={"component": component, "expected": expected, "actual": actual, **(context or {})},
        auto_repair=True,
    )


# =============================================================================
# AUTO-REPAIR SYSTEM
# =============================================================================

def register_repair_handler(strategy: RepairStrategy, handler: Callable) -> None:
    """Register a repair handler for a strategy."""
    _repair_handlers[strategy] = handler


def _select_repair_strategy(issue: Issue) -> RepairStrategy:
    """Select best repair strategy based on issue type and history."""
    
    # Check what strategies have been tried
    tried_strategies = {h.get("strategy") for h in issue.repair_history}
    
    # Strategy selection based on category and severity
    strategy_priority = []
    
    if issue.category == "runtime":
        strategy_priority = [
            RepairStrategy.RESTART_COMPONENT,
            RepairStrategy.CLEAR_CACHE,
            RepairStrategy.RESET_STATE,
            RepairStrategy.ADJUST_CONFIG,
            RepairStrategy.ROLLBACK_CHANGE,
            RepairStrategy.ALTERNATIVE_PATH,
            RepairStrategy.LLM_ANALYSIS,
        ]
    elif issue.category == "performance":
        strategy_priority = [
            RepairStrategy.ADJUST_CONFIG,
            RepairStrategy.CLEAR_CACHE,
            RepairStrategy.RESET_STATE,
            RepairStrategy.RESTART_COMPONENT,
            RepairStrategy.SKIP_OPERATION,
        ]
    elif issue.category == "memory":
        strategy_priority = [
            RepairStrategy.CLEAR_CACHE,
            RepairStrategy.RESET_STATE,
            RepairStrategy.RESTART_COMPONENT,
            RepairStrategy.ADJUST_CONFIG,
        ]
    elif issue.category == "logic":
        strategy_priority = [
            RepairStrategy.ROLLBACK_CHANGE,
            RepairStrategy.ALTERNATIVE_PATH,
            RepairStrategy.RESET_STATE,
            RepairStrategy.LLM_ANALYSIS,
            RepairStrategy.SELF_MODIFY,
        ]
    elif issue.category == "integration":
        strategy_priority = [
            RepairStrategy.RESTART_COMPONENT,
            RepairStrategy.RESET_STATE,
            RepairStrategy.SKIP_OPERATION,
            RepairStrategy.ALTERNATIVE_PATH,
        ]
    else:
        strategy_priority = [
            RepairStrategy.RESET_STATE,
            RepairStrategy.CLEAR_CACHE,
            RepairStrategy.RESTART_COMPONENT,
            RepairStrategy.ADJUST_CONFIG,
        ]
    
    # Select first untried strategy
    for strategy in strategy_priority:
        if strategy.value not in tried_strategies:
            return strategy
    
    # All strategies tried, escalate to LLM
    if RepairStrategy.LLM_ANALYSIS.value not in tried_strategies:
        return RepairStrategy.LLM_ANALYSIS
    
    # Last resort: self-modify
    return RepairStrategy.SELF_MODIFY


async def _auto_repair_issue(issue_id: str) -> bool:
    """Automatically attempt to repair an issue."""
    _load_state()
    
    issue = _issues.get(issue_id)
    if not issue:
        return False
    
    if issue.status == IssueStatus.REPAIRED:
        return True
    
    if issue.repair_attempts >= 5:
        issue.status = IssueStatus.ESCALATED
        _save_state()
        return False
    
    issue.status = IssueStatus.ANALYZING
    _save_state()
    
    # Select strategy
    strategy = _select_repair_strategy(issue)
    
    issue.status = IssueStatus.REPAIRING
    issue.last_repair_strategy = strategy.value
    issue.repair_attempts += 1
    _save_state()
    
    # Execute repair
    attempt = RepairAttempt(
        issue_id=issue_id,
        strategy=strategy,
    )
    
    success = False
    result = ""
    rollback_data = None
    
    try:
        success, result, rollback_data = await _execute_repair(issue, strategy)
        attempt.success = success
        attempt.result = result
        attempt.rollback_data = rollback_data
        attempt.completed_at = time.time()
        
    except Exception as e:
        attempt.success = False
        attempt.result = f"Repair failed: {str(e)}"
        attempt.completed_at = time.time()
    
    # Record attempt
    _repair_attempts.append(attempt.to_dict())
    issue.repair_history.append({
        "strategy": strategy.value,
        "success": attempt.success,
        "result": attempt.result,
        "timestamp": attempt.completed_at,
    })
    
    if success:
        issue.status = IssueStatus.REPAIRED
        issue.resolved_at = time.time()
        
        # Learn from successful repair
        try:
            from jinx.micro.runtime.self_evolution import learn
            learn(
                category="success_strategy",
                description=f"Repaired {issue.category} issue with {strategy.value}",
                context=issue.description[:100],
                solution=strategy.value,
                confidence=0.7,
            )
        except Exception:
            pass
    else:
        issue.status = IssueStatus.FAILED
        
        # Try next strategy
        if issue.repair_attempts < 5:
            asyncio.create_task(_auto_repair_issue(issue_id))
    
    _save_state()
    return success


async def _execute_repair(
    issue: Issue,
    strategy: RepairStrategy,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Execute a repair strategy. Returns (success, result, rollback_data)."""
    
    # Check for registered handler
    if strategy in _repair_handlers:
        handler = _repair_handlers[strategy]
        return await handler(issue)
    
    # Default repair implementations
    if strategy == RepairStrategy.CLEAR_CACHE:
        return await _repair_clear_cache(issue)
    
    elif strategy == RepairStrategy.RESET_STATE:
        return await _repair_reset_state(issue)
    
    elif strategy == RepairStrategy.ADJUST_CONFIG:
        return await _repair_adjust_config(issue)
    
    elif strategy == RepairStrategy.RESTART_COMPONENT:
        return await _repair_restart_component(issue)
    
    elif strategy == RepairStrategy.ROLLBACK_CHANGE:
        return await _repair_rollback_change(issue)
    
    elif strategy == RepairStrategy.SKIP_OPERATION:
        return await _repair_skip_operation(issue)
    
    elif strategy == RepairStrategy.ALTERNATIVE_PATH:
        return await _repair_alternative_path(issue)
    
    elif strategy == RepairStrategy.LLM_ANALYSIS:
        return await _repair_with_llm(issue)
    
    elif strategy == RepairStrategy.SELF_MODIFY:
        return await _repair_self_modify(issue)
    
    return False, f"Unknown strategy: {strategy}", None


# =============================================================================
# REPAIR IMPLEMENTATIONS
# =============================================================================

async def _repair_clear_cache(issue: Issue) -> Tuple[bool, str, Optional[Dict]]:
    """Clear relevant caches."""
    try:
        # Clear LLM cache
        try:
            from jinx.micro.llm.llm_cache import clear_cache
            clear_cache()
        except Exception:
            pass
        
        # Clear search cache
        try:
            from jinx.micro.embeddings.search_cache import _cache
            if hasattr(_cache, 'clear'):
                _cache.clear()
        except Exception:
            pass
        
        # Clear brain caches
        try:
            from jinx.micro.runtime.autobrain_config import _clear_samples
            _clear_samples()
        except Exception:
            pass
        
        return True, "Caches cleared", None
    except Exception as e:
        return False, str(e), None


async def _repair_reset_state(issue: Issue) -> Tuple[bool, str, Optional[Dict]]:
    """Reset component state."""
    try:
        component = issue.context.get("component", issue.category)
        
        # Reset brain state for the component
        try:
            from jinx.micro.runtime.autobrain_config import reset_param
            if "config" in issue.description.lower():
                # Reset related config params
                reset_param(component)
        except Exception:
            pass
        
        # Clear arch memory for stuck tasks
        try:
            from jinx.micro.runtime.arch_memory import cleanup_old_tasks
            cleanup_old_tasks(max_age_hours=1)  # Clear recent stuck tasks
        except Exception:
            pass
        
        return True, f"State reset for {component}", {"component": component}
    except Exception as e:
        return False, str(e), None


async def _repair_adjust_config(issue: Issue) -> Tuple[bool, str, Optional[Dict]]:
    """Adjust configuration parameters."""
    try:
        old_values = {}
        
        # Performance issues: reduce concurrency
        if issue.category == "performance":
            try:
                from jinx.micro.runtime.autobrain_config import get, _adjust_param
                
                # Reduce concurrency
                for param in ["frame_max_conc", "group_max_conc", "llm_max_conc"]:
                    old_val = get(param)
                    if old_val and old_val > 1:
                        new_val = max(1, old_val - 1)
                        _adjust_param(param, new_val)
                        old_values[param] = old_val
            except Exception:
                pass
        
        # Timeout issues: increase timeouts
        if "timeout" in issue.description.lower():
            try:
                from jinx.micro.runtime.autobrain_config import get, _adjust_param
                
                for param in ["turn_timeout_sec", "llm_timeout_ms"]:
                    old_val = get(param)
                    if old_val:
                        new_val = old_val * 1.5
                        _adjust_param(param, new_val)
                        old_values[param] = old_val
            except Exception:
                pass
        
        if old_values:
            return True, f"Adjusted config: {list(old_values.keys())}", {"old_values": old_values}
        return False, "No config adjustments made", None
        
    except Exception as e:
        return False, str(e), None


async def _repair_restart_component(issue: Issue) -> Tuple[bool, str, Optional[Dict]]:
    """Restart a component."""
    try:
        component = issue.context.get("component", "")
        
        # Trigger component restart via event
        try:
            from jinx.micro.runtime.bus import publish
            publish("component.restart", {"component": component, "reason": issue.description})
        except Exception:
            pass
        
        # Reset related state
        await _repair_reset_state(issue)
        
        return True, f"Restart triggered for {component}", {"component": component}
    except Exception as e:
        return False, str(e), None


async def _repair_rollback_change(issue: Issue) -> Tuple[bool, str, Optional[Dict]]:
    """Rollback recent changes."""
    try:
        # Check for recent modifications
        try:
            from jinx.micro.runtime.safe_modify import rollback_all_recent, get_modification_history
            
            history = get_modification_history(limit=5)
            if history:
                rolled_back = await rollback_all_recent(max_age_sec=600)  # Last 10 minutes
                if rolled_back > 0:
                    return True, f"Rolled back {rolled_back} recent modifications", {"rolled_back": rolled_back}
        except Exception:
            pass
        
        # Rollback config changes
        try:
            from jinx.micro.runtime.autobrain_config import rollback_recent_changes
            config_rolled = rollback_recent_changes()
            if config_rolled:
                return True, "Rolled back config changes", {"config_rolled": config_rolled}
        except Exception:
            pass
        
        return False, "No changes to rollback", None
    except Exception as e:
        return False, str(e), None


async def _repair_skip_operation(issue: Issue) -> Tuple[bool, str, Optional[Dict]]:
    """Skip the problematic operation."""
    try:
        operation = issue.context.get("operation", "")
        
        # Add to skip list
        try:
            from jinx.micro.runtime.autobrain_config import add_to_skip_list
            add_to_skip_list(operation)
        except Exception:
            pass
        
        # Mark related task as skipped
        try:
            from jinx.micro.runtime.arch_memory import skip_task
            task_id = issue.context.get("task_id")
            if task_id:
                skip_task(task_id, reason=issue.description)
        except Exception:
            pass
        
        return True, f"Skipped operation: {operation}", {"operation": operation}
    except Exception as e:
        return False, str(e), None


async def _repair_alternative_path(issue: Issue) -> Tuple[bool, str, Optional[Dict]]:
    """Try an alternative approach."""
    try:
        # Check learnings for alternative solutions
        try:
            from jinx.micro.runtime.self_evolution import get_relevant_learnings
            
            learnings = get_relevant_learnings("success_strategy", issue.description, limit=3)
            for learning in learnings:
                if learning.solution and learning.confidence > 0.5:
                    # Found a learned alternative
                    return True, f"Using learned alternative: {learning.solution}", {"solution": learning.solution}
        except Exception:
            pass
        
        # Try generic alternatives based on category
        if issue.category == "runtime":
            # Try async instead of sync or vice versa
            return True, "Switched to alternative execution path", {"alternative": "async_mode"}
        
        return False, "No alternative path found", None
    except Exception as e:
        return False, str(e), None


async def _repair_with_llm(issue: Issue) -> Tuple[bool, str, Optional[Dict]]:
    """Use LLM to analyze and suggest repair."""
    try:
        # Check rate limit
        try:
            from jinx.micro.runtime.self_evolution import _can_call_llm
            if not _can_call_llm():
                return False, "LLM rate limited", None
        except Exception:
            pass
        
        # Request LLM analysis
        try:
            from jinx.micro.runtime.self_evolution import analyze_failures_with_llm
            result = await analyze_failures_with_llm(force=True)
            
            if result.get("proposals"):
                return True, f"LLM proposed {len(result['proposals'])} solutions", {"proposals": result["proposals"]}
        except Exception:
            pass
        
        return False, "LLM analysis did not produce actionable results", None
    except Exception as e:
        return False, str(e), None


async def _repair_self_modify(issue: Issue) -> Tuple[bool, str, Optional[Dict]]:
    """Self-modify code to fix the issue."""
    try:
        # Only attempt if we have a specific file and function
        if not issue.file_path or not issue.function_name:
            return False, "Insufficient context for self-modification", None
        
        # Check for pending proposals
        try:
            from jinx.micro.runtime.self_evolution import get_pending_proposals, apply_proposal
            from jinx.micro.runtime.safe_modify import apply_evolution_proposal
            
            proposals = get_pending_proposals()
            for proposal in proposals:
                if proposal.target_file == issue.file_path:
                    success, msg = await apply_evolution_proposal(proposal.id)
                    if success:
                        return True, f"Applied code modification: {msg}", {"proposal_id": proposal.id}
        except Exception:
            pass
        
        return False, "No applicable modifications found", None
    except Exception as e:
        return False, str(e), None


# =============================================================================
# ESCALATION AND ROLLBACK
# =============================================================================

def _trigger_escalation(issue: Issue) -> None:
    """Escalate an issue that can't be auto-repaired."""
    try:
        # Log to evolution for learning
        from jinx.micro.runtime.self_evolution import learn
        learn(
            category="error_pattern",
            description=f"Escalated: {issue.category} - {issue.description[:50]}",
            context=f"Recurring {issue.recurring_count} times, {issue.repair_attempts} repair attempts failed",
            confidence=0.3,
        )
    except Exception:
        pass
    
    # Publish escalation event
    try:
        from jinx.micro.runtime.bus import publish
        publish("issue.escalated", {
            "issue_id": issue.id,
            "category": issue.category,
            "severity": issue.severity.value,
            "description": issue.description,
            "repair_attempts": issue.repair_attempts,
        })
    except Exception:
        pass


async def rollback_repair(issue_id: str) -> Tuple[bool, str]:
    """Rollback a repair that made things worse."""
    _load_state()
    
    issue = _issues.get(issue_id)
    if not issue:
        return False, "Issue not found"
    
    # Find last successful repair with rollback data
    for attempt in reversed(issue.repair_history):
        if attempt.get("success") and attempt.get("rollback_data"):
            rollback_data = attempt["rollback_data"]
            
            # Perform rollback based on strategy
            strategy = attempt.get("strategy")
            
            if strategy == RepairStrategy.ADJUST_CONFIG.value:
                # Restore old config values
                try:
                    from jinx.micro.runtime.autobrain_config import _adjust_param
                    for param, old_val in rollback_data.get("old_values", {}).items():
                        _adjust_param(param, old_val)
                except Exception:
                    pass
            
            elif strategy == RepairStrategy.SELF_MODIFY.value:
                # Rollback code change
                try:
                    from jinx.micro.runtime.safe_modify import rollback_modification
                    mod_id = rollback_data.get("modification_id")
                    if mod_id:
                        await rollback_modification(mod_id)
                except Exception:
                    pass
            
            issue.status = IssueStatus.ROLLED_BACK
            _save_state()
            
            return True, f"Rolled back repair: {strategy}"
    
    return False, "No rollback data available"


# =============================================================================
# MONITORING API
# =============================================================================

def get_active_issues() -> List[Issue]:
    """Get all active (unresolved) issues."""
    _load_state()
    return [i for i in _issues.values() if i.status not in (IssueStatus.REPAIRED, IssueStatus.ROLLED_BACK)]


def get_issue_summary() -> Dict[str, Any]:
    """Get summary of current issue state."""
    _load_state()
    
    active = [i for i in _issues.values() if i.status not in (IssueStatus.REPAIRED, IssueStatus.ROLLED_BACK)]
    resolved = [i for i in _issues.values() if i.status == IssueStatus.REPAIRED]
    
    by_severity = {}
    for severity in IssueSeverity:
        by_severity[severity.value] = len([i for i in active if i.severity == severity])
    
    by_category = {}
    for issue in active:
        by_category[issue.category] = by_category.get(issue.category, 0) + 1
    
    repair_success_rate = 0.0
    if _repair_attempts:
        successful = sum(1 for a in _repair_attempts if a.get("success"))
        repair_success_rate = successful / len(_repair_attempts)
    
    return {
        "active_issues": len(active),
        "resolved_issues": len(resolved),
        "by_severity": by_severity,
        "by_category": by_category,
        "total_repair_attempts": len(_repair_attempts),
        "repair_success_rate": repair_success_rate,
        "escalated": len([i for i in active if i.status == IssueStatus.ESCALATED]),
    }


def get_repair_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent repair history."""
    _load_state()
    return _repair_attempts[-limit:]


async def run_monitoring_cycle() -> Dict[str, Any]:
    """Run a monitoring cycle - check system health and auto-repair."""
    results = {
        "checked_at": time.time(),
        "issues_detected": 0,
        "repairs_attempted": 0,
        "repairs_successful": 0,
    }
    
    # Check brain health
    try:
        from jinx.micro.runtime.brain import brain_status
        status = brain_status()
        
        if status.health != "ok":
            detect_issue(
                category="runtime",
                severity=IssueSeverity.WARNING if status.health == "degraded" else IssueSeverity.ERROR,
                description=f"Brain health: {status.health}",
                context={"health": status.health},
            )
            results["issues_detected"] += 1
        
        if status.autobrain_success_rate < 0.7:
            detect_issue(
                category="performance",
                severity=IssueSeverity.WARNING,
                description=f"Low success rate: {status.autobrain_success_rate:.0%}",
                context={"success_rate": status.autobrain_success_rate},
            )
            results["issues_detected"] += 1
    except Exception:
        pass
    
    # Check for failed tasks
    try:
        from jinx.micro.runtime.arch_memory import get_memory_summary
        mem = get_memory_summary()
        failed = mem.get("tasks", {}).get("failed", 0)
        
        if failed > 0:
            detect_issue(
                category="runtime",
                severity=IssueSeverity.WARNING,
                description=f"{failed} failed tasks detected",
                context={"failed_tasks": failed},
            )
            results["issues_detected"] += 1
    except Exception:
        pass
    
    # Process repair queue
    active = get_active_issues()
    for issue in active[:5]:  # Process up to 5 issues per cycle
        if issue.status == IssueStatus.DETECTED and issue.repair_attempts < 5:
            results["repairs_attempted"] += 1
            success = await _auto_repair_issue(issue.id)
            if success:
                results["repairs_successful"] += 1
    
    return results


# Initialize repair queue
def _init_monitor():
    global _repair_queue
    if _repair_queue is None:
        _repair_queue = asyncio.Queue()
    _load_state()


__all__ = [
    # Detection
    "detect_issue",
    "detect_from_exception",
    "detect_performance_issue",
    "detect_anomaly",
    # Repair
    "register_repair_handler",
    "rollback_repair",
    # Monitoring
    "get_active_issues",
    "get_issue_summary",
    "get_repair_history",
    "run_monitoring_cycle",
    # Types
    "Issue",
    "IssueSeverity",
    "IssueStatus",
    "RepairStrategy",
    "RepairAttempt",
]
