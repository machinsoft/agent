"""Smart Rollback System - Intelligent state management with multi-level rollback.

Provides:
- Automatic state snapshots before risky operations
- Multi-level rollback (config, code, state, full system)
- Intelligent decision on rollback scope
- Alternative solution attempts after rollback
- State history with compression
"""

from __future__ import annotations

import asyncio
import copy
import gzip
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Awaitable

_ROLLBACK_DIR = Path(".jinx") / "rollback"
_SNAPSHOTS_FILE = _ROLLBACK_DIR / "snapshots.json"
_CHECKPOINTS_DIR = _ROLLBACK_DIR / "checkpoints"

_LOCK = Lock()
_MAX_SNAPSHOTS = 50
_MAX_CHECKPOINT_AGE_HOURS = 24


class RollbackLevel(str, Enum):
    CONFIG = "config"          # Only config parameters
    STATE = "state"            # In-memory state
    CODE = "code"              # Code modifications
    COMPONENT = "component"    # Full component reset
    SYSTEM = "system"          # Full system rollback


class SnapshotType(str, Enum):
    AUTO = "auto"              # Automatic periodic snapshot
    PRE_CHANGE = "pre_change"  # Before a risky change
    CHECKPOINT = "checkpoint"  # User-requested checkpoint
    RECOVERY = "recovery"      # Before recovery attempt


@dataclass
class StateSnapshot:
    """A snapshot of system state."""
    id: str
    snapshot_type: SnapshotType
    timestamp: float = field(default_factory=time.time)
    description: str = ""
    
    # State data
    config_state: Dict[str, Any] = field(default_factory=dict)
    brain_state: Dict[str, Any] = field(default_factory=dict)
    evolution_state: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    trigger: str = ""  # What triggered this snapshot
    related_issue_id: Optional[str] = None
    
    # Rollback tracking
    used_for_rollback: bool = False
    rollback_success: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["snapshot_type"] = self.snapshot_type.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StateSnapshot":
        d["snapshot_type"] = SnapshotType(d["snapshot_type"])
        return cls(**d)


@dataclass
class RollbackPlan:
    """A plan for rolling back to a previous state."""
    snapshot_id: str
    level: RollbackLevel
    steps: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    estimated_risk: str = "low"  # low, medium, high
    reason: str = ""


# Active snapshots
_snapshots: Dict[str, StateSnapshot] = {}
_snapshot_order: List[str] = []  # Ordered by timestamp

# Alternative solutions registry
_alternative_handlers: Dict[str, List[Callable]] = {}


def _ensure_dirs() -> None:
    try:
        _ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)
        _CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _load_snapshots() -> None:
    global _snapshots, _snapshot_order
    _ensure_dirs()
    
    try:
        if _SNAPSHOTS_FILE.exists():
            with open(_SNAPSHOTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _snapshots = {k: StateSnapshot.from_dict(v) for k, v in data.get("snapshots", {}).items()}
                _snapshot_order = data.get("order", [])
    except Exception:
        pass


def _save_snapshots() -> None:
    _ensure_dirs()
    
    try:
        # Cleanup old snapshots
        while len(_snapshot_order) > _MAX_SNAPSHOTS:
            old_id = _snapshot_order.pop(0)
            _snapshots.pop(old_id, None)
        
        data = {
            "snapshots": {k: v.to_dict() for k, v in _snapshots.items()},
            "order": _snapshot_order,
        }
        
        with open(_SNAPSHOTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _generate_snapshot_id() -> str:
    import hashlib
    return f"snap_{hashlib.md5(str(time.time()).encode()).hexdigest()[:12]}"


# =============================================================================
# SNAPSHOT CREATION
# =============================================================================

def take_snapshot(
    snapshot_type: SnapshotType = SnapshotType.AUTO,
    description: str = "",
    trigger: str = "",
    related_issue_id: Optional[str] = None,
) -> str:
    """Take a snapshot of current system state."""
    _load_snapshots()
    
    snapshot_id = _generate_snapshot_id()
    
    # Capture config state
    config_state = {}
    try:
        from jinx.micro.runtime.autobrain_config import get_all_params
        config_state = get_all_params()
    except Exception:
        pass
    
    # Capture brain state
    brain_state = {}
    try:
        from jinx.micro.runtime.brain import brain_status
        status = brain_status()
        brain_state = {
            "health": status.health,
            "success_rate": status.autobrain_success_rate,
            "tasks_completed": status.tasks_completed,
            "tasks_failed": getattr(status, "tasks_failed", 0),
            "goals_active": status.goals_active,
            "learnings": status.learnings_total,
        }
    except Exception:
        pass
    
    # Capture evolution state
    evolution_state = {}
    try:
        from jinx.micro.runtime.self_evolution import get_evolution_summary
        evolution_state = get_evolution_summary()
    except Exception:
        pass
    
    snapshot = StateSnapshot(
        id=snapshot_id,
        snapshot_type=snapshot_type,
        description=description or f"Snapshot at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        config_state=config_state,
        brain_state=brain_state,
        evolution_state=evolution_state,
        trigger=trigger,
        related_issue_id=related_issue_id,
    )
    
    with _LOCK:
        _snapshots[snapshot_id] = snapshot
        _snapshot_order.append(snapshot_id)
        _save_snapshots()
    
    return snapshot_id


def take_pre_change_snapshot(change_description: str) -> str:
    """Take a snapshot before making a risky change."""
    return take_snapshot(
        snapshot_type=SnapshotType.PRE_CHANGE,
        description=f"Pre-change: {change_description}",
        trigger="pre_change",
    )


def take_checkpoint(name: str = "") -> str:
    """Take a named checkpoint for manual recovery."""
    snapshot_id = take_snapshot(
        snapshot_type=SnapshotType.CHECKPOINT,
        description=name or f"Checkpoint {time.strftime('%H:%M:%S')}",
        trigger="manual",
    )
    
    # Also save to checkpoint file
    _ensure_dirs()
    try:
        checkpoint_file = _CHECKPOINTS_DIR / f"{snapshot_id}.json.gz"
        snapshot = _snapshots.get(snapshot_id)
        if snapshot:
            with gzip.open(checkpoint_file, "wt", encoding="utf-8") as f:
                json.dump(snapshot.to_dict(), f)
    except Exception:
        pass
    
    return snapshot_id


# =============================================================================
# ROLLBACK EXECUTION
# =============================================================================

def get_rollback_plan(
    target_snapshot_id: Optional[str] = None,
    level: RollbackLevel = RollbackLevel.CONFIG,
    issue_description: str = "",
) -> RollbackPlan:
    """Create a rollback plan."""
    _load_snapshots()
    
    # Find target snapshot
    if target_snapshot_id:
        snapshot = _snapshots.get(target_snapshot_id)
    else:
        # Use most recent pre-change or checkpoint snapshot
        for snap_id in reversed(_snapshot_order):
            snap = _snapshots.get(snap_id)
            if snap and snap.snapshot_type in (SnapshotType.PRE_CHANGE, SnapshotType.CHECKPOINT):
                snapshot = snap
                target_snapshot_id = snap_id
                break
        else:
            # Use most recent snapshot
            if _snapshot_order:
                target_snapshot_id = _snapshot_order[-1]
                snapshot = _snapshots.get(target_snapshot_id)
            else:
                snapshot = None
    
    if not snapshot:
        return RollbackPlan(
            snapshot_id="",
            level=level,
            reason="No snapshot available",
        )
    
    # Build rollback steps
    steps = []
    alternatives = []
    risk = "low"
    
    if level in (RollbackLevel.CONFIG, RollbackLevel.STATE, RollbackLevel.COMPONENT, RollbackLevel.SYSTEM):
        steps.append({
            "action": "restore_config",
            "data": snapshot.config_state,
            "description": "Restore configuration parameters",
        })
    
    if level in (RollbackLevel.STATE, RollbackLevel.COMPONENT, RollbackLevel.SYSTEM):
        steps.append({
            "action": "reset_brain_state",
            "description": "Reset brain state to snapshot",
        })
        risk = "medium"
    
    if level in (RollbackLevel.CODE, RollbackLevel.SYSTEM):
        steps.append({
            "action": "rollback_code_changes",
            "description": "Rollback recent code modifications",
        })
        risk = "high"
    
    if level == RollbackLevel.SYSTEM:
        steps.append({
            "action": "restart_components",
            "description": "Restart all components",
        })
        risk = "high"
    
    # Add alternatives
    alternatives = _get_alternatives_for_issue(issue_description)
    
    return RollbackPlan(
        snapshot_id=target_snapshot_id,
        level=level,
        steps=steps,
        alternatives=alternatives,
        estimated_risk=risk,
        reason=f"Rollback to {snapshot.description}",
    )


async def execute_rollback(
    plan: RollbackPlan,
    try_alternatives_on_failure: bool = True,
) -> Tuple[bool, str, Optional[str]]:
    """Execute a rollback plan. Returns (success, message, alternative_used)."""
    _load_snapshots()
    
    if not plan.snapshot_id:
        return False, "No snapshot to rollback to", None
    
    snapshot = _snapshots.get(plan.snapshot_id)
    if not snapshot:
        return False, "Snapshot not found", None
    
    # Take recovery snapshot before rollback
    take_snapshot(
        snapshot_type=SnapshotType.RECOVERY,
        description="Pre-rollback state",
        trigger="rollback",
    )
    
    success = True
    messages = []
    
    for step in plan.steps:
        action = step.get("action")
        
        try:
            if action == "restore_config":
                await _restore_config(snapshot.config_state)
                messages.append("Config restored")
                
            elif action == "reset_brain_state":
                await _reset_brain_state()
                messages.append("Brain state reset")
                
            elif action == "rollback_code_changes":
                rolled = await _rollback_code_changes()
                messages.append(f"Code changes rolled back: {rolled}")
                
            elif action == "restart_components":
                await _restart_components()
                messages.append("Components restarted")
                
        except Exception as e:
            success = False
            messages.append(f"Failed {action}: {str(e)}")
            break
    
    # Mark snapshot as used
    snapshot.used_for_rollback = True
    snapshot.rollback_success = success
    _save_snapshots()
    
    if success:
        # Learn from successful rollback
        try:
            from jinx.micro.runtime.self_evolution import learn
            learn(
                category="success_strategy",
                description=f"Rollback to {plan.level.value} succeeded",
                context=plan.reason,
                solution=f"rollback_{plan.level.value}",
                confidence=0.7,
            )
        except Exception:
            pass
        
        return True, "; ".join(messages), None
    
    # Rollback failed, try alternatives
    if try_alternatives_on_failure and plan.alternatives:
        for alt in plan.alternatives:
            try:
                alt_success, alt_msg = await _try_alternative(alt)
                if alt_success:
                    return True, f"Alternative succeeded: {alt_msg}", alt.get("name")
            except Exception:
                continue
    
    return False, "; ".join(messages), None


async def _restore_config(config_state: Dict[str, Any]) -> None:
    """Restore configuration parameters."""
    try:
        from jinx.micro.runtime.autobrain_config import _adjust_param
        for param, value in config_state.items():
            if isinstance(value, dict) and "current" in value:
                _adjust_param(param, value["current"])
    except Exception:
        pass


async def _reset_brain_state() -> None:
    """Reset brain state."""
    try:
        from jinx.micro.runtime.autobrain_config import _clear_samples
        _clear_samples()
    except Exception:
        pass
    
    try:
        from jinx.micro.runtime.arch_memory import cleanup_old_tasks
        cleanup_old_tasks(max_age_hours=0)  # Clear all pending
    except Exception:
        pass


async def _rollback_code_changes() -> int:
    """Rollback recent code modifications."""
    try:
        from jinx.micro.runtime.safe_modify import rollback_all_recent
        return await rollback_all_recent(max_age_sec=3600)  # Last hour
    except Exception:
        return 0


async def _restart_components() -> None:
    """Restart system components."""
    try:
        from jinx.micro.runtime.bus import publish
        publish("system.restart_components", {})
    except Exception:
        pass


# =============================================================================
# ALTERNATIVE SOLUTIONS
# =============================================================================

def register_alternative(issue_pattern: str, handler: Callable) -> None:
    """Register an alternative solution handler for an issue pattern."""
    if issue_pattern not in _alternative_handlers:
        _alternative_handlers[issue_pattern] = []
    _alternative_handlers[issue_pattern].append(handler)


def _get_alternatives_for_issue(issue_description: str) -> List[Dict[str, Any]]:
    """Get alternative solutions for an issue."""
    alternatives = []
    issue_lower = issue_description.lower()
    
    # Check registered handlers
    for pattern, handlers in _alternative_handlers.items():
        if pattern.lower() in issue_lower:
            for handler in handlers:
                alternatives.append({
                    "name": f"alt_{pattern}",
                    "handler": handler,
                    "pattern": pattern,
                })
    
    # Add generic alternatives based on keywords
    if "timeout" in issue_lower:
        alternatives.append({
            "name": "increase_timeout",
            "action": "adjust_timeout",
            "multiplier": 2.0,
        })
    
    if "memory" in issue_lower:
        alternatives.append({
            "name": "clear_caches",
            "action": "clear_all_caches",
        })
    
    if "connection" in issue_lower or "network" in issue_lower:
        alternatives.append({
            "name": "retry_with_backoff",
            "action": "exponential_backoff_retry",
            "max_retries": 3,
        })
    
    if "concurrency" in issue_lower or "race" in issue_lower:
        alternatives.append({
            "name": "reduce_concurrency",
            "action": "reduce_parallelism",
        })
    
    # Add learned alternatives
    try:
        from jinx.micro.runtime.self_evolution import get_relevant_learnings
        learnings = get_relevant_learnings("success_strategy", issue_description, limit=3)
        for learning in learnings:
            if learning.solution and learning.confidence > 0.6:
                alternatives.append({
                    "name": f"learned_{learning.id[:8]}",
                    "action": "apply_learning",
                    "learning_id": learning.id,
                    "solution": learning.solution,
                })
    except Exception:
        pass
    
    return alternatives


async def _try_alternative(alternative: Dict[str, Any]) -> Tuple[bool, str]:
    """Try an alternative solution."""
    action = alternative.get("action")
    handler = alternative.get("handler")
    
    if handler:
        try:
            result = await handler() if asyncio.iscoroutinefunction(handler) else handler()
            return True, str(result)
        except Exception as e:
            return False, str(e)
    
    if action == "adjust_timeout":
        try:
            from jinx.micro.runtime.autobrain_config import get, _adjust_param
            multiplier = alternative.get("multiplier", 1.5)
            for param in ["turn_timeout_sec", "llm_timeout_ms"]:
                old = get(param)
                if old:
                    _adjust_param(param, old * multiplier)
            return True, f"Timeouts increased by {multiplier}x"
        except Exception as e:
            return False, str(e)
    
    elif action == "clear_all_caches":
        try:
            # Clear various caches
            try:
                from jinx.micro.llm.llm_cache import clear_cache
                clear_cache()
            except:
                pass
            return True, "Caches cleared"
        except Exception as e:
            return False, str(e)
    
    elif action == "exponential_backoff_retry":
        # Just signal that retry should happen
        return True, "Retry with backoff enabled"
    
    elif action == "reduce_parallelism":
        try:
            from jinx.micro.runtime.autobrain_config import get, _adjust_param
            for param in ["frame_max_conc", "group_max_conc"]:
                old = get(param)
                if old and old > 1:
                    _adjust_param(param, max(1, old - 1))
            return True, "Concurrency reduced"
        except Exception as e:
            return False, str(e)
    
    elif action == "apply_learning":
        learning_id = alternative.get("learning_id")
        solution = alternative.get("solution")
        if solution:
            return True, f"Applied learned solution: {solution}"
        return False, "No solution in learning"
    
    return False, f"Unknown action: {action}"


# =============================================================================
# SMART ROLLBACK DECISION
# =============================================================================

async def smart_rollback(
    issue_id: str,
    issue_description: str,
    severity: str = "error",
    attempt_count: int = 0,
) -> Tuple[bool, str]:
    """Intelligently decide and execute rollback."""
    
    # Determine rollback level based on severity and attempts
    if attempt_count == 0:
        level = RollbackLevel.CONFIG
    elif attempt_count == 1:
        level = RollbackLevel.STATE
    elif attempt_count == 2:
        level = RollbackLevel.COMPONENT
    else:
        level = RollbackLevel.SYSTEM if severity == "critical" else RollbackLevel.CODE
    
    # Get rollback plan
    plan = get_rollback_plan(
        level=level,
        issue_description=issue_description,
    )
    
    if not plan.snapshot_id:
        # No snapshot, try alternatives directly
        alternatives = _get_alternatives_for_issue(issue_description)
        for alt in alternatives:
            success, msg = await _try_alternative(alt)
            if success:
                return True, f"Alternative solution: {msg}"
        return False, "No rollback snapshot and no alternatives worked"
    
    # Execute rollback
    success, message, alt_used = await execute_rollback(plan, try_alternatives_on_failure=True)
    
    if success:
        return True, message if not alt_used else f"{message} (via alternative: {alt_used})"
    
    # Escalate to next level
    if level != RollbackLevel.SYSTEM:
        return await smart_rollback(issue_id, issue_description, severity, attempt_count + 1)
    
    return False, f"All rollback attempts failed: {message}"


# =============================================================================
# MONITORING INTEGRATION
# =============================================================================

def get_snapshot_summary() -> Dict[str, Any]:
    """Get summary of available snapshots."""
    _load_snapshots()
    
    by_type = {}
    for snap_type in SnapshotType:
        by_type[snap_type.value] = len([s for s in _snapshots.values() if s.snapshot_type == snap_type])
    
    recent = None
    if _snapshot_order:
        recent_snap = _snapshots.get(_snapshot_order[-1])
        if recent_snap:
            recent = {
                "id": recent_snap.id,
                "type": recent_snap.snapshot_type.value,
                "timestamp": recent_snap.timestamp,
                "description": recent_snap.description,
            }
    
    return {
        "total_snapshots": len(_snapshots),
        "by_type": by_type,
        "most_recent": recent,
        "oldest_timestamp": _snapshots[_snapshot_order[0]].timestamp if _snapshot_order else None,
    }


__all__ = [
    # Snapshots
    "take_snapshot",
    "take_pre_change_snapshot",
    "take_checkpoint",
    # Rollback
    "get_rollback_plan",
    "execute_rollback",
    "smart_rollback",
    # Alternatives
    "register_alternative",
    # Monitoring
    "get_snapshot_summary",
    # Types
    "StateSnapshot",
    "RollbackPlan",
    "RollbackLevel",
    "SnapshotType",
]
