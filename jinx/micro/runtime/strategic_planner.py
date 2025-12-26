"""Strategic Planner - Decompose complex tasks into executable subtasks.

Enables Jinx to:
- Break down complex goals into smaller achievable steps
- Track dependencies between subtasks
- Learn optimal task decomposition patterns
- Prioritize based on urgency and dependencies
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"  # Dependencies met
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    """A subtask in a strategic plan."""
    id: str
    description: str
    parent_plan_id: str
    order: int
    status: SubtaskStatus = SubtaskStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    estimated_complexity: str = "medium"  # low, medium, high
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class StrategicPlan:
    """A strategic plan with subtasks."""
    id: str
    goal: str
    subtasks: List[Subtask] = field(default_factory=list)
    status: str = "active"  # active, completed, failed, paused
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    success_rate: float = 0.0


# Active plans
_plans: Dict[str, StrategicPlan] = {}


def _generate_id(prefix: str, content: str) -> str:
    return f"{prefix}_{hashlib.md5(content.encode()).hexdigest()[:8]}"


# Task decomposition patterns learned from experience
_DECOMPOSITION_PATTERNS = {
    "implement": [
        ("Analyze requirements", "low"),
        ("Design solution", "medium"),
        ("Implement core logic", "high"),
        ("Add error handling", "medium"),
        ("Verify implementation", "medium"),
    ],
    "fix": [
        ("Identify root cause", "medium"),
        ("Design fix", "medium"),
        ("Apply fix", "medium"),
        ("Test fix", "low"),
    ],
    "refactor": [
        ("Analyze current code", "medium"),
        ("Plan refactoring steps", "medium"),
        ("Apply changes incrementally", "high"),
        ("Verify behavior unchanged", "medium"),
    ],
    "create": [
        ("Define structure", "low"),
        ("Implement components", "high"),
        ("Connect components", "medium"),
        ("Test integration", "medium"),
    ],
    "optimize": [
        ("Profile performance", "medium"),
        ("Identify bottlenecks", "medium"),
        ("Apply optimizations", "high"),
        ("Measure improvement", "low"),
    ],
    "integrate": [
        ("Understand interfaces", "medium"),
        ("Design integration points", "medium"),
        ("Implement connectors", "high"),
        ("Test integration", "medium"),
    ],
    "debug": [
        ("Reproduce issue", "medium"),
        ("Add diagnostics", "low"),
        ("Trace execution", "medium"),
        ("Apply fix", "medium"),
    ],
}


def detect_task_type(goal: str) -> str:
    """Detect task type from goal description."""
    goal_lower = goal.lower()
    for task_type in _DECOMPOSITION_PATTERNS.keys():
        if task_type in goal_lower:
            return task_type
    return "implement"  # Default


def create_plan(goal: str, custom_subtasks: Optional[List[str]] = None) -> StrategicPlan:
    """Create a strategic plan for a goal."""
    plan_id = _generate_id("plan", goal + str(time.time()))
    
    subtasks = []
    
    if custom_subtasks:
        # Use provided subtasks
        for i, desc in enumerate(custom_subtasks):
            subtask = Subtask(
                id=_generate_id("sub", f"{plan_id}_{i}"),
                description=desc,
                parent_plan_id=plan_id,
                order=i,
                depends_on=[subtasks[-1].id] if subtasks else [],
            )
            subtasks.append(subtask)
    else:
        # Auto-decompose based on patterns
        task_type = detect_task_type(goal)
        pattern = _DECOMPOSITION_PATTERNS.get(task_type, _DECOMPOSITION_PATTERNS["implement"])
        
        for i, (desc, complexity) in enumerate(pattern):
            subtask = Subtask(
                id=_generate_id("sub", f"{plan_id}_{i}"),
                description=f"{desc}: {goal[:50]}",
                parent_plan_id=plan_id,
                order=i,
                estimated_complexity=complexity,
                depends_on=[subtasks[-1].id] if subtasks else [],
            )
            subtasks.append(subtask)
    
    # Mark first subtask as ready
    if subtasks:
        subtasks[0].status = SubtaskStatus.READY
    
    plan = StrategicPlan(
        id=plan_id,
        goal=goal,
        subtasks=subtasks,
    )
    
    _plans[plan_id] = plan
    
    # Register with arch_memory
    try:
        from jinx.micro.runtime.arch_memory import create_task
        for sub in subtasks:
            create_task(
                sub.description,
                parent_id=plan_id,
                depends_on=sub.depends_on,
                metadata={"complexity": sub.estimated_complexity},
            )
    except Exception:
        pass
    
    return plan


def get_plan(plan_id: str) -> Optional[StrategicPlan]:
    """Get a plan by ID."""
    return _plans.get(plan_id)


def get_next_subtask(plan_id: str) -> Optional[Subtask]:
    """Get the next ready subtask to execute."""
    plan = _plans.get(plan_id)
    if not plan:
        return None
    
    for subtask in plan.subtasks:
        if subtask.status == SubtaskStatus.READY:
            return subtask
    
    return None


def start_subtask(subtask_id: str) -> bool:
    """Mark a subtask as running."""
    for plan in _plans.values():
        for subtask in plan.subtasks:
            if subtask.id == subtask_id:
                subtask.status = SubtaskStatus.RUNNING
                subtask.started_at = time.time()
                return True
    return False


def complete_subtask(subtask_id: str, success: bool, result: str = "") -> bool:
    """Complete a subtask and update dependencies."""
    for plan in _plans.values():
        for i, subtask in enumerate(plan.subtasks):
            if subtask.id == subtask_id:
                subtask.status = SubtaskStatus.DONE if success else SubtaskStatus.FAILED
                subtask.completed_at = time.time()
                subtask.result = result if success else None
                subtask.error = result if not success else None
                
                # Update dependent subtasks
                if success:
                    for other in plan.subtasks:
                        if subtask_id in other.depends_on:
                            # Check if all dependencies are done
                            all_done = all(
                                any(s.id == dep and s.status == SubtaskStatus.DONE 
                                    for s in plan.subtasks)
                                for dep in other.depends_on
                            )
                            if all_done and other.status == SubtaskStatus.PENDING:
                                other.status = SubtaskStatus.READY
                
                # Check if plan is complete
                _update_plan_status(plan)
                
                # Learn from outcome
                try:
                    from jinx.micro.runtime.self_evolution import learn
                    if success:
                        learn(
                            "success_strategy",
                            f"Subtask completed: {subtask.description[:40]}",
                            f"Complexity: {subtask.estimated_complexity}",
                            confidence=0.6,
                        )
                except Exception:
                    pass
                
                return True
    return False


def _update_plan_status(plan: StrategicPlan) -> None:
    """Update plan status based on subtasks."""
    done_count = sum(1 for s in plan.subtasks if s.status == SubtaskStatus.DONE)
    failed_count = sum(1 for s in plan.subtasks if s.status == SubtaskStatus.FAILED)
    total = len(plan.subtasks)
    
    if done_count == total:
        plan.status = "completed"
        plan.completed_at = time.time()
        plan.success_rate = 1.0
    elif failed_count > 0 and (done_count + failed_count) == total:
        plan.status = "failed"
        plan.completed_at = time.time()
        plan.success_rate = done_count / total if total > 0 else 0.0
    else:
        plan.success_rate = done_count / total if total > 0 else 0.0


def get_plan_progress(plan_id: str) -> Dict[str, Any]:
    """Get progress summary for a plan."""
    plan = _plans.get(plan_id)
    if not plan:
        return {"error": "Plan not found"}
    
    subtask_status = {}
    for status in SubtaskStatus:
        subtask_status[status.value] = sum(1 for s in plan.subtasks if s.status == status)
    
    return {
        "plan_id": plan_id,
        "goal": plan.goal,
        "status": plan.status,
        "subtasks": subtask_status,
        "progress": plan.success_rate,
        "current": get_next_subtask(plan_id),
    }


def get_active_plans() -> List[StrategicPlan]:
    """Get all active plans."""
    return [p for p in _plans.values() if p.status == "active"]


async def execute_plan(plan_id: str, executor: Any = None) -> Tuple[bool, str]:
    """Execute all subtasks in a plan sequentially.
    
    executor: async function that takes subtask description and returns (success, result)
    """
    plan = _plans.get(plan_id)
    if not plan:
        return False, "Plan not found"
    
    results = []
    
    while True:
        subtask = get_next_subtask(plan_id)
        if not subtask:
            break
        
        start_subtask(subtask.id)
        
        if executor:
            try:
                success, result = await executor(subtask.description)
            except Exception as e:
                success, result = False, str(e)
        else:
            # Default: just mark as done
            success, result = True, "auto-completed"
        
        complete_subtask(subtask.id, success, result)
        results.append((subtask.description, success, result))
        
        if not success:
            return False, f"Failed at: {subtask.description}"
    
    plan = _plans.get(plan_id)
    return plan.status == "completed", f"Completed {len(results)} subtasks"


def build_plan_context() -> str:
    """Build context block for LLM prompts with active plans."""
    active = get_active_plans()
    if not active:
        return ""
    
    lines = ["<strategic_plans>"]
    for plan in active[:3]:
        progress = get_plan_progress(plan.id)
        current = progress.get("current")
        lines.append(f"Plan: {plan.goal[:50]}")
        lines.append(f"  Progress: {progress['progress']:.0%}")
        if current:
            lines.append(f"  Current: {current.description[:40]}")
    lines.append("</strategic_plans>")
    
    return "\n".join(lines)


__all__ = [
    "create_plan",
    "get_plan",
    "get_next_subtask",
    "start_subtask",
    "complete_subtask",
    "get_plan_progress",
    "get_active_plans",
    "execute_plan",
    "build_plan_context",
    "StrategicPlan",
    "Subtask",
    "SubtaskStatus",
]
