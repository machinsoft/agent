"""Unified Brain API - Single entry point to Jinx autonomous intelligence.

Provides unified access to:
- AutoBrain: adaptive configuration
- Arch Memory: task tracking and context
- Self Evolution: goals, learnings, self-improvement
- Intelligent Retry: evolution-driven retry
- Safe Modify: self-modification with rollback

Usage:
    from jinx.micro.runtime.brain import Brain
    
    brain = Brain()
    brain.start()  # Initialize all systems
    
    # Track work
    brain.begin_task("implement feature X")
    brain.complete_task(success=True)
    
    # Get status
    print(brain.status())
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Awaitable, TypeVar

T = TypeVar("T")


@dataclass
class BrainStatus:
    """Current status of the brain systems."""
    health: str  # "ok", "degraded", "critical"
    autobrain_samples: int
    autobrain_success_rate: float
    tasks_active: int
    tasks_completed: int
    goals_active: int
    goals_achieved: int
    learnings_total: int
    llm_calls: int
    self_modifications: int
    uptime_sec: float


class Brain:
    """Unified interface to Jinx autonomous intelligence systems."""
    
    _instance: Optional["Brain"] = None
    _start_time: float = 0.0
    _initialized: bool = False
    _current_task_id: Optional[str] = None
    _current_goal_id: Optional[str] = None
    
    def __new__(cls) -> "Brain":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def start(self) -> None:
        """Initialize all brain systems."""
        if self._initialized:
            return
        
        self._start_time = time.time()
        
        # Initialize AutoBrain
        try:
            from jinx.micro.runtime.autobrain_config import _init as _ab_init
            _ab_init()
        except Exception:
            pass
        
        # Initialize Arch Memory
        try:
            from jinx.micro.runtime.arch_memory import _init as _am_init
            _am_init()
        except Exception:
            pass
        
        # Initialize Self Evolution
        try:
            from jinx.micro.runtime.self_evolution import _init as _se_init, initialize_system_goals
            _se_init()
            initialize_system_goals()
        except Exception:
            pass
        
        self._initialized = True
    
    def status(self) -> BrainStatus:
        """Get current brain status."""
        self.start()
        
        # AutoBrain status
        ab_samples = 0
        ab_success = 0.5
        try:
            from jinx.micro.runtime.autobrain_config import get_diagnostics
            diag = get_diagnostics()
            ab_samples = diag.get("total_samples", 0)
            ab_success = diag.get("avg_success_rate", 0.5)
            health = diag.get("health", "ok")
        except Exception:
            health = "unknown"
        
        # Arch Memory status
        tasks_active = 0
        tasks_completed = 0
        try:
            from jinx.micro.runtime.arch_memory import get_memory_summary
            mem = get_memory_summary()
            tasks = mem.get("tasks", {})
            tasks_active = tasks.get("in_progress", 0) + tasks.get("pending", 0)
            tasks_completed = tasks.get("completed", 0)
        except Exception:
            pass
        
        # Evolution status
        goals_active = 0
        goals_achieved = 0
        learnings = 0
        llm_calls = 0
        self_mods = 0
        try:
            from jinx.micro.runtime.self_evolution import get_evolution_summary
            evo = get_evolution_summary()
            goals = evo.get("goals", {})
            goals_active = goals.get("active", 0)
            goals_achieved = goals.get("achieved", 0)
            learnings = evo.get("learnings", {}).get("total", 0)
            metrics = evo.get("metrics", {})
            llm_calls = metrics.get("llm_calls", 0)
            self_mods = metrics.get("self_modifications", 0)
        except Exception:
            pass
        
        return BrainStatus(
            health=health,
            autobrain_samples=ab_samples,
            autobrain_success_rate=ab_success,
            tasks_active=tasks_active,
            tasks_completed=tasks_completed,
            goals_active=goals_active,
            goals_achieved=goals_achieved,
            learnings_total=learnings,
            llm_calls=llm_calls,
            self_modifications=self_mods,
            uptime_sec=time.time() - self._start_time if self._start_time else 0,
        )
    
    def begin_task(self, description: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Begin tracking a new task."""
        self.start()
        
        # Create task in arch memory
        try:
            from jinx.micro.runtime.arch_memory import create_task, start_task, update_context
            task_id = create_task(description, metadata=metadata or {})
            start_task(task_id)
            update_context(add_query=description[:200], push_intent=f"task:{task_id[:8]}")
            self._current_task_id = task_id
        except Exception:
            self._current_task_id = None
        
        # Detect and create goal
        try:
            from jinx.micro.runtime.self_evolution import track_user_request
            _, goal_id = track_user_request(description)
            self._current_goal_id = goal_id
        except Exception:
            self._current_goal_id = None
        
        return self._current_task_id or ""
    
    def complete_task(self, success: bool = True, result: str = "") -> None:
        """Complete the current task."""
        self.start()
        
        # Complete in arch memory
        if self._current_task_id:
            try:
                from jinx.micro.runtime.arch_memory import complete_task, fail_task, update_context
                if success:
                    complete_task(self._current_task_id, result)
                else:
                    fail_task(self._current_task_id, result)
                update_context(pop_intent=True)
            except Exception:
                pass
        
        # Update evolution goal
        if self._current_goal_id:
            try:
                from jinx.micro.runtime.self_evolution import complete_user_request
                complete_user_request("", self._current_goal_id, success, result)
            except Exception:
                pass
        
        # Learn from outcome
        try:
            from jinx.micro.runtime.self_evolution import learn
            if success:
                learn("success_strategy", "Task completed", result[:100], confidence=0.6)
            else:
                learn("error_pattern", "Task failed", result[:100], confidence=0.4)
        except Exception:
            pass
        
        self._current_task_id = None
        self._current_goal_id = None
    
    def set_goal(self, description: str, priority: str = "medium") -> str:
        """Set a new goal to achieve."""
        self.start()
        
        try:
            from jinx.micro.runtime.self_evolution import set_goal, GoalPriority
            goal_id = set_goal(description, f"Goal: {description}", GoalPriority(priority))
            return goal_id
        except Exception:
            return ""
    
    def learn(self, category: str, description: str, context: str, solution: Optional[str] = None) -> str:
        """Record a learning."""
        self.start()
        
        try:
            from jinx.micro.runtime.self_evolution import learn as _learn
            return _learn(category, description, context, solution)
        except Exception:
            return ""
    
    def get_config(self, name: str) -> Any:
        """Get adaptive configuration value."""
        self.start()
        
        try:
            from jinx.micro.runtime.autobrain_config import get
            return get(name)
        except Exception:
            return None
    
    def record_outcome(self, param: str, success: bool, latency_ms: float = 0) -> None:
        """Record outcome for adaptive learning."""
        self.start()
        
        try:
            from jinx.micro.runtime.autobrain_config import record_outcome
            record_outcome(param, success, latency_ms)
        except Exception:
            pass
    
    async def with_retry(self, operation: str, func: Callable[[], Awaitable[T]]) -> T:
        """Execute with intelligent retry."""
        self.start()
        
        try:
            from jinx.micro.runtime.intelligent_retry import with_intelligent_retry
            return await with_intelligent_retry(operation, func)
        except Exception:
            return await func()
    
    async def safe_modify(
        self,
        file_path: str,
        old_code: str,
        new_code: str,
    ) -> Tuple[bool, str]:
        """Safely modify code with backup and rollback."""
        self.start()
        
        try:
            from jinx.micro.runtime.safe_modify import safe_modify as _safe_modify
            success, message, _ = await _safe_modify(file_path, old_code, new_code)
            return success, message
        except Exception as e:
            return False, str(e)
    
    async def analyze_and_improve(self, force: bool = False) -> Dict[str, Any]:
        """Run evolution cycle to analyze failures and improve."""
        self.start()
        
        try:
            from jinx.micro.runtime.self_evolution import run_evolution_cycle
            return await run_evolution_cycle()
        except Exception as e:
            return {"error": str(e)}
    
    def create_plan(self, goal: str) -> str:
        """Create a strategic plan with auto-decomposed subtasks."""
        self.start()
        
        try:
            from jinx.micro.runtime.strategic_planner import create_plan as _create_plan
            plan = _create_plan(goal)
            return plan.id
        except Exception:
            return ""
    
    def get_plan_progress(self, plan_id: str) -> Dict[str, Any]:
        """Get progress of a strategic plan."""
        self.start()
        
        try:
            from jinx.micro.runtime.strategic_planner import get_plan_progress as _get_progress
            return _get_progress(plan_id)
        except Exception:
            return {}
    
    def next_subtask(self, plan_id: str) -> Optional[str]:
        """Get next ready subtask description."""
        self.start()
        
        try:
            from jinx.micro.runtime.strategic_planner import get_next_subtask
            subtask = get_next_subtask(plan_id)
            return subtask.description if subtask else None
        except Exception:
            return None
    
    def complete_subtask(self, subtask_id: str, success: bool, result: str = "") -> None:
        """Complete a subtask in a plan."""
        self.start()
        
        try:
            from jinx.micro.runtime.strategic_planner import complete_subtask as _complete
            _complete(subtask_id, success, result)
        except Exception:
            pass
    
    def get_context_block(self) -> str:
        """Get combined context block for LLM prompts."""
        self.start()
        
        blocks = []
        
        try:
            from jinx.micro.runtime.arch_memory import build_context_block
            block = build_context_block()
            if block:
                blocks.append(block)
        except Exception:
            pass
        
        try:
            from jinx.micro.runtime.self_evolution import build_evolution_context
            block = build_evolution_context()
            if block:
                blocks.append(block)
        except Exception:
            pass
        
        try:
            from jinx.micro.runtime.strategic_planner import build_plan_context
            block = build_plan_context()
            if block:
                blocks.append(block)
        except Exception:
            pass
        
        return "\n".join(blocks)
    
    def dashboard(self) -> str:
        """Get detailed brain dashboard with insights."""
        self.start()
        
        try:
            from jinx.micro.runtime.brain_metrics import format_dashboard
            return format_dashboard()
        except Exception:
            return self.summary()
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """Get performance insights and recommendations."""
        self.start()
        
        try:
            from jinx.micro.runtime.brain_metrics import analyze_performance
            insights = analyze_performance()
            return [
                {"category": i.category, "severity": i.severity, "message": i.message, "recommendation": i.recommendation}
                for i in insights
            ]
        except Exception:
            return []
    
    def suggest_improvements(self) -> List[Dict[str, str]]:
        """Get proactive improvement suggestions based on patterns."""
        self.start()
        
        suggestions = []
        
        # Get goal suggestions
        try:
            from jinx.micro.runtime.self_evolution import suggest_goals_from_patterns
            goal_suggestions = suggest_goals_from_patterns()
            for s in goal_suggestions:
                suggestions.append({
                    "type": "goal",
                    "description": s.get("description", ""),
                    "priority": s.get("priority", "medium"),
                    "reason": s.get("reason", ""),
                })
        except Exception:
            pass
        
        # Get high-confidence learnings that could be applied
        try:
            from jinx.micro.runtime.self_evolution import get_high_confidence_learnings
            learnings = get_high_confidence_learnings(min_confidence=0.7)
            for l in learnings[:3]:
                if l.solution:
                    suggestions.append({
                        "type": "apply_learning",
                        "description": l.description,
                        "priority": "medium",
                        "reason": f"Confidence: {l.confidence:.0%}, Solution: {l.solution[:50]}",
                    })
        except Exception:
            pass
        
        # Get pending code proposals
        try:
            from jinx.micro.runtime.self_evolution import get_pending_proposals
            proposals = get_pending_proposals()
            for p in proposals[:2]:
                suggestions.append({
                    "type": "code_change",
                    "description": p.description,
                    "priority": p.risk_level,
                    "reason": p.reason[:100],
                })
        except Exception:
            pass
        
        return suggestions
    
    def record_metrics(self, success: bool, latency_ms: float) -> None:
        """Record task metrics for analysis."""
        self.start()
        
        try:
            from jinx.micro.runtime.brain_metrics import record_task_completion
            record_task_completion(success, latency_ms)
        except Exception:
            pass
    
    def summary(self) -> str:
        """Get human-readable summary of brain state."""
        status = self.status()
        
        lines = [
            "╔═══════════════════════════════════════════════════╗",
            "║              JINX BRAIN STATUS                    ║",
            "╠═══════════════════════════════════════════════════╣",
            f"║  Health: {status.health:<10} Uptime: {status.uptime_sec:.0f}s          ║",
            f"║  AutoBrain: {status.autobrain_samples} samples, {status.autobrain_success_rate:.0%} success    ║",
            f"║  Tasks: {status.tasks_active} active, {status.tasks_completed} completed          ║",
            f"║  Goals: {status.goals_active} active, {status.goals_achieved} achieved           ║",
            f"║  Learnings: {status.learnings_total} | LLM calls: {status.llm_calls}             ║",
            "╚═══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# Singleton instance
_brain: Optional[Brain] = None


def get_brain() -> Brain:
    """Get the singleton Brain instance."""
    global _brain
    if _brain is None:
        _brain = Brain()
        _brain.start()
    return _brain


# Convenience functions
def brain_status() -> BrainStatus:
    return get_brain().status()

def brain_summary() -> str:
    return get_brain().summary()

def brain_check() -> str:
    """Quick one-line brain health check."""
    s = get_brain().status()
    return f"Brain: {s.health} | {s.autobrain_success_rate:.0%} success | {s.tasks_completed} tasks | {s.goals_active} goals | {s.learnings_total} learnings"

def begin_task(description: str) -> str:
    return get_brain().begin_task(description)

def complete_task(success: bool = True, result: str = "") -> None:
    get_brain().complete_task(success, result)


__all__ = [
    "Brain",
    "BrainStatus",
    "get_brain",
    "brain_status",
    "brain_summary",
    "brain_check",
    "begin_task",
    "complete_task",
]
