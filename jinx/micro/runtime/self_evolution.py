"""Self-Evolution Engine - Goal-driven self-improvement system.

Jinx evolves to achieve goals at any cost:
- Tracks user goals and system goals (self-improvement)
- Analyzes failures and learns from them
- Proposes and applies fixes to its own code when needed
- Uses LLM sparingly for strategic decisions only
- Rebuilds architecture if necessary to succeed

Core principles:
1. Goal achievement is paramount
2. Learn from every failure
3. Minimal LLM calls - batch analysis, not per-request
4. Self-modification only when other options exhausted
5. Persistent progress - never lose learned improvements
"""

from __future__ import annotations

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from threading import Lock
from enum import Enum

_EVOLUTION_DIR = Path(".jinx") / "evolution"
_GOALS_PATH = _EVOLUTION_DIR / "goals.json"
_LEARNINGS_PATH = _EVOLUTION_DIR / "learnings.json"
_PROPOSALS_PATH = _EVOLUTION_DIR / "proposals.json"
_METRICS_PATH = _EVOLUTION_DIR / "metrics.json"

_LOCK = Lock()

# Rate limiting for LLM calls - max 1 analysis per 5 minutes
_LAST_LLM_ANALYSIS = 0.0
_LLM_COOLDOWN_SEC = 300.0


class GoalStatus(str, Enum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    FAILED = "failed"
    PAUSED = "paused"


class GoalPriority(str, Enum):
    CRITICAL = "critical"  # Must achieve
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Goal:
    """A goal to achieve."""
    id: str
    description: str
    success_criteria: str
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.ACTIVE
    progress: float = 0.0  # 0-1
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    created_at: float = field(default_factory=time.time)
    last_attempt: Optional[float] = None
    achieved_at: Optional[float] = None
    failure_patterns: List[str] = field(default_factory=list)
    learned_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["priority"] = self.priority.value
        d["status"] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Goal":
        d = dict(d)
        d["priority"] = GoalPriority(d.get("priority", "medium"))
        d["status"] = GoalStatus(d.get("status", "active"))
        return cls(**d)


@dataclass
class Learning:
    """A learned insight from experience."""
    id: str
    category: str  # "error_pattern", "success_strategy", "optimization", "architecture"
    description: str
    context: str
    solution: Optional[str] = None
    confidence: float = 0.5  # 0-1
    applications: int = 0
    success_rate: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_applied: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Learning":
        return cls(**d)


@dataclass
class CodeProposal:
    """A proposed code modification."""
    id: str
    target_file: str
    description: str
    reason: str
    old_code: str
    new_code: str
    risk_level: str  # "low", "medium", "high", "critical"
    status: str = "pending"  # "pending", "approved", "applied", "rejected", "failed"
    created_at: float = field(default_factory=time.time)
    applied_at: Optional[float] = None
    result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CodeProposal":
        return cls(**d)


# =============================================================================
# GLOBAL STATE
# =============================================================================

_goals: Dict[str, Goal] = {}
_learnings: Dict[str, Learning] = {}
_proposals: Dict[str, CodeProposal] = {}
_metrics: Dict[str, Any] = {
    "total_attempts": 0,
    "total_successes": 0,
    "total_failures": 0,
    "llm_calls": 0,
    "self_modifications": 0,
    "evolution_cycles": 0,
}
_initialized = False


def _ensure_dirs() -> None:
    try:
        _EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _load_all() -> None:
    global _goals, _learnings, _proposals, _metrics, _initialized
    if _initialized:
        return
    
    _ensure_dirs()
    
    try:
        if _GOALS_PATH.exists():
            with open(_GOALS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _goals = {k: Goal.from_dict(v) for k, v in data.items()}
    except Exception:
        _goals = {}
    
    try:
        if _LEARNINGS_PATH.exists():
            with open(_LEARNINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _learnings = {k: Learning.from_dict(v) for k, v in data.items()}
    except Exception:
        _learnings = {}
    
    try:
        if _PROPOSALS_PATH.exists():
            with open(_PROPOSALS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _proposals = {k: CodeProposal.from_dict(v) for k, v in data.items()}
    except Exception:
        _proposals = {}
    
    try:
        if _METRICS_PATH.exists():
            with open(_METRICS_PATH, "r", encoding="utf-8") as f:
                _metrics.update(json.load(f))
    except Exception:
        pass
    
    _initialized = True


def _save_goals() -> None:
    _ensure_dirs()
    try:
        with open(_GOALS_PATH, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in _goals.items()}, f, indent=2)
    except Exception:
        pass


def _save_learnings() -> None:
    _ensure_dirs()
    try:
        with open(_LEARNINGS_PATH, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in _learnings.items()}, f, indent=2)
    except Exception:
        pass


def _save_proposals() -> None:
    _ensure_dirs()
    try:
        with open(_PROPOSALS_PATH, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in _proposals.items()}, f, indent=2)
    except Exception:
        pass


def _save_metrics() -> None:
    _ensure_dirs()
    try:
        with open(_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(_metrics, f, indent=2)
    except Exception:
        pass


def _init() -> None:
    with _LOCK:
        _load_all()


# =============================================================================
# GOAL MANAGEMENT
# =============================================================================

def set_goal(
    description: str,
    success_criteria: str,
    priority: GoalPriority = GoalPriority.MEDIUM,
) -> str:
    """Set a new goal to achieve."""
    _init()
    
    goal_id = f"goal_{hashlib.md5(description.encode()).hexdigest()[:8]}"
    
    # Check if similar goal exists
    if goal_id in _goals:
        existing = _goals[goal_id]
        if existing.status == GoalStatus.ACHIEVED:
            return goal_id  # Already achieved
        # Reactivate if failed/paused
        existing.status = GoalStatus.ACTIVE
        existing.priority = priority
        with _LOCK:
            _save_goals()
        return goal_id
    
    goal = Goal(
        id=goal_id,
        description=description,
        success_criteria=success_criteria,
        priority=priority,
    )
    
    with _LOCK:
        _goals[goal_id] = goal
        _save_goals()
    
    return goal_id


def record_attempt(goal_id: str, success: bool, details: str = "") -> None:
    """Record an attempt to achieve a goal."""
    _init()
    
    goal = _goals.get(goal_id)
    if not goal:
        return
    
    with _LOCK:
        goal.attempts += 1
        goal.last_attempt = time.time()
        _metrics["total_attempts"] += 1
        
        if success:
            goal.successes += 1
            # Progress scales with success rate
            success_rate = goal.successes / max(1, goal.attempts)
            goal.progress = min(1.0, success_rate)
            _metrics["total_successes"] += 1
            
            # Check achievement criteria
            if _check_goal_achieved(goal):
                goal.status = GoalStatus.ACHIEVED
                goal.achieved_at = time.time()
                _metrics["goals_achieved"] = _metrics.get("goals_achieved", 0) + 1
            
            # Learn from success
            if details:
                goal.learned_strategies.append(details[:200])
                if len(goal.learned_strategies) > 10:
                    goal.learned_strategies = goal.learned_strategies[-10:]
        else:
            goal.failures += 1
            _metrics["total_failures"] += 1
            
            # Learn from failure
            if details:
                goal.failure_patterns.append(details[:200])
                if len(goal.failure_patterns) > 10:
                    goal.failure_patterns = goal.failure_patterns[-10:]
        
        _save_goals()
        _save_metrics()


def _check_goal_achieved(goal: Goal) -> bool:
    """Check if goal achievement criteria are met."""
    # Minimum attempts required
    if goal.attempts < 5:
        return False
    
    # Success rate threshold based on priority
    thresholds = {
        GoalPriority.CRITICAL: 0.9,
        GoalPriority.HIGH: 0.8,
        GoalPriority.MEDIUM: 0.7,
        GoalPriority.LOW: 0.6,
    }
    
    threshold = thresholds.get(goal.priority, 0.7)
    success_rate = goal.successes / max(1, goal.attempts)
    
    return success_rate >= threshold and goal.successes >= 5


def get_active_goals() -> List[Goal]:
    """Get all active goals sorted by priority."""
    _init()
    
    priority_order = {
        GoalPriority.CRITICAL: 0,
        GoalPriority.HIGH: 1,
        GoalPriority.MEDIUM: 2,
        GoalPriority.LOW: 3,
    }
    
    active = [g for g in _goals.values() if g.status == GoalStatus.ACTIVE]
    active.sort(key=lambda g: (priority_order.get(g.priority, 99), -g.failures))
    return active


def get_goal(goal_id: str) -> Optional[Goal]:
    """Get a specific goal."""
    _init()
    return _goals.get(goal_id)


# =============================================================================
# LEARNING SYSTEM
# =============================================================================

def learn(
    category: str,
    description: str,
    context: str,
    solution: Optional[str] = None,
    confidence: float = 0.5,
) -> str:
    """Record a learning insight."""
    _init()
    
    learning_id = f"learn_{hashlib.md5((category + description).encode()).hexdigest()[:8]}"
    
    # Update existing or create new
    existing = _learnings.get(learning_id)
    if existing:
        # Reinforce existing learning
        existing.confidence = min(1.0, existing.confidence + 0.1)
        existing.applications += 1
        if solution:
            existing.solution = solution
        with _LOCK:
            _save_learnings()
        return learning_id
    
    learning = Learning(
        id=learning_id,
        category=category,
        description=description,
        context=context,
        solution=solution,
        confidence=confidence,
    )
    
    with _LOCK:
        _learnings[learning_id] = learning
        _save_learnings()
    
    return learning_id


def apply_learning(learning_id: str, success: bool) -> None:
    """Record application of a learning."""
    _init()
    
    learning = _learnings.get(learning_id)
    if not learning:
        return
    
    with _LOCK:
        learning.applications += 1
        learning.last_applied = time.time()
        
        # Update success rate
        total = learning.applications
        current_successes = learning.success_rate * (total - 1)
        new_successes = current_successes + (1.0 if success else 0.0)
        learning.success_rate = new_successes / total
        
        # Adjust confidence based on success
        if success:
            learning.confidence = min(1.0, learning.confidence + 0.05)
        else:
            learning.confidence = max(0.1, learning.confidence - 0.1)
        
        _save_learnings()


def decay_learning_confidence() -> int:
    """Apply time-based confidence decay to learnings.
    
    Called periodically to prevent stale learnings from dominating.
    Returns number of learnings decayed.
    """
    _init()
    
    now = time.time()
    decay_threshold_sec = 7 * 24 * 3600  # 7 days
    decay_rate = 0.05
    decayed_count = 0
    
    with _LOCK:
        for learning in _learnings.values():
            if learning.last_applied:
                age_sec = now - learning.last_applied
            else:
                age_sec = now - learning.created_at
            
            # Decay if not applied recently
            if age_sec > decay_threshold_sec:
                old_conf = learning.confidence
                learning.confidence = max(0.1, learning.confidence - decay_rate)
                if learning.confidence < old_conf:
                    decayed_count += 1
        
        if decayed_count > 0:
            _save_learnings()
    
    return decayed_count


def reinforce_learning(learning_id: str, boost: float = 0.1) -> None:
    """Reinforce a learning that proved useful."""
    _init()
    
    learning = _learnings.get(learning_id)
    if not learning:
        return
    
    with _LOCK:
        learning.confidence = min(1.0, learning.confidence + boost)
        learning.last_applied = time.time()
        _save_learnings()


def get_high_confidence_learnings(min_confidence: float = 0.7) -> List[Learning]:
    """Get learnings with high confidence for reliable application."""
    _init()
    return [l for l in _learnings.values() if l.confidence >= min_confidence]


def prune_low_confidence_learnings(max_age_days: float = 30.0, min_confidence: float = 0.2) -> int:
    """Remove old learnings with very low confidence.
    
    Returns number of learnings removed.
    """
    _init()
    
    now = time.time()
    max_age_sec = max_age_days * 24 * 3600
    removed = 0
    
    with _LOCK:
        to_remove = []
        for learning_id, learning in _learnings.items():
            age_sec = now - learning.created_at
            if age_sec > max_age_sec and learning.confidence < min_confidence:
                to_remove.append(learning_id)
        
        for learning_id in to_remove:
            del _learnings[learning_id]
            removed += 1
        
        if removed > 0:
            _save_learnings()
    
    return removed


def get_relevant_learnings(category: str, context: str, limit: int = 5) -> List[Learning]:
    """Get learnings relevant to a context."""
    _init()
    
    relevant = []
    context_lower = context.lower()
    
    for learning in _learnings.values():
        if learning.category == category or learning.category == "general":
            # Simple relevance scoring
            score = learning.confidence * learning.success_rate
            if any(word in context_lower for word in learning.context.lower().split()[:5]):
                score += 0.3
            relevant.append((score, learning))
    
    relevant.sort(key=lambda x: x[0], reverse=True)
    return [l for _, l in relevant[:limit]]


# =============================================================================
# SELF-MODIFICATION PROPOSALS
# =============================================================================

def propose_modification(
    target_file: str,
    description: str,
    reason: str,
    old_code: str,
    new_code: str,
    risk_level: str = "medium",
) -> str:
    """Propose a code modification."""
    _init()
    
    proposal_id = f"prop_{int(time.time() * 1000)}_{len(_proposals)}"
    
    proposal = CodeProposal(
        id=proposal_id,
        target_file=target_file,
        description=description,
        reason=reason,
        old_code=old_code,
        new_code=new_code,
        risk_level=risk_level,
    )
    
    with _LOCK:
        _proposals[proposal_id] = proposal
        _save_proposals()
    
    return proposal_id


async def apply_proposal(proposal_id: str) -> Tuple[bool, str]:
    """Apply a code modification proposal."""
    _init()
    
    proposal = _proposals.get(proposal_id)
    if not proposal:
        return False, "Proposal not found"
    
    if proposal.status != "pending":
        return False, f"Proposal status is {proposal.status}"
    
    try:
        target_path = Path(proposal.target_file)
        
        if not target_path.exists():
            proposal.status = "failed"
            proposal.result = "Target file not found"
            with _LOCK:
                _save_proposals()
            return False, "Target file not found"
        
        # Read current content
        content = target_path.read_text(encoding="utf-8")
        
        # Check if old_code exists
        if proposal.old_code and proposal.old_code not in content:
            proposal.status = "failed"
            proposal.result = "Old code not found in file"
            with _LOCK:
                _save_proposals()
            return False, "Old code not found in file"
        
        # Apply modification
        if proposal.old_code:
            new_content = content.replace(proposal.old_code, proposal.new_code, 1)
        else:
            # Append if no old_code specified
            new_content = content + "\n" + proposal.new_code
        
        # Write back
        target_path.write_text(new_content, encoding="utf-8")
        
        proposal.status = "applied"
        proposal.applied_at = time.time()
        proposal.result = "Successfully applied"
        
        with _LOCK:
            _metrics["self_modifications"] += 1
            _save_proposals()
            _save_metrics()
        
        return True, "Applied successfully"
        
    except Exception as e:
        proposal.status = "failed"
        proposal.result = str(e)
        with _LOCK:
            _save_proposals()
        return False, str(e)


def get_pending_proposals() -> List[CodeProposal]:
    """Get all pending proposals."""
    _init()
    return [p for p in _proposals.values() if p.status == "pending"]


# =============================================================================
# LLM-POWERED ANALYSIS (Rate Limited)
# =============================================================================

async def analyze_failures_with_llm(force: bool = False) -> Optional[Dict[str, Any]]:
    """Use LLM to analyze failures and propose improvements.
    
    Rate limited to avoid excessive API calls.
    """
    global _LAST_LLM_ANALYSIS
    _init()
    
    now = time.time()
    if not force and (now - _LAST_LLM_ANALYSIS) < _LLM_COOLDOWN_SEC:
        return None  # Cooldown not elapsed
    
    # Collect failure data
    failure_data = []
    for goal in _goals.values():
        if goal.failures > 0 and goal.failure_patterns:
            failure_data.append({
                "goal": goal.description,
                "failures": goal.failures,
                "patterns": goal.failure_patterns[-5:],
            })
    
    if not failure_data:
        return None  # No failures to analyze
    
    try:
        from jinx.micro.llm.service import spark_openai as _spark
        
        prompt = f"""Analyze these system failures and propose improvements:

{json.dumps(failure_data, indent=2)}

Provide a JSON response with:
{{
  "root_causes": ["cause1", "cause2"],
  "recommendations": ["rec1", "rec2"],
  "code_fixes": [
    {{"file": "path", "description": "what to fix", "priority": "high/medium/low"}}
  ]
}}

Be concise. Focus on actionable fixes."""

        response, _ = await asyncio.wait_for(_spark(prompt), timeout=30.0)
        
        _LAST_LLM_ANALYSIS = now
        with _LOCK:
            _metrics["llm_calls"] += 1
            _save_metrics()
        
        # Parse response
        try:
            import re
            match = re.search(r"\{[\s\S]*\}", response or "")
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        
        return {"raw_response": response}
        
    except Exception as e:
        return {"error": str(e)}


async def propose_architecture_change(problem: str, context: str) -> Optional[str]:
    """Use LLM to propose an architecture change when stuck.
    
    Only called when multiple attempts have failed.
    """
    global _LAST_LLM_ANALYSIS
    _init()
    
    now = time.time()
    if (now - _LAST_LLM_ANALYSIS) < _LLM_COOLDOWN_SEC:
        return None
    
    try:
        from jinx.micro.llm.service import spark_openai as _spark
        
        prompt = f"""System is stuck on this problem:
{problem}

Context:
{context}

Propose a minimal code change to fix this. Response format:
FILE: <path>
OLD:
```
<old code>
```
NEW:
```
<new code>
```
REASON: <why this fixes it>

Be minimal and precise."""

        response, _ = await asyncio.wait_for(_spark(prompt), timeout=30.0)
        
        _LAST_LLM_ANALYSIS = now
        with _LOCK:
            _metrics["llm_calls"] += 1
            _save_metrics()
        
        return response
        
    except Exception:
        return None


# =============================================================================
# EVOLUTION CYCLE
# =============================================================================

async def run_evolution_cycle() -> Dict[str, Any]:
    """Run one evolution cycle - analyze, learn, improve.
    
    This is the main self-improvement loop. Called periodically.
    """
    _init()
    
    results = {
        "analyzed_goals": 0,
        "learnings_created": 0,
        "proposals_created": 0,
        "llm_called": False,
    }
    
    try:
        # 1. Check active goals
        active_goals = get_active_goals()
        results["analyzed_goals"] = len(active_goals)
        
        # 2. Find struggling goals (high failure rate)
        struggling = [g for g in active_goals if g.failures > 3 and g.successes < g.failures]
        
        # 3. If struggling, try to learn from patterns
        for goal in struggling:
            # Create learning from failure patterns
            if goal.failure_patterns:
                pattern = goal.failure_patterns[-1]
                learn_id = learn(
                    category="error_pattern",
                    description=f"Failure in: {goal.description[:50]}",
                    context=pattern,
                    confidence=0.3,
                )
                results["learnings_created"] += 1
        
        # 4. If still struggling after many attempts, request LLM analysis
        very_stuck = [g for g in struggling if g.failures > 10]
        if very_stuck and not results["llm_called"]:
            analysis = await analyze_failures_with_llm()
            if analysis:
                results["llm_called"] = True
                
                # Create proposals from analysis
                code_fixes = analysis.get("code_fixes", [])
                for fix in code_fixes[:3]:  # Max 3 proposals
                    if fix.get("file") and fix.get("description"):
                        # We don't have the actual code yet, mark for manual review
                        learn(
                            category="architecture",
                            description=fix.get("description", ""),
                            context=f"File: {fix.get('file')}",
                            solution="Needs manual implementation",
                            confidence=0.4,
                        )
                        results["proposals_created"] += 1
        
        # 5. Update metrics
        with _LOCK:
            _metrics["evolution_cycles"] += 1
            _save_metrics()
        
    except Exception as e:
        results["error"] = str(e)
    
    return results


# =============================================================================
# SYSTEM GOAL: SELF-IMPROVEMENT
# =============================================================================

_SYSTEM_GOALS = [
    {
        "description": "Achieve user tasks successfully",
        "success_criteria": "Success rate > 80%",
        "priority": "critical",
    },
    {
        "description": "Learn from failures and improve",
        "success_criteria": "Failure rate decreasing over time",
        "priority": "high",
    },
    {
        "description": "Optimize response latency",
        "success_criteria": "Average response time < 10s",
        "priority": "medium",
    },
    {
        "description": "Minimize errors and exceptions",
        "success_criteria": "Error rate < 5%",
        "priority": "high",
    },
]


def initialize_system_goals() -> None:
    """Initialize default system goals."""
    _init()
    
    for goal_def in _SYSTEM_GOALS:
        set_goal(
            description=goal_def["description"],
            success_criteria=goal_def["success_criteria"],
            priority=GoalPriority(goal_def["priority"]),
        )


# =============================================================================
# SUMMARY & DIAGNOSTICS
# =============================================================================

def get_evolution_summary() -> Dict[str, Any]:
    """Get summary of evolution state."""
    _init()
    
    active_goals = [g for g in _goals.values() if g.status == GoalStatus.ACTIVE]
    achieved_goals = [g for g in _goals.values() if g.status == GoalStatus.ACHIEVED]
    
    return {
        "goals": {
            "active": len(active_goals),
            "achieved": len(achieved_goals),
            "total": len(_goals),
        },
        "learnings": {
            "total": len(_learnings),
            "high_confidence": len([l for l in _learnings.values() if l.confidence > 0.7]),
        },
        "proposals": {
            "pending": len([p for p in _proposals.values() if p.status == "pending"]),
            "applied": len([p for p in _proposals.values() if p.status == "applied"]),
        },
        "metrics": _metrics,
        "llm_cooldown_remaining": max(0, _LLM_COOLDOWN_SEC - (time.time() - _LAST_LLM_ANALYSIS)),
    }


def build_evolution_context() -> str:
    """Build context block for LLM prompts."""
    _init()
    
    lines = ["<evolution_state>"]
    
    # Active goals
    active = get_active_goals()
    if active:
        lines.append(f"Active Goals: {len(active)}")
        for g in active[:3]:
            status = f"[{g.successes}/{g.attempts}]" if g.attempts > 0 else ""
            lines.append(f"  - {g.description[:50]} {status}")
    
    # Recent learnings
    recent = sorted(_learnings.values(), key=lambda l: l.created_at, reverse=True)[:3]
    if recent:
        lines.append("Recent Learnings:")
        for l in recent:
            lines.append(f"  - {l.description[:50]} (conf: {l.confidence:.0%})")
    
    lines.append("</evolution_state>")
    return "\n".join(lines)


def suggest_goals_from_patterns() -> List[Dict[str, str]]:
    """Suggest new goals based on observed patterns."""
    _init()
    
    suggestions = []
    
    # Analyze failure patterns
    failure_types: Dict[str, int] = {}
    for goal in _goals.values():
        for pattern in goal.failure_patterns:
            # Extract error type
            if ":" in pattern:
                err_type = pattern.split(":")[0].strip()
                failure_types[err_type] = failure_types.get(err_type, 0) + 1
    
    # Suggest goals for recurring failures
    for err_type, count in failure_types.items():
        if count >= 3:
            suggestions.append({
                "description": f"Reduce {err_type} errors",
                "priority": "high",
                "reason": f"Observed {count} occurrences",
            })
    
    # Suggest based on success patterns
    successful_categories: Dict[str, int] = {}
    for learning in _learnings.values():
        if learning.category == "success_strategy" and learning.confidence > 0.6:
            successful_categories[learning.category] = successful_categories.get(learning.category, 0) + 1
    
    # Analyze task completion patterns
    completed_tasks = sum(1 for g in _goals.values() if g.status == GoalStatus.ACHIEVED)
    active_tasks = sum(1 for g in _goals.values() if g.status == GoalStatus.ACTIVE)
    
    if active_tasks > 10 and completed_tasks < 3:
        suggestions.append({
            "description": "Focus on completing existing goals",
            "priority": "high",
            "reason": f"{active_tasks} active but only {completed_tasks} completed",
        })
    
    # Suggest optimization if latency is high
    try:
        from jinx.micro.runtime.brain_metrics import get_current_metrics
        metrics = get_current_metrics()
        if metrics.avg_task_latency_ms > 20000:
            suggestions.append({
                "description": "Optimize task execution speed",
                "priority": "medium",
                "reason": f"Avg latency is {metrics.avg_task_latency_ms:.0f}ms",
            })
    except Exception:
        pass
    
    return suggestions[:5]  # Max 5 suggestions


def auto_create_suggested_goals() -> List[str]:
    """Automatically create goals from suggestions."""
    _init()
    
    suggestions = suggest_goals_from_patterns()
    created_ids = []
    
    for suggestion in suggestions:
        # Check if similar goal already exists
        exists = any(
            suggestion["description"].lower() in g.description.lower()
            for g in _goals.values()
            if g.status == GoalStatus.ACTIVE
        )
        
        if not exists:
            goal_id = set_goal(
                description=suggestion["description"],
                success_criteria=suggestion.get("reason", "Improve system performance"),
                priority=GoalPriority(suggestion.get("priority", "medium")),
            )
            created_ids.append(goal_id)
    
    return created_ids


# =============================================================================
# USER GOAL DETECTION
# =============================================================================

_GOAL_PATTERNS = {
    # Code patterns
    "create": ("Create/implement", "high"),
    "implement": ("Implement feature", "high"),
    "build": ("Build system", "high"),
    "add": ("Add functionality", "medium"),
    "fix": ("Fix issue", "high"),
    "debug": ("Debug problem", "medium"),
    "refactor": ("Refactor code", "medium"),
    "optimize": ("Optimize performance", "medium"),
    "remove": ("Remove/clean up", "low"),
    "delete": ("Delete/remove", "low"),
    # Analysis patterns
    "analyze": ("Analyze/understand", "medium"),
    "explain": ("Explain/document", "low"),
    "find": ("Find/locate", "medium"),
    "search": ("Search codebase", "low"),
    # Integration patterns
    "integrate": ("Integrate systems", "high"),
    "connect": ("Connect components", "medium"),
    "migrate": ("Migrate/upgrade", "high"),
    # Architecture patterns
    "design": ("Design architecture", "critical"),
    "architect": ("Architect system", "critical"),
    "restructure": ("Restructure codebase", "high"),
}


def detect_user_goal(task_description: str) -> Optional[str]:
    """Detect and create a goal from user's task description.
    
    Returns goal_id if a goal was created, None otherwise.
    """
    _init()
    
    if not task_description or len(task_description) < 5:
        return None
    
    task_lower = task_description.lower()
    
    # Find matching pattern
    detected_priority = "medium"
    detected_type = None
    
    for keyword, (goal_type, priority) in _GOAL_PATTERNS.items():
        if keyword in task_lower:
            detected_type = goal_type
            detected_priority = priority
            break
    
    if not detected_type:
        detected_type = "Complete task"
    
    # Create goal
    description = f"{detected_type}: {task_description[:80]}"
    success_criteria = f"Task completed successfully"
    
    goal_id = set_goal(
        description=description,
        success_criteria=success_criteria,
        priority=GoalPriority(detected_priority),
    )
    
    return goal_id


def track_user_request(request_text: str) -> Tuple[str, Optional[str]]:
    """Track a user request and create appropriate goal.
    
    Returns (request_hash, goal_id or None).
    """
    _init()
    
    if not request_text:
        return "", None
    
    # Hash to avoid duplicates
    req_hash = hashlib.md5(request_text.encode()).hexdigest()[:12]
    
    # Detect and create goal
    goal_id = detect_user_goal(request_text)
    
    # Update context
    try:
        from jinx.micro.runtime.arch_memory import update_context
        update_context(add_query=request_text[:200])
    except Exception:
        pass
    
    return req_hash, goal_id


def complete_user_request(request_hash: str, goal_id: Optional[str], success: bool, details: str = "") -> None:
    """Mark a user request as completed."""
    _init()
    
    if goal_id:
        record_attempt(goal_id, success, details)
        
        if success:
            # Learn from successful completion
            learn(
                category="success_strategy",
                description=f"Completed user request",
                context=details[:200],
                confidence=0.6,
            )
        else:
            # Learn from failure
            learn(
                category="error_pattern",
                description=f"Failed user request",
                context=details[:200],
                confidence=0.4,
            )


__all__ = [
    # Goals
    "set_goal",
    "record_attempt",
    "get_active_goals",
    "get_goal",
    "Goal",
    "GoalStatus",
    "GoalPriority",
    # Learning
    "learn",
    "apply_learning",
    "get_relevant_learnings",
    "decay_learning_confidence",
    "reinforce_learning",
    "get_high_confidence_learnings",
    "prune_low_confidence_learnings",
    "Learning",
    # Proposals
    "propose_modification",
    "apply_proposal",
    "get_pending_proposals",
    "CodeProposal",
    # LLM Analysis
    "analyze_failures_with_llm",
    "propose_architecture_change",
    # Evolution
    "run_evolution_cycle",
    "initialize_system_goals",
    "suggest_goals_from_patterns",
    "auto_create_suggested_goals",
    # User Goal Detection
    "detect_user_goal",
    "track_user_request",
    "complete_user_request",
    # Summary
    "get_evolution_summary",
    "build_evolution_context",
]
