"""Architectural Memory - Persistent task tracking and context system.

Provides Jinx with memory of what it's doing across sessions:
- Task Graph: tracks current/completed/pending tasks with dependencies
- Execution Context: remembers the state of ongoing work
- Decision History: records why decisions were made for learning
- Project Map: understands project structure and relationships

This enables:
- Resuming work after interruption
- Intelligent task decomposition and sequencing
- Learning from past execution patterns
- Scaling across multiple parallel tasks
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from threading import Lock
from enum import Enum

_MEMORY_DIR = Path(".jinx") / "arch_memory"
_TASK_GRAPH_PATH = _MEMORY_DIR / "task_graph.json"
_CONTEXT_PATH = _MEMORY_DIR / "execution_context.json"
_DECISIONS_PATH = _MEMORY_DIR / "decisions.json"
_PROJECT_MAP_PATH = _MEMORY_DIR / "project_map.json"

_LOCK = Lock()


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """A single task in the execution graph."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        d = dict(d)
        d["status"] = TaskStatus(d.get("status", "pending"))
        return cls(**d)


@dataclass
class ExecutionContext:
    """Current execution state for resumability."""
    active_task_id: Optional[str] = None
    active_group: str = "main"
    working_files: List[str] = field(default_factory=list)
    working_symbols: List[str] = field(default_factory=list)
    recent_queries: List[str] = field(default_factory=list)
    intent_stack: List[str] = field(default_factory=list)
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionContext":
        return cls(**d)


@dataclass  
class Decision:
    """A recorded decision for learning and auditing."""
    id: str
    task_id: Optional[str]
    decision_type: str  # e.g., "strategy_selection", "file_choice", "parameter_tuning"
    options: List[str]
    chosen: str
    reasoning: str
    outcome: Optional[str] = None
    success: Optional[bool] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Decision":
        return cls(**d)


@dataclass
class ProjectNode:
    """A node in the project structure map."""
    path: str
    node_type: str  # "file", "directory", "module", "class", "function"
    name: str
    importance: float = 0.5  # 0-1 scale
    last_modified: Optional[float] = None
    last_accessed: Optional[float] = None
    related_tasks: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectNode":
        return cls(**d)


# =============================================================================
# GLOBAL STATE
# =============================================================================

_task_graph: Dict[str, Task] = {}
_context: ExecutionContext = ExecutionContext()
_decisions: List[Decision] = []
_project_map: Dict[str, ProjectNode] = {}
_initialized = False


def _ensure_dirs() -> None:
    try:
        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _load_all() -> None:
    global _task_graph, _context, _decisions, _project_map, _initialized
    if _initialized:
        return
    
    _ensure_dirs()
    
    # Load task graph
    try:
        if _TASK_GRAPH_PATH.exists():
            with open(_TASK_GRAPH_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _task_graph = {k: Task.from_dict(v) for k, v in data.items()}
    except Exception:
        _task_graph = {}
    
    # Load execution context
    try:
        if _CONTEXT_PATH.exists():
            with open(_CONTEXT_PATH, "r", encoding="utf-8") as f:
                _context = ExecutionContext.from_dict(json.load(f))
    except Exception:
        _context = ExecutionContext()
    
    # Load decisions (last 100)
    try:
        if _DECISIONS_PATH.exists():
            with open(_DECISIONS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _decisions = [Decision.from_dict(d) for d in data[-100:]]
    except Exception:
        _decisions = []
    
    # Load project map
    try:
        if _PROJECT_MAP_PATH.exists():
            with open(_PROJECT_MAP_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _project_map = {k: ProjectNode.from_dict(v) for k, v in data.items()}
    except Exception:
        _project_map = {}
    
    _initialized = True


def _save_task_graph() -> None:
    _ensure_dirs()
    try:
        with open(_TASK_GRAPH_PATH, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in _task_graph.items()}, f, indent=2)
    except Exception:
        pass


def _save_context() -> None:
    _ensure_dirs()
    try:
        with open(_CONTEXT_PATH, "w", encoding="utf-8") as f:
            json.dump(_context.to_dict(), f, indent=2)
    except Exception:
        pass


def _save_decisions() -> None:
    _ensure_dirs()
    try:
        with open(_DECISIONS_PATH, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in _decisions[-100:]], f, indent=2)
    except Exception:
        pass


def _save_project_map() -> None:
    _ensure_dirs()
    try:
        with open(_PROJECT_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in _project_map.items()}, f, indent=2)
    except Exception:
        pass


def _init() -> None:
    with _LOCK:
        _load_all()


# =============================================================================
# TASK GRAPH API
# =============================================================================

def create_task(
    description: str,
    parent_id: Optional[str] = None,
    depends_on: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a new task and return its ID."""
    _init()
    
    task_id = f"task_{int(time.time() * 1000)}_{len(_task_graph)}"
    
    task = Task(
        id=task_id,
        description=description,
        parent_id=parent_id,
        depends_on=depends_on or [],
        metadata=metadata or {},
    )
    
    # Check if blocked by dependencies
    if depends_on:
        for dep_id in depends_on:
            dep = _task_graph.get(dep_id)
            if dep and dep.status != TaskStatus.COMPLETED:
                task.status = TaskStatus.BLOCKED
                break
    
    # Add to parent's children
    if parent_id and parent_id in _task_graph:
        _task_graph[parent_id].children.append(task_id)
    
    with _LOCK:
        _task_graph[task_id] = task
        _save_task_graph()
    
    return task_id


def start_task(task_id: str) -> bool:
    """Mark a task as in progress."""
    _init()
    
    task = _task_graph.get(task_id)
    if not task:
        return False
    
    # Check dependencies
    for dep_id in task.depends_on:
        dep = _task_graph.get(dep_id)
        if dep and dep.status != TaskStatus.COMPLETED:
            return False  # Still blocked
    
    with _LOCK:
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()
        _context.active_task_id = task_id
        _context.last_updated = time.time()
        _save_task_graph()
        _save_context()
    
    return True


def complete_task(task_id: str, result: Optional[str] = None) -> bool:
    """Mark a task as completed."""
    _init()
    
    task = _task_graph.get(task_id)
    if not task:
        return False
    
    with _LOCK:
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.result = result
        
        # Unblock dependent tasks
        for other_id, other in _task_graph.items():
            if task_id in other.depends_on and other.status == TaskStatus.BLOCKED:
                # Check if all dependencies are now complete
                all_complete = all(
                    _task_graph.get(d, Task(id="", description="")).status == TaskStatus.COMPLETED
                    for d in other.depends_on
                )
                if all_complete:
                    other.status = TaskStatus.PENDING
        
        if _context.active_task_id == task_id:
            _context.active_task_id = None
        _context.last_updated = time.time()
        
        _save_task_graph()
        _save_context()
    
    return True


def fail_task(task_id: str, error: str) -> bool:
    """Mark a task as failed."""
    _init()
    
    task = _task_graph.get(task_id)
    if not task:
        return False
    
    with _LOCK:
        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        task.error = error
        
        if _context.active_task_id == task_id:
            _context.active_task_id = None
        _context.last_updated = time.time()
        
        _save_task_graph()
        _save_context()
    
    return True


def get_task(task_id: str) -> Optional[Task]:
    """Get a task by ID."""
    _init()
    return _task_graph.get(task_id)


def get_active_tasks() -> List[Task]:
    """Get all in-progress tasks."""
    _init()
    return [t for t in _task_graph.values() if t.status == TaskStatus.IN_PROGRESS]


def get_pending_tasks() -> List[Task]:
    """Get all pending (ready to execute) tasks."""
    _init()
    return [t for t in _task_graph.values() if t.status == TaskStatus.PENDING]


def get_blocked_tasks() -> List[Task]:
    """Get all blocked tasks."""
    _init()
    return [t for t in _task_graph.values() if t.status == TaskStatus.BLOCKED]


def get_recent_tasks(limit: int = 10) -> List[Task]:
    """Get most recent tasks by creation time."""
    _init()
    tasks = sorted(_task_graph.values(), key=lambda t: t.created_at, reverse=True)
    return tasks[:limit]


def get_task_tree(root_id: Optional[str] = None) -> Dict[str, Any]:
    """Get task tree starting from root (or all root tasks if None)."""
    _init()
    
    def build_tree(task_id: str) -> Dict[str, Any]:
        task = _task_graph.get(task_id)
        if not task:
            return {}
        return {
            "id": task.id,
            "description": task.description,
            "status": task.status.value,
            "children": [build_tree(c) for c in task.children],
        }
    
    if root_id:
        return build_tree(root_id)
    
    # Find root tasks (no parent)
    roots = [t for t in _task_graph.values() if not t.parent_id]
    return {"roots": [build_tree(r.id) for r in roots]}


# =============================================================================
# EXECUTION CONTEXT API
# =============================================================================

def get_context() -> ExecutionContext:
    """Get current execution context."""
    _init()
    return _context


def update_context(
    working_files: Optional[List[str]] = None,
    working_symbols: Optional[List[str]] = None,
    add_query: Optional[str] = None,
    push_intent: Optional[str] = None,
    pop_intent: bool = False,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> None:
    """Update execution context."""
    _init()
    
    with _LOCK:
        if working_files is not None:
            _context.working_files = working_files[:20]  # Cap at 20
        if working_symbols is not None:
            _context.working_symbols = working_symbols[:50]  # Cap at 50
        if add_query:
            _context.recent_queries = ([add_query] + _context.recent_queries)[:10]
        if push_intent:
            _context.intent_stack.append(push_intent)
            if len(_context.intent_stack) > 10:
                _context.intent_stack = _context.intent_stack[-10:]
        if pop_intent and _context.intent_stack:
            _context.intent_stack.pop()
        if checkpoint:
            _context.checkpoint_data.update(checkpoint)
        
        _context.last_updated = time.time()
        _save_context()


def clear_context() -> None:
    """Clear execution context (fresh start)."""
    global _context
    _init()
    
    with _LOCK:
        _context = ExecutionContext()
        _save_context()


# =============================================================================
# DECISION HISTORY API  
# =============================================================================

def record_decision(
    decision_type: str,
    options: List[str],
    chosen: str,
    reasoning: str,
    task_id: Optional[str] = None,
) -> str:
    """Record a decision for learning and auditing."""
    _init()
    
    decision_id = f"dec_{int(time.time() * 1000)}_{len(_decisions)}"
    
    decision = Decision(
        id=decision_id,
        task_id=task_id or _context.active_task_id,
        decision_type=decision_type,
        options=options,
        chosen=chosen,
        reasoning=reasoning,
    )
    
    with _LOCK:
        _decisions.append(decision)
        if len(_decisions) > 100:
            _decisions[:] = _decisions[-100:]
        _save_decisions()
    
    return decision_id


def update_decision_outcome(decision_id: str, outcome: str, success: bool) -> bool:
    """Update a decision with its outcome."""
    _init()
    
    with _LOCK:
        for d in _decisions:
            if d.id == decision_id:
                d.outcome = outcome
                d.success = success
                _save_decisions()
                return True
    return False


def get_decisions_for_type(decision_type: str, limit: int = 10) -> List[Decision]:
    """Get recent decisions of a specific type."""
    _init()
    matches = [d for d in reversed(_decisions) if d.decision_type == decision_type]
    return matches[:limit]


def get_successful_patterns(decision_type: str) -> List[Decision]:
    """Get successful decisions of a type for learning."""
    _init()
    return [d for d in _decisions if d.decision_type == decision_type and d.success]


# =============================================================================
# PROJECT MAP API
# =============================================================================

def update_project_node(
    path: str,
    node_type: str,
    name: str,
    importance: Optional[float] = None,
    symbols: Optional[List[str]] = None,
    task_id: Optional[str] = None,
) -> None:
    """Update or create a project node."""
    _init()
    
    with _LOCK:
        existing = _project_map.get(path)
        if existing:
            if importance is not None:
                existing.importance = importance
            if symbols is not None:
                existing.symbols = symbols
            if task_id and task_id not in existing.related_tasks:
                existing.related_tasks.append(task_id)
            existing.last_accessed = time.time()
        else:
            _project_map[path] = ProjectNode(
                path=path,
                node_type=node_type,
                name=name,
                importance=importance or 0.5,
                symbols=symbols or [],
                related_tasks=[task_id] if task_id else [],
                last_accessed=time.time(),
            )
        _save_project_map()


def get_important_files(limit: int = 10) -> List[ProjectNode]:
    """Get most important files by score."""
    _init()
    files = [n for n in _project_map.values() if n.node_type == "file"]
    files.sort(key=lambda n: n.importance, reverse=True)
    return files[:limit]


def get_recently_accessed(limit: int = 10) -> List[ProjectNode]:
    """Get recently accessed project nodes."""
    _init()
    nodes = sorted(
        [n for n in _project_map.values() if n.last_accessed],
        key=lambda n: n.last_accessed or 0,
        reverse=True,
    )
    return nodes[:limit]


def find_related_files(task_id: str) -> List[ProjectNode]:
    """Find files related to a task."""
    _init()
    return [n for n in _project_map.values() if task_id in n.related_tasks]


# =============================================================================
# SUMMARY & DIAGNOSTICS
# =============================================================================

def get_memory_summary() -> Dict[str, Any]:
    """Get a summary of architectural memory state."""
    _init()
    
    task_counts = {
        "total": len(_task_graph),
        "pending": len([t for t in _task_graph.values() if t.status == TaskStatus.PENDING]),
        "in_progress": len([t for t in _task_graph.values() if t.status == TaskStatus.IN_PROGRESS]),
        "completed": len([t for t in _task_graph.values() if t.status == TaskStatus.COMPLETED]),
        "failed": len([t for t in _task_graph.values() if t.status == TaskStatus.FAILED]),
        "blocked": len([t for t in _task_graph.values() if t.status == TaskStatus.BLOCKED]),
    }
    
    return {
        "tasks": task_counts,
        "active_task": _context.active_task_id,
        "working_files": len(_context.working_files),
        "intent_depth": len(_context.intent_stack),
        "decisions_recorded": len(_decisions),
        "project_nodes": len(_project_map),
        "last_updated": _context.last_updated,
    }


def build_context_block() -> str:
    """Build a context block for LLM prompts with current state."""
    _init()
    
    lines = ["<arch_memory>"]
    
    # Active task
    if _context.active_task_id:
        task = _task_graph.get(_context.active_task_id)
        if task:
            lines.append(f"Active Task: {task.description}")
    
    # Intent stack
    if _context.intent_stack:
        lines.append(f"Intent: {' → '.join(_context.intent_stack[-3:])}")
    
    # Working files
    if _context.working_files:
        lines.append(f"Working on: {', '.join(_context.working_files[:5])}")
    
    # Pending tasks
    pending = get_pending_tasks()
    if pending:
        lines.append(f"Pending tasks: {len(pending)}")
        for t in pending[:3]:
            lines.append(f"  - {t.description[:60]}")
    
    lines.append("</arch_memory>")
    return "\n".join(lines)


async def cleanup_old_tasks(max_age_hours: float = 24.0) -> int:
    """Remove completed/failed tasks older than max_age_hours."""
    _init()
    
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    
    with _LOCK:
        to_remove = []
        for task_id, task in _task_graph.items():
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                if task.completed_at and task.completed_at < cutoff:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del _task_graph[task_id]
            removed += 1
        
        if removed > 0:
            _save_task_graph()
    
    return removed


__all__ = [
    # Task Graph
    "create_task",
    "start_task",
    "complete_task",
    "fail_task",
    "get_task",
    "get_active_tasks",
    "get_pending_tasks",
    "get_blocked_tasks",
    "get_recent_tasks",
    "get_task_tree",
    "TaskStatus",
    "Task",
    # Execution Context
    "get_context",
    "update_context",
    "clear_context",
    "ExecutionContext",
    # Decisions
    "record_decision",
    "update_decision_outcome",
    "get_decisions_for_type",
    "get_successful_patterns",
    "Decision",
    # Project Map
    "update_project_node",
    "get_important_files",
    "get_recently_accessed",
    "find_related_files",
    "ProjectNode",
    # Summary
    "get_memory_summary",
    "build_context_block",
    "cleanup_old_tasks",
]
