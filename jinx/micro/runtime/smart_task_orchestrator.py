"""Smart Task Orchestrator - Intelligent task batching, splitting, and restructuring.

Jinx can act as a "constructor" that:
- Combines multiple related tasks into a single API request (batching)
- Splits complex tasks into smaller, manageable sub-tasks
- Restructures poorly formulated tasks on the fly
- Optimizes API usage based on task relationships and complexity
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Callable

_ORCHESTRATOR_DIR = Path(".jinx") / "orchestrator"
_TASK_HISTORY_FILE = _ORCHESTRATOR_DIR / "task_history.json"

_LOCK = Lock()


class TaskAction(str, Enum):
    EXECUTE_SINGLE = "execute_single"      # Execute as-is
    BATCH_COMBINE = "batch_combine"        # Combine with other tasks
    SPLIT_SUBTASKS = "split_subtasks"      # Split into smaller tasks
    RESTRUCTURE = "restructure"            # Rewrite/improve the task
    DEFER = "defer"                        # Defer for later
    REJECT = "reject"                      # Task is invalid/impossible


class TaskComplexity(str, Enum):
    TRIVIAL = "trivial"      # Simple question/action
    SIMPLE = "simple"        # Single-step task
    MODERATE = "moderate"    # Multi-step but manageable
    COMPLEX = "complex"      # Requires decomposition
    MASSIVE = "massive"      # Should definitely be split


@dataclass
class TaskAnalysis:
    """Analysis result for a task."""
    task_id: str
    original_text: str
    complexity: TaskComplexity
    recommended_action: TaskAction
    
    # For batching
    can_batch: bool = False
    batch_key: str = ""  # Tasks with same key can be batched
    
    # For splitting
    suggested_subtasks: List[str] = field(default_factory=list)
    
    # For restructuring
    restructured_text: Optional[str] = None
    restructure_reason: str = ""
    
    # Metadata
    estimated_api_calls: int = 1
    priority_score: float = 0.5
    keywords: List[str] = field(default_factory=list)
    related_task_ids: List[str] = field(default_factory=list)


@dataclass
class TaskBatch:
    """A batch of related tasks to execute together."""
    batch_id: str
    tasks: List[TaskAnalysis]
    combined_prompt: str
    estimated_savings: float  # API calls saved by batching
    created_at: float = field(default_factory=time.time)


# Task patterns for analysis
_COMPLEXITY_PATTERNS = {
    TaskComplexity.TRIVIAL: [
        r"^(what|who|when|where|how much|how many)\s",
        r"^(hi|hello|hey|thanks|ok|yes|no)\b",
        r"^(define|explain briefly)\s",
    ],
    TaskComplexity.SIMPLE: [
        r"^(show|list|find|get|read|print)\s",
        r"^(check|verify|validate)\s",
        r"^(create|make|add)\s+(a|an|one)\s",
    ],
    TaskComplexity.MODERATE: [
        r"^(implement|write|build|develop)\s",
        r"^(fix|debug|repair|solve)\s",
        r"^(refactor|improve|optimize)\s",
        r"\band\b.*\band\b",  # Multiple "and" conjunctions
    ],
    TaskComplexity.COMPLEX: [
        r"^(design|architect|plan)\s",
        r"(system|framework|architecture)",
        r"(multiple|several|many)\s+(files|components|modules)",
        r"\d+\s+(tasks|steps|things|items)",
    ],
    TaskComplexity.MASSIVE: [
        r"(entire|whole|complete|full)\s+(project|system|application)",
        r"(rewrite|rebuild|recreate)\s+(everything|all)",
        r"(\d{2,})\s+(files|components)",  # 10+ files/components
    ],
}

# Patterns for task batching (tasks with similar patterns can be batched)
_BATCH_PATTERNS = {
    "file_operations": [r"(create|delete|rename|move|copy)\s+file", r"(read|write)\s+to\s+file"],
    "code_review": [r"(review|check|analyze)\s+(code|function|class)", r"look\s+at\s+(this|the)\s+code"],
    "documentation": [r"(document|add\s+docs|write\s+comments)", r"(explain|describe)\s+(function|code)"],
    "testing": [r"(test|write\s+test|add\s+test)", r"(verify|validate)\s+(function|code)"],
    "refactoring": [r"(refactor|clean\s+up|improve)", r"(simplify|optimize)\s+(code|function)"],
}

# Patterns that indicate task should be split
_SPLIT_INDICATORS = [
    (r"(\d+)\s+things", lambda m: int(m.group(1)) > 3),
    (r"(\d+)\s+tasks", lambda m: int(m.group(1)) > 2),
    (r"(\d+)\s+files", lambda m: int(m.group(1)) > 5),
    (r"(first|then|after that|next|finally)", lambda m: True),  # Sequential steps mentioned
    (r"(and|also|plus|additionally)\s+\w+\s+(and|also|plus)", lambda m: True),  # Multiple conjunctions
]

# Patterns for restructuring poorly formulated tasks
_RESTRUCTURE_PATTERNS = [
    # Vague requests
    (r"^(do|make|fix)\s+(it|this|that|something)$", "Task is too vague - needs specific target"),
    (r"^(help|please|pls)\s*$", "Task has no content"),
    # Missing context
    (r"(the|this|that)\s+(error|bug|issue|problem)$", "Task references unknown error without details"),
    # Ambiguous scope
    (r"^(everything|all|anything)", "Task scope is too broad"),
    # Incomplete sentences
    (r"^[a-z]", "Task should start with capital letter or be a complete sentence"),
]


def _ensure_dirs() -> None:
    try:
        _ORCHESTRATOR_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _generate_task_id(text: str) -> str:
    return f"task_{hashlib.md5(f'{text}:{time.time()}'.encode()).hexdigest()[:12]}"


def _generate_batch_key(text: str) -> str:
    """Generate a key for grouping batchable tasks."""
    text_lower = text.lower()
    for key, patterns in _BATCH_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return key
    return ""


def _extract_keywords(text: str) -> List[str]:
    """Extract important keywords from task text."""
    # Remove common words and extract significant terms
    stop_words = {"the", "a", "an", "is", "are", "to", "for", "and", "or", "in", "on", "at", "this", "that", "it"}
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
    return [w for w in words if w not in stop_words and len(w) > 2][:10]


def _detect_complexity(text: str) -> TaskComplexity:
    """Detect task complexity based on patterns."""
    text_lower = text.lower()
    
    # Check from most complex to least
    for complexity in [TaskComplexity.MASSIVE, TaskComplexity.COMPLEX, 
                       TaskComplexity.MODERATE, TaskComplexity.SIMPLE, TaskComplexity.TRIVIAL]:
        patterns = _COMPLEXITY_PATTERNS.get(complexity, [])
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return complexity
    
    # Default based on length
    if len(text) < 20:
        return TaskComplexity.TRIVIAL
    elif len(text) < 50:
        return TaskComplexity.SIMPLE
    elif len(text) < 150:
        return TaskComplexity.MODERATE
    else:
        return TaskComplexity.COMPLEX


def _should_split(text: str) -> Tuple[bool, List[str]]:
    """Determine if task should be split and suggest subtasks."""
    text_lower = text.lower()
    
    # Check split indicators
    for pattern, condition in _SPLIT_INDICATORS:
        match = re.search(pattern, text_lower)
        if match and condition(match):
            # Try to extract subtasks
            subtasks = _extract_subtasks(text)
            if len(subtasks) > 1:
                return True, subtasks
    
    return False, []


def _extract_subtasks(text: str) -> List[str]:
    """Extract potential subtasks from a complex task."""
    subtasks = []
    
    # Split by numbered list
    numbered = re.findall(r'(?:^|\n)\s*(\d+)[.):]\s*(.+?)(?=(?:\n\s*\d+[.):])|\Z)', text, re.DOTALL)
    if numbered:
        subtasks = [item[1].strip() for item in numbered if item[1].strip()]
        if subtasks:
            return subtasks
    
    # Split by bullet points
    bullets = re.findall(r'(?:^|\n)\s*[-•*]\s*(.+?)(?=(?:\n\s*[-•*])|\Z)', text, re.DOTALL)
    if bullets:
        subtasks = [item.strip() for item in bullets if item.strip()]
        if subtasks:
            return subtasks
    
    # Split by sequential indicators
    sequential = re.split(r'\b(first|then|after that|next|finally|also|and then)\b', text, flags=re.IGNORECASE)
    if len(sequential) > 2:
        subtasks = []
        current = ""
        for part in sequential:
            if part.lower() in ('first', 'then', 'after that', 'next', 'finally', 'also', 'and then'):
                if current.strip():
                    subtasks.append(current.strip())
                current = ""
            else:
                current += part
        if current.strip():
            subtasks.append(current.strip())
        if len(subtasks) > 1:
            return subtasks
    
    # Split by "and" if task seems like a list
    if text.count(' and ') >= 2:
        parts = re.split(r'\s+and\s+', text)
        if len(parts) > 2:
            return [p.strip() for p in parts if p.strip()]
    
    return []


def _should_restructure(text: str) -> Tuple[bool, str, str]:
    """Check if task should be restructured and suggest improvement."""
    text_stripped = text.strip()
    
    for pattern, reason in _RESTRUCTURE_PATTERNS:
        if re.search(pattern, text_stripped, re.IGNORECASE):
            return True, "", reason
    
    # Check for very short tasks that might need context
    if len(text_stripped) < 10 and not text_stripped.endswith('?'):
        return True, "", "Task is very short and may need more context"
    
    return False, "", ""


def _can_batch_with(task1: TaskAnalysis, task2: TaskAnalysis) -> bool:
    """Check if two tasks can be batched together."""
    # Same batch key
    if task1.batch_key and task1.batch_key == task2.batch_key:
        return True
    
    # Similar keywords (>50% overlap)
    if task1.keywords and task2.keywords:
        overlap = len(set(task1.keywords) & set(task2.keywords))
        total = len(set(task1.keywords) | set(task2.keywords))
        if total > 0 and overlap / total > 0.5:
            return True
    
    # Both trivial/simple - can often be batched
    if task1.complexity in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE) and \
       task2.complexity in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE):
        return True
    
    return False


# =============================================================================
# MAIN API
# =============================================================================

def analyze_task(text: str, context: Optional[Dict[str, Any]] = None) -> TaskAnalysis:
    """Analyze a task and determine optimal handling strategy."""
    task_id = _generate_task_id(text)
    
    # Detect complexity
    complexity = _detect_complexity(text)
    
    # Check if should split
    should_split_task, subtasks = _should_split(text)
    
    # Check if should restructure
    should_restructure, restructured, reason = _should_restructure(text)
    
    # Determine batch key
    batch_key = _generate_batch_key(text)
    
    # Extract keywords
    keywords = _extract_keywords(text)
    
    # Determine recommended action
    if should_restructure and not restructured:
        action = TaskAction.RESTRUCTURE
    elif should_split_task and subtasks:
        action = TaskAction.SPLIT_SUBTASKS
    elif batch_key:
        action = TaskAction.BATCH_COMBINE
    else:
        action = TaskAction.EXECUTE_SINGLE
    
    # Estimate API calls
    if action == TaskAction.SPLIT_SUBTASKS:
        estimated_calls = len(subtasks)
    elif action == TaskAction.BATCH_COMBINE:
        estimated_calls = 1  # Will be combined
    else:
        estimated_calls = 1
    
    # Calculate priority score
    priority = 0.5
    if complexity == TaskComplexity.TRIVIAL:
        priority = 0.3
    elif complexity == TaskComplexity.SIMPLE:
        priority = 0.4
    elif complexity == TaskComplexity.MODERATE:
        priority = 0.5
    elif complexity == TaskComplexity.COMPLEX:
        priority = 0.7
    elif complexity == TaskComplexity.MASSIVE:
        priority = 0.9
    
    return TaskAnalysis(
        task_id=task_id,
        original_text=text,
        complexity=complexity,
        recommended_action=action,
        can_batch=bool(batch_key),
        batch_key=batch_key,
        suggested_subtasks=subtasks,
        restructured_text=restructured if restructured else None,
        restructure_reason=reason,
        estimated_api_calls=estimated_calls,
        priority_score=priority,
        keywords=keywords,
    )


def analyze_task_batch(tasks: List[str]) -> Tuple[List[TaskAnalysis], List[TaskBatch]]:
    """Analyze multiple tasks and suggest optimal batching."""
    analyses = [analyze_task(t) for t in tasks]
    batches = []
    
    # Group by batch key
    batch_groups: Dict[str, List[TaskAnalysis]] = {}
    unbatched = []
    
    for analysis in analyses:
        if analysis.can_batch and analysis.batch_key:
            if analysis.batch_key not in batch_groups:
                batch_groups[analysis.batch_key] = []
            batch_groups[analysis.batch_key].append(analysis)
        else:
            unbatched.append(analysis)
    
    # Create batches for groups with multiple tasks
    for batch_key, group in batch_groups.items():
        if len(group) >= 2:
            batch_id = f"batch_{hashlib.md5(batch_key.encode()).hexdigest()[:8]}"
            combined = _combine_tasks_for_batch(group)
            savings = len(group) - 1  # Saved API calls
            
            batches.append(TaskBatch(
                batch_id=batch_id,
                tasks=group,
                combined_prompt=combined,
                estimated_savings=savings,
            ))
        else:
            unbatched.extend(group)
    
    # Check unbatched for additional batching opportunities
    for i, task1 in enumerate(unbatched):
        for j, task2 in enumerate(unbatched[i+1:], i+1):
            if _can_batch_with(task1, task2):
                task1.related_task_ids.append(task2.task_id)
                task2.related_task_ids.append(task1.task_id)
    
    return analyses, batches


def _combine_tasks_for_batch(tasks: List[TaskAnalysis]) -> str:
    """Combine multiple tasks into a single prompt."""
    if len(tasks) == 1:
        return tasks[0].original_text
    
    combined = "Please handle the following related tasks:\n\n"
    for i, task in enumerate(tasks, 1):
        combined += f"{i}. {task.original_text}\n"
    combined += "\nProvide responses for each task."
    
    return combined


def restructure_task(text: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Intelligently restructure a poorly formulated task."""
    analysis = analyze_task(text, context)
    
    if analysis.restructured_text:
        return analysis.restructured_text
    
    # Auto-restructure based on patterns
    text_clean = text.strip()
    
    # Fix capitalization
    if text_clean and text_clean[0].islower():
        text_clean = text_clean[0].upper() + text_clean[1:]
    
    # Add question mark for questions
    question_words = ('what', 'who', 'when', 'where', 'why', 'how', 'which', 'is', 'are', 'can', 'could', 'would', 'should')
    if text_clean.lower().split()[0] in question_words and not text_clean.endswith('?'):
        text_clean += '?'
    
    # Expand abbreviated commands
    abbreviations = {
        r'^pls\s': 'Please ',
        r'^plz\s': 'Please ',
        r'\bthx\b': 'thanks',
        r'\bu\b': 'you',
        r'\br\b': 'are',
        r'\bsmth\b': 'something',
        r'\bsmth\b': 'something',
    }
    for pattern, replacement in abbreviations.items():
        text_clean = re.sub(pattern, replacement, text_clean, flags=re.IGNORECASE)
    
    return text_clean


def split_complex_task(text: str) -> List[str]:
    """Split a complex task into manageable subtasks."""
    analysis = analyze_task(text)
    
    if analysis.suggested_subtasks:
        return analysis.suggested_subtasks
    
    # If no subtasks detected but task is complex, try harder
    if analysis.complexity in (TaskComplexity.COMPLEX, TaskComplexity.MASSIVE):
        # Try sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # Return original if can't split
    return [text]


# =============================================================================
# TASK QUEUE INTEGRATION
# =============================================================================

_pending_analyses: Dict[str, TaskAnalysis] = {}
_pending_batches: List[TaskBatch] = []
_batch_window_ms: int = 500  # Window to collect tasks for batching


async def submit_task(text: str, group: str = "main") -> Tuple[str, TaskAction, Optional[List[str]]]:
    """Submit a task for smart orchestration.
    
    Returns: (task_id, recommended_action, subtasks_or_batch_tasks)
    """
    analysis = analyze_task(text)
    
    with _LOCK:
        _pending_analyses[analysis.task_id] = analysis
    
    # If task should be split, return subtasks
    if analysis.recommended_action == TaskAction.SPLIT_SUBTASKS:
        return analysis.task_id, TaskAction.SPLIT_SUBTASKS, analysis.suggested_subtasks
    
    # If task can be batched, check for pending batchable tasks
    if analysis.can_batch:
        await asyncio.sleep(_batch_window_ms / 1000.0)  # Wait for more tasks
        
        with _LOCK:
            batchable = [a for a in _pending_analyses.values() 
                        if a.can_batch and a.batch_key == analysis.batch_key 
                        and a.task_id != analysis.task_id]
            
            if batchable:
                # Create batch
                batch_tasks = [analysis] + batchable
                for a in batch_tasks:
                    _pending_analyses.pop(a.task_id, None)
                
                return analysis.task_id, TaskAction.BATCH_COMBINE, [a.original_text for a in batch_tasks]
    
    # Remove from pending
    with _LOCK:
        _pending_analyses.pop(analysis.task_id, None)
    
    # If should restructure, return restructured text
    if analysis.recommended_action == TaskAction.RESTRUCTURE:
        restructured = restructure_task(text)
        return analysis.task_id, TaskAction.RESTRUCTURE, [restructured]
    
    return analysis.task_id, TaskAction.EXECUTE_SINGLE, None


def get_task_analysis(task_id: str) -> Optional[TaskAnalysis]:
    """Get analysis for a specific task."""
    return _pending_analyses.get(task_id)


def optimize_task_queue(tasks: List[str]) -> Dict[str, Any]:
    """Optimize a queue of tasks for efficient execution.
    
    Returns optimization plan with batches, splits, and execution order.
    """
    analyses, batches = analyze_task_batch(tasks)
    
    # Build execution plan
    plan = {
        "total_tasks": len(tasks),
        "batches": [],
        "splits": [],
        "single_tasks": [],
        "estimated_api_calls": 0,
        "estimated_savings": 0,
    }
    
    processed_ids = set()
    
    # Add batches
    for batch in batches:
        plan["batches"].append({
            "batch_id": batch.batch_id,
            "task_count": len(batch.tasks),
            "combined_prompt": batch.combined_prompt,
            "savings": batch.estimated_savings,
        })
        plan["estimated_api_calls"] += 1
        plan["estimated_savings"] += batch.estimated_savings
        for t in batch.tasks:
            processed_ids.add(t.task_id)
    
    # Add remaining tasks
    for analysis in analyses:
        if analysis.task_id in processed_ids:
            continue
        
        if analysis.recommended_action == TaskAction.SPLIT_SUBTASKS:
            plan["splits"].append({
                "task_id": analysis.task_id,
                "original": analysis.original_text,
                "subtasks": analysis.suggested_subtasks,
            })
            plan["estimated_api_calls"] += len(analysis.suggested_subtasks)
        else:
            plan["single_tasks"].append({
                "task_id": analysis.task_id,
                "text": analysis.original_text,
                "complexity": analysis.complexity.value,
                "priority": analysis.priority_score,
            })
            plan["estimated_api_calls"] += 1
    
    # Sort single tasks by priority
    plan["single_tasks"].sort(key=lambda x: -x["priority"])
    
    return plan


__all__ = [
    # Analysis
    "analyze_task",
    "analyze_task_batch",
    "restructure_task",
    "split_complex_task",
    # Queue integration
    "submit_task",
    "get_task_analysis",
    "optimize_task_queue",
    # Types
    "TaskAnalysis",
    "TaskBatch",
    "TaskAction",
    "TaskComplexity",
]
