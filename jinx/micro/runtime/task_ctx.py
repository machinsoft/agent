from __future__ import annotations

from contextvars import ContextVar
from typing import Dict, List, Any, Optional
from collections import deque
from threading import Lock
import time

# Current logical task group for a conversation turn (e.g., session/thread id)
current_group: ContextVar[str] = ContextVar("jinx_current_group", default="main")

# Current task ID within a group (for memory isolation)
current_task_id: ContextVar[str] = ContextVar("jinx_current_task_id", default="")

# Task sequence number for ordering
current_task_seq: ContextVar[int] = ContextVar("jinx_current_task_seq", default=0)


def get_current_group() -> str:
    try:
        v = current_group.get()
        return (v or "main").strip() or "main"
    except Exception:
        return "main"


def get_current_task_id() -> str:
    try:
        return current_task_id.get() or ""
    except Exception:
        return ""


# Task memory buffer for sequential merge
_task_memory_lock = Lock()
_task_memory_buffers: Dict[str, List[Dict[str, Any]]] = {}  # group -> list of {seq, task_id, entries}
_task_sequence_counter: Dict[str, int] = {}  # group -> next sequence number


def allocate_task_sequence(group: str) -> int:
    """Allocate a sequence number for a new task in a group."""
    with _task_memory_lock:
        seq = _task_sequence_counter.get(group, 0)
        _task_sequence_counter[group] = seq + 1
        return seq


def buffer_task_memory(group: str, task_id: str, seq: int, entry: str) -> None:
    """Buffer a memory entry for later sequential merge."""
    with _task_memory_lock:
        if group not in _task_memory_buffers:
            _task_memory_buffers[group] = []
        _task_memory_buffers[group].append({
            "seq": seq,
            "task_id": task_id,
            "entry": entry,
            "ts": time.time(),
        })


def flush_task_memory(group: str, up_to_seq: int) -> List[str]:
    """Flush buffered memory entries in sequence order up to given seq."""
    with _task_memory_lock:
        if group not in _task_memory_buffers:
            return []
        
        # Sort by sequence number
        buf = _task_memory_buffers[group]
        ready = [e for e in buf if e["seq"] <= up_to_seq]
        ready.sort(key=lambda x: (x["seq"], x["ts"]))
        
        # Remove flushed entries
        _task_memory_buffers[group] = [e for e in buf if e["seq"] > up_to_seq]
        
        return [e["entry"] for e in ready]


def get_pending_task_count(group: str) -> int:
    """Get count of pending memory entries for a group."""
    with _task_memory_lock:
        if group not in _task_memory_buffers:
            return 0
        return len(_task_memory_buffers[group])


def clear_task_memory(group: str) -> None:
    """Clear all buffered memory for a group."""
    with _task_memory_lock:
        _task_memory_buffers.pop(group, None)
        _task_sequence_counter.pop(group, None)
