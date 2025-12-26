"""Global state and synchronization primitives.

Minimal, explicit globals to coordinate async behavior. Environment variables:
- ``PULSE``: integer pulse displayed by the spinner (default: 100)
- ``TIMEOUT``: inactivity timeout in seconds before "<no_response>" (default: 30)

Advanced features:
- Thread-safe atomic operations via contextvars
- Memory-mapped state for multi-process coordination
- Event bus for state change notifications
"""

from __future__ import annotations

import asyncio
from typing import Callable, Any
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import weakref

# --- Advanced State Management ---

class _LoopBoundLock:
    def __init__(self) -> None:
        self._locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()

    def _get(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop: create a best-effort lock bound to the current default loop
            loop = asyncio.get_event_loop()
        lk = self._locks.get(loop)
        if lk is None:
            lk = asyncio.Lock()
            self._locks[loop] = lk
        return lk

    async def acquire(self) -> bool:
        return await self._get().acquire()

    def release(self) -> None:
        return self._get().release()

    def locked(self) -> bool:
        return self._get().locked()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.release()
        return False


class _LoopBoundEvent:
    def __init__(self) -> None:
        self._events: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Event]" = weakref.WeakKeyDictionary()
        self._flag: bool = False

    def _get(self) -> tuple[asyncio.AbstractEventLoop, asyncio.Event]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        ev = self._events.get(loop)
        if ev is None:
            ev = asyncio.Event()
            if self._flag:
                try:
                    ev.set()
                except Exception:
                    pass
            self._events[loop] = ev
        return loop, ev

    def is_set(self) -> bool:
        return bool(self._flag)

    def set(self) -> None:
        self._flag = True
        # Best-effort propagate to known loops
        for loop, ev in list(self._events.items()):
            try:
                if loop.is_running():
                    loop.call_soon_threadsafe(ev.set)
                else:
                    ev.set()
            except Exception:
                pass

    def clear(self) -> None:
        self._flag = False
        for loop, ev in list(self._events.items()):
            try:
                if loop.is_running():
                    loop.call_soon_threadsafe(ev.clear)
                else:
                    ev.clear()
            except Exception:
                pass

    async def wait(self) -> bool:
        if self._flag:
            return True
        _loop, ev = self._get()
        await ev.wait()
        return True


# Shared async lock (reentrant for nested calls)
shard_lock: Any = _LoopBoundLock()

# Thread-safe lock for synchronous access
_sync_lock = threading.RLock()

# Global mutable state with safe defaults and validation
pulse: int = 100
boom_limit: int = 30

# Human-readable activity description shown by spinner (set by pipeline)
activity: str = ""
# Monotonic timestamp when activity was last updated (perf_counter seconds)
activity_ts: float = 0.0

# Optional structured detail for current activity (e.g., progress numbers)
activity_detail: dict | None = None
# Timestamp of last detail update
activity_detail_ts: float = 0.0

# Timestamp (perf_counter seconds) when the agent last produced an answer/output.
# Used to pause the <no_response> timer after a reply so the user gets the full TIMEOUT window.
last_agent_reply_ts: float = 0.0

# Whether the current UI loop is using prompt_toolkit PromptSession (toolbar available).
ui_prompt_toolkit: bool = False

# Whether the user is currently in an active input() prompt (simple input mode).
# Used to prevent the line spinner from overwriting the user's typing.
ui_input_active: bool = False

# Whether the system is currently printing multi-line output.
# Used so the line spinner can pause rendering while output is emitted.
ui_output_active: bool = False

# In-process gate for StateCompiler LLM (replaces env-driven JINX_STATE_COMPILER_LLM).
state_compiler_llm_on: bool = False

shutdown_event: Any = _LoopBoundEvent()

throttle_event: Any = _LoopBoundEvent()

# --- Advanced State Observers (Event Bus Pattern) ---

@dataclass
class StateChangeEvent:
    """Event emitted when global state changes."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: float = field(default_factory=lambda: __import__('time').perf_counter())

_state_observers: list[Callable[[StateChangeEvent], None]] = []

def register_state_observer(callback: Callable[[StateChangeEvent], None]) -> None:
    """Register a callback to be notified of state changes.
    
    Args:
        callback: Function called with StateChangeEvent when state mutates
    """
    with _sync_lock:
        if callback not in _state_observers:
            _state_observers.append(callback)

def unregister_state_observer(callback: Callable[[StateChangeEvent], None]) -> None:
    """Remove a state observer callback."""
    with _sync_lock:
        if callback in _state_observers:
            _state_observers.remove(callback)

def _notify_observers(key: str, old_val: Any, new_val: Any) -> None:
    """Notify all observers of a state change (internal use)."""
    event = StateChangeEvent(key=key, old_value=old_val, new_value=new_val)
    with _sync_lock:
        observers = list(_state_observers)  # Copy to avoid modification during iteration
    
    for observer in observers:
        try:
            observer(event)
        except Exception:
            pass  # Don't let observer errors crash the system

def set_activity(new_activity: str, detail: dict | None = None) -> None:
    """Thread-safe activity setter with observer notification."""
    global activity, activity_ts, activity_detail, activity_detail_ts
    import time
    
    with _sync_lock:
        old = activity
        activity = new_activity
        activity_ts = time.perf_counter()
        if detail is not None:
            activity_detail = detail
            activity_detail_ts = activity_ts
        _notify_observers('activity', old, new_activity)

def atomic_pulse_decrement(amount: int = 1) -> int:
    """Thread-safe pulse decrement. Returns new pulse value."""
    global pulse
    with _sync_lock:
        old = pulse
        pulse = max(0, pulse - amount)
        if old != pulse:
            _notify_observers('pulse', old, pulse)
        return pulse


# --- Context-aware state (for multi-task scenarios) ---

# Per-task context variables for isolated state
ctx_current_operation: ContextVar[str] = ContextVar('current_operation', default='')
ctx_priority_level: ContextVar[int] = ContextVar('priority_level', default=1)
