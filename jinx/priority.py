from __future__ import annotations

import asyncio
import heapq
import contextlib
import time
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Pattern, Callable, Optional
from collections import defaultdict
from functools import lru_cache
import threading

from jinx.settings import Settings


@dataclass
class PriorityMetrics:
    """Real-time metrics for priority queue performance."""
    total_processed: int = 0
    high_priority_count: int = 0
    normal_priority_count: int = 0
    low_priority_count: int = 0
    avg_queue_time_ms: float = 0.0
    max_queue_time_ms: float = 0.0
    classification_hits: Dict[int, int] = field(default_factory=dict)
    denied_admissions: int = 0


@dataclass(frozen=True)
class PriorityRule:
    """Configurable priority classification rule."""
    priority: int
    pattern: str
    is_regex: bool = False
    position_limit: Optional[int] = None  # Check only first N chars
    weight: float = 1.0  # For future ML-based scoring


class PriorityClassifier:
    """Advanced priority classifier with caching and extensible rules."""
    
    _instance: Optional['PriorityClassifier'] = None
    _lock = threading.RLock()
    
    def __init__(self):
        self._rules: List[Tuple[int, Pattern | str, Optional[int]]] = []
        self._compiled_patterns: Dict[str, Pattern] = {}
        self._cache_lock = threading.RLock()
        self._setup_default_rules()
    
    @classmethod
    def get_instance(cls) -> 'PriorityClassifier':
        """Thread-safe singleton access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _setup_default_rules(self) -> None:
        """Initialize default priority rules with advanced pattern matching."""
        # Priority 0: Critical/Urgent
        self._add_rule(0, r'^!!', is_regex=True)
        self._add_rule(0, r'^/urgent', is_regex=True)
        self._add_rule(0, r'^/critical', is_regex=True)
        self._add_rule(0, r'\basap\b', is_regex=True, position_limit=20)
        self._add_rule(0, r'\bemergency\b', is_regex=True, position_limit=30)
        
        # Priority 1: High
        self._add_rule(1, r'^![^!]', is_regex=True)  # Single ! but not !!
        self._add_rule(1, r'^/high', is_regex=True)
        self._add_rule(1, r'^/important', is_regex=True)
        self._add_rule(1, r'^/priority', is_regex=True)
        
        # Priority 3: Low
        self._add_rule(3, r'^/low', is_regex=True)
        self._add_rule(3, r'^/defer', is_regex=True)
        self._add_rule(3, r'^/later', is_regex=True)
        
        # Priority 4: Background
        self._add_rule(4, r'^<no_response>$', is_regex=True)
        self._add_rule(4, r'^/background', is_regex=True)
    
    def _add_rule(self, priority: int, pattern: str, is_regex: bool = False, 
                  position_limit: Optional[int] = None) -> None:
        """Add a priority rule with optional regex compilation."""
        if is_regex:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self._compiled_patterns[pattern] = compiled
                self._rules.append((priority, compiled, position_limit))
            except re.error:
                # Fallback to literal string matching
                self._rules.append((priority, pattern.lower(), position_limit))
        else:
            self._rules.append((priority, pattern.lower(), position_limit))
    
    @lru_cache(maxsize=2048)
    def classify(self, msg: str) -> int:
        """Classify message priority with caching and advanced pattern matching.
        
        Returns priority level (0=highest, 4=lowest/background, 2=normal default).
        """
        if not msg:
            return 2
        
        s = msg.strip().lower()
        if not s:
            return 2
        
        # Check rules in order (lower priority value = higher urgency)
        for priority, pattern, position_limit in self._rules:
            text_to_check = s[:position_limit] if position_limit else s
            
            if isinstance(pattern, Pattern):
                # Regex pattern matching
                if pattern.search(text_to_check):
                    return priority
            else:
                # String prefix matching
                if text_to_check.startswith(pattern):
                    return priority
        
        # Default: normal priority
        return 2
    
    def clear_cache(self) -> None:
        """Clear classification cache (useful for testing or config updates)."""
        with self._cache_lock:
            self.classify.cache_clear()
    
    def add_custom_rule(self, rule: PriorityRule) -> None:
        """Add a custom priority rule at runtime."""
        with self._cache_lock:
            self._add_rule(rule.priority, rule.pattern, rule.is_regex, rule.position_limit)
            self.clear_cache()  # Invalidate cache when rules change


# Global classifier instance
_classifier = PriorityClassifier.get_instance()


def classify_priority(msg: str) -> int:
    """
    Advanced message priority classification with multi-level detection.

    Uses configurable pattern-based rules with caching for performance.
    
    Priority levels (lower = higher priority):
    - 0: Critical/Urgent (!, /urgent, ASAP)
    - 1: High (!!, /high, /important)
    - 2: Normal (default)
    - 3: Low (/low, /defer)
    - 4: Background (e.g., '<no_response>', /background)
    """
    return _classifier.classify(msg)


def start_priority_dispatcher_task(src: "asyncio.Queue[str]", dst: "asyncio.Queue[str]", settings: Settings) -> "asyncio.Task[None]":
    async def _run() -> None:
        # Advanced priority dispatcher with metrics and starvation prevention
        try:
            from jinx.observability.otel import span as _span
        except Exception:
            from contextlib import nullcontext as _span  # type: ignore
        loop = asyncio.get_running_loop()
        budget = max(1, settings.runtime.hard_rt_budget_ms) / 1000.0
        next_yield = loop.time() + budget

        heap: List[Tuple[int, int, float, str]] = []  # (priority, seq, enqueue_time, msg)
        seq = 0
        metrics = PriorityMetrics()
        capacity = max(1, int(settings.runtime.queue_maxsize))
        policy = (settings.runtime.queue_policy or "block").strip().lower()
        
        # Starvation prevention: track time since last low-priority dispatch
        last_low_priority_dispatch: Dict[int, float] = defaultdict(lambda: loop.time())
        starvation_threshold_s = 30.0  # Boost priority after 30s wait

        _running = True
        get_task: asyncio.Task[str] | None = None
        flush_task: asyncio.Task[None] | None = None

        async def _dispatch_one() -> bool:
            if not heap:
                return False

            current_time = loop.time()
            modified = False
            for i, (pr0, s0, enq_t0, m0) in enumerate(heap):
                if pr0 >= 2:
                    wait_time = current_time - enq_t0
                    if wait_time >= starvation_threshold_s:
                        heap[i] = (max(0, pr0 - 2), s0, enq_t0, m0)
                        modified = True
            if modified:
                heapq.heapify(heap)

            with _span("priority.dispatch_one"):
                pr1, s1, enq_t1, item = heapq.heappop(heap)
                await dst.put(item)

            queue_time_ms = (current_time - enq_t1) * 1000.0
            metrics.total_processed += 1
            metrics.avg_queue_time_ms = (
                (metrics.avg_queue_time_ms * (metrics.total_processed - 1) + queue_time_ms)
                / metrics.total_processed
            )
            metrics.max_queue_time_ms = max(metrics.max_queue_time_ms, queue_time_ms)
            last_low_priority_dispatch[pr1] = current_time
            return True
        
        async def _awaitable_src_get() -> str:
            if not _running:
                raise asyncio.CancelledError()
            return await src.get()

        try:
            while _running:
                try:
                    if not settings.runtime.use_priority_queue:
                        # Fast path: FIFO pass-through with cooperative yield
                        msg = await src.get()
                        await dst.put(msg)
                        metrics.total_processed += 1
                        if loop.time() >= next_yield:
                            await asyncio.sleep(0)
                            next_yield = loop.time() + budget
                        continue

                    # Create tasks lazily and keep them until they complete.
                    if get_task is None:
                        get_task = asyncio.create_task(_awaitable_src_get())
                    if flush_task is None:
                        flush_task = asyncio.create_task(asyncio.sleep(0.01))

                    with _span("priority.await_or_flush"):
                        done, _ = await asyncio.wait({get_task, flush_task}, return_when=asyncio.FIRST_COMPLETED)

                    if flush_task in done:
                        flush_task = None

                    if get_task in done:
                        try:
                            msg = get_task.result()
                        finally:
                            get_task = None
                        with _span("priority.classify"):
                            pr = classify_priority(msg)
                        enqueue_time = loop.time()
                        # Admission control
                        admitted = True
                        already_pushed = False
                        if len(heap) >= capacity:
                            if policy not in ("drop_newest", "drop_oldest"):
                                heapq.heappush(heap, (pr, seq, enqueue_time, msg))
                                seq += 1
                                already_pushed = True
                                while len(heap) > capacity:
                                    if not await _dispatch_one():
                                        await asyncio.sleep(0.001)
                            elif pr <= 1:
                                # Try to evict one low-priority item to admit critical/high
                                evicted = False
                                if heap:
                                    idx = -1
                                    best_s = 10**12
                                    for i, (p0, s0, t0, m0) in enumerate(heap):
                                        if p0 >= 2 and s0 < best_s:
                                            best_s = s0
                                            idx = i
                                    if idx >= 0:
                                        heap[idx] = heap[-1]
                                        heap.pop()
                                        heapq.heapify(heap)
                                        evicted = True
                                if not evicted:
                                    if policy == "drop_newest":
                                        admitted = False
                                    elif policy == "drop_oldest":
                                        if heap:
                                            idx = 0
                                            best_s = heap[0][1]
                                            for i in range(1, len(heap)):
                                                if heap[i][1] < best_s:
                                                    best_s = heap[i][1]
                                                    idx = i
                                            heap[idx] = heap[-1]
                                            heap.pop()
                                            heapq.heapify(heap)
                                        else:
                                            admitted = False
                                    else:
                                        admitted = False
                            else:
                                if policy == "drop_newest":
                                    admitted = False
                                elif policy == "drop_oldest":
                                    if heap:
                                        idx = 0
                                        best_s = heap[0][1]
                                        for i in range(1, len(heap)):
                                            if heap[i][1] < best_s:
                                                best_s = heap[i][1]
                                                idx = i
                                        heap[idx] = heap[-1]
                                        heap.pop()
                                        heapq.heapify(heap)
                                    else:
                                        admitted = False
                                else:
                                    admitted = False
                        if admitted:
                            if not already_pushed:
                                heapq.heappush(heap, (pr, seq, enqueue_time, msg))
                                seq += 1
                        else:
                            metrics.denied_admissions += 1

                        if pr == 0:
                            metrics.high_priority_count += 1
                        elif pr == 1:
                            metrics.normal_priority_count += 1
                        else:
                            metrics.low_priority_count += 1
                        metrics.classification_hits[pr] = metrics.classification_hits.get(pr, 0) + 1

                    await _dispatch_one()

                    if loop.time() >= next_yield:
                        await asyncio.sleep(0)
                        next_yield = loop.time() + budget
                except asyncio.CancelledError:
                    _running = False
                    break
                except Exception:
                    continue
        finally:
            # Ensure all tasks are cleaned up
            _running = False
            if get_task is not None and not get_task.done():
                get_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_task
            if flush_task is not None and not flush_task.done():
                flush_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await flush_task

    async def _run_with_cleanup():
        """Wrapper to ensure cleanup on cancellation."""
        try:
            await _run()
        except asyncio.CancelledError:
            # Do not cancel unrelated tasks in the event loop.
            # The dispatcher is expected to be safely cancellable by itself.
            return
    
    return asyncio.create_task(_run_with_cleanup(), name="priority-dispatcher-service")
