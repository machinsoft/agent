"""Intelligent Retry System - Evolution-driven retry with learned strategies.

Uses learnings from self_evolution to:
- Apply known successful strategies before retry
- Adjust parameters based on failure patterns
- Skip retries when learnings indicate futility
- Escalate to LLM analysis when local strategies exhausted
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Awaitable

T = TypeVar("T")

# Retry configuration
_MAX_RETRIES = 3
_BASE_DELAY_MS = 100
_MAX_DELAY_MS = 2000
_BACKOFF_FACTOR = 2.0


@dataclass
class RetryContext:
    """Context for intelligent retry decisions."""
    operation: str
    attempt: int
    last_error: Optional[str]
    last_error_type: Optional[str]
    elapsed_ms: float
    applied_strategies: List[str]


@dataclass
class RetryDecision:
    """Decision about whether and how to retry."""
    should_retry: bool
    delay_ms: float
    strategy: Optional[str]
    reason: str
    adjust_params: Dict[str, Any]


def _get_relevant_learnings(error_type: str, context: str) -> List[Tuple[str, float, Optional[str]]]:
    """Get learnings relevant to this error."""
    try:
        from jinx.micro.runtime.self_evolution import get_relevant_learnings
        learnings = get_relevant_learnings("error_pattern", f"{error_type}: {context}", limit=5)
        return [(l.description, l.confidence, l.solution) for l in learnings]
    except Exception:
        return []


def _get_successful_strategies(operation: str) -> List[Tuple[str, float]]:
    """Get successful strategies for this operation type."""
    try:
        from jinx.micro.runtime.self_evolution import get_relevant_learnings
        learnings = get_relevant_learnings("success_strategy", operation, limit=3)
        return [(l.description, l.confidence) for l in learnings if l.confidence > 0.5]
    except Exception:
        return []


def _record_retry_outcome(
    operation: str,
    attempt: int,
    success: bool,
    error_type: Optional[str],
    strategy_used: Optional[str],
) -> None:
    """Record retry outcome for learning."""
    try:
        from jinx.micro.runtime.self_evolution import learn, record_attempt
        
        if success:
            if attempt > 1:
                # Learned that retry worked
                learn(
                    category="success_strategy",
                    description=f"Retry succeeded on attempt {attempt}",
                    context=f"Operation: {operation}",
                    solution=strategy_used,
                    confidence=0.6,
                )
        else:
            # Learned that all retries failed
            learn(
                category="error_pattern",
                description=f"All {attempt} retries failed for {operation}",
                context=f"Error: {error_type}",
                confidence=0.5,
            )
    except Exception:
        pass


def decide_retry(ctx: RetryContext) -> RetryDecision:
    """Make intelligent decision about retry."""
    
    # Check if we've exhausted retries
    if ctx.attempt >= _MAX_RETRIES:
        return RetryDecision(
            should_retry=False,
            delay_ms=0,
            strategy=None,
            reason="Max retries exhausted",
            adjust_params={},
        )
    
    # Get relevant learnings for this error
    learnings = _get_relevant_learnings(ctx.last_error_type or "", ctx.last_error or "")
    
    # Check if learnings indicate this error is not retryable
    for desc, confidence, solution in learnings:
        if confidence > 0.7 and "not retryable" in desc.lower():
            return RetryDecision(
                should_retry=False,
                delay_ms=0,
                strategy=None,
                reason=f"Learned: {desc}",
                adjust_params={},
            )
    
    # Get successful strategies
    strategies = _get_successful_strategies(ctx.operation)
    
    # Select strategy not yet tried
    strategy_to_try = None
    for strat_desc, confidence in strategies:
        if strat_desc not in ctx.applied_strategies and confidence > 0.5:
            strategy_to_try = strat_desc
            break
    
    # Calculate delay with exponential backoff
    delay = min(_MAX_DELAY_MS, _BASE_DELAY_MS * (_BACKOFF_FACTOR ** ctx.attempt))
    
    # Adjust params based on error type
    adjust_params: Dict[str, Any] = {}
    if ctx.last_error_type == "TimeoutError":
        adjust_params["timeout_multiplier"] = 1.5
    elif ctx.last_error_type == "ConnectionError":
        adjust_params["delay_multiplier"] = 2.0
        delay *= 2
    
    return RetryDecision(
        should_retry=True,
        delay_ms=delay,
        strategy=strategy_to_try,
        reason=f"Attempt {ctx.attempt + 1}/{_MAX_RETRIES}",
        adjust_params=adjust_params,
    )


async def with_intelligent_retry(
    operation: str,
    func: Callable[[], Awaitable[T]],
    max_retries: int = _MAX_RETRIES,
) -> T:
    """Execute function with intelligent retry logic.
    
    Uses evolution learnings to decide when and how to retry.
    """
    ctx = RetryContext(
        operation=operation,
        attempt=0,
        last_error=None,
        last_error_type=None,
        elapsed_ms=0,
        applied_strategies=[],
    )
    
    start_time = time.perf_counter()
    last_exception: Optional[Exception] = None
    
    while ctx.attempt < max_retries:
        try:
            result = await func()
            
            # Record success
            _record_retry_outcome(
                operation=operation,
                attempt=ctx.attempt + 1,
                success=True,
                error_type=None,
                strategy_used=ctx.applied_strategies[-1] if ctx.applied_strategies else None,
            )
            
            return result
            
        except Exception as e:
            last_exception = e
            ctx.attempt += 1
            ctx.last_error = str(e)
            ctx.last_error_type = type(e).__name__
            ctx.elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Decide whether to retry
            decision = decide_retry(ctx)
            
            if not decision.should_retry:
                # Record final failure
                _record_retry_outcome(
                    operation=operation,
                    attempt=ctx.attempt,
                    success=False,
                    error_type=ctx.last_error_type,
                    strategy_used=None,
                )
                break
            
            # Apply strategy if available
            if decision.strategy:
                ctx.applied_strategies.append(decision.strategy)
            
            # Wait before retry
            if decision.delay_ms > 0:
                await asyncio.sleep(decision.delay_ms / 1000)
    
    # All retries failed
    if last_exception:
        raise last_exception
    raise RuntimeError(f"Retry failed for {operation}")


async def retry_with_escalation(
    operation: str,
    func: Callable[[], Awaitable[T]],
    on_escalate: Optional[Callable[[str, str], Awaitable[None]]] = None,
) -> T:
    """Execute with retry, escalating to LLM analysis if all retries fail.
    
    on_escalate is called with (operation, error_details) when escalation needed.
    """
    try:
        return await with_intelligent_retry(operation, func)
    except Exception as e:
        error_details = f"{type(e).__name__}: {str(e)}"
        
        # Escalate to evolution system for LLM analysis
        if on_escalate:
            await on_escalate(operation, error_details)
        else:
            # Default escalation: trigger evolution analysis
            try:
                from jinx.micro.runtime.self_evolution import analyze_failures_with_llm
                await analyze_failures_with_llm(force=False)
            except Exception:
                pass
        
        raise


__all__ = [
    "RetryContext",
    "RetryDecision",
    "decide_retry",
    "with_intelligent_retry",
    "retry_with_escalation",
]
