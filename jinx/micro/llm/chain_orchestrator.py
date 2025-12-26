"""Chain Orchestrator - intelligent coordination of LLM chains with brain systems.

Production-grade orchestration with ML enhancement, real-time constraints, and adaptive optimization.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ChainContext:
    """Enriched chain context with ML intelligence."""
    user_text: str
    expanded_query: Optional[str]
    evidence: Optional[str]
    embeddings: Dict[str, Any]
    ml_params: Dict[str, Any]
    brain_suggestions: List[Tuple[str, float]]
    timestamp: float


@dataclass
class ChainResult:
    """Chain execution result with metrics."""
    success: bool
    data: Dict[str, Any]
    latency_ms: float
    tokens_used: int
    quality_score: float
    error: Optional[str]


class ChainOrchestrator:
    """Orchestrates LLM chain execution with brain systems integration."""
    
    def __init__(self):
        self._execution_history: List[Tuple[float, bool, float]] = []
        self._lock = asyncio.Lock()
    
    async def execute_planning_chain(
        self,
        user_text: str,
        *,
        max_subqueries: Optional[int] = None,
        timeout_ms: int = 5000
    ) -> ChainResult:
        """Execute planning chain with full brain integration and structured blocks."""
        t0 = time.time()
        
        # Step 1: Enhance with brain systems
        context = await self._build_enhanced_context(user_text)
        
        # Step 2: Execute with timeout
        result_data = None
        error = None
        success = False
        
        async def _execute():
            nonlocal result_data, success
            from jinx.micro.llm.chain_plan import run_planner
            result_data = await run_planner(
                context.expanded_query or context.user_text,
                max_subqueries=max_subqueries
            )
            success = bool(result_data and result_data.get('sub_queries'))
        
        # Execute with timeout
        if timeout_ms > 0:
            await asyncio.wait_for(_execute(), timeout=timeout_ms / 1000.0)
        else:
            await _execute()
        
        latency_ms = (time.time() - t0) * 1000
        
        # Step 3: Assess quality
        quality = await self._assess_quality(result_data, context)
        
        # Step 4: Build structured blocks (similar to embeddings)
        structured_data = await self._build_structured_output(
            result_data,
            context,
            latency_ms,
            quality,
            success
        )
        
        # Step 5: Record for learning
        await self._record_execution(success, latency_ms, quality)
        
        return ChainResult(
            success=success,
            data=structured_data,
            latency_ms=latency_ms,
            tokens_used=0,
            quality_score=quality,
            error=error
        )
    
    async def _build_structured_output(
        self,
        result_data: Optional[Dict[str, Any]],
        context: ChainContext,
        latency_ms: float,
        quality: float,
        success: bool
    ) -> Dict[str, Any]:
        """Build structured output with chain blocks."""
        from jinx.micro.llm.chain_blocks import (
            build_chain_plan_block,
            build_chain_intelligence_block,
            build_chain_meta_block,
            build_chain_outcome_block,
            compact_chain_blocks
        )
        
        # Build plan block
        plan_block = build_chain_plan_block(result_data or {})
        
        # Build intelligence block
        intel_data = {
            'expanded_query': context.expanded_query,
            'memory_hints': [
                {'content': str(m)} for m in context.embeddings.get('memories', [])
            ],
            'brain_suggestions': context.brain_suggestions,
            'adaptive_params': context.ml_params
        }
        intel_block = build_chain_intelligence_block(intel_data)
        
        # Build meta block
        meta_data = {
            'mappings': {},
            'execution_time_ms': latency_ms,
            'cache_hit': False,
            'quality_score': quality
        }
        # Add subquery mappings
        if result_data and result_data.get('sub_queries'):
            for i, sq in enumerate(result_data['sub_queries'], 1):
                meta_data['mappings'][f'Q{i}'] = f"subquery: {sq}"
        
        meta_block = build_chain_meta_block(meta_data)
        
        # Build outcome block
        outcome_data = {
            'success': success,
            'phase': 'planning',
            'subqueries_generated': len(result_data.get('sub_queries', [])) if result_data else 0,
            'intelligence_applied': 'brain_enhanced' if context.expanded_query else 'direct',
            'latency_ms': latency_ms,
            'quality': quality
        }
        outcome_block = build_chain_outcome_block(outcome_data)
        
        # Compact all blocks
        compacted = compact_chain_blocks(
            plan=plan_block,
            intelligence=intel_block,
            meta=meta_block,
            outcome=outcome_block,
            max_chars=5000
        )
        
        # Return structured data
        return {
            **result_data,
            'chain_blocks': compacted,
            'structured': {
                'plan': plan_block,
                'intelligence': intel_block,
                'meta': meta_block,
                'outcome': outcome_block
            }
        }
    
    async def execute_context_chain(
        self,
        user_text: str,
        *,
        timeout_ms: int = 3000
    ) -> ChainResult:
        """Execute context building chain with ML optimization."""
        t0 = time.time()
        
        # Use brain systems for context optimization
        context = await self._build_enhanced_context(user_text)
        
        # Determine optimal parameters
        from jinx.micro.brain import select_retrieval_params
        k, retrieval_timeout = await select_retrieval_params(
            user_text,
            {'chain_type': 'context'}
        )
        
        # Execute context gathering
        result_data = None
        success = False
        
        async def _execute():
            nonlocal result_data, success
            from jinx.micro.llm.chain_context import gather_context_for_subs
            result_data = await gather_context_for_subs(
                context.user_text,
                subqueries=[],  # Will be populated from plan
                k=k,
                timeout_ms=retrieval_timeout
            )
            success = bool(result_data)
        
        # Execute with timeout
        await asyncio.wait_for(_execute(), timeout=timeout_ms / 1000.0)
        
        latency_ms = (time.time() - t0) * 1000
        quality = 0.8 if success else 0.3
        
        await self._record_execution(success, latency_ms, quality)
        
        return ChainResult(
            success=success,
            data={'context': result_data} if result_data else {},
            latency_ms=latency_ms,
            tokens_used=0,
            quality_score=quality,
            error=None if success else "context_failed"
        )
    
    async def _build_enhanced_context(self, user_text: str) -> ChainContext:
        """Build enhanced context with brain systems."""
        # Query expansion
        expanded_query = None
        try:
            from jinx.micro.brain import expand_query
            expanded_result = await expand_query(user_text)
            if expanded_result and expanded_result.confidence > 0.6:
                expanded_query = expanded_result.expanded
        except Exception:
            expanded_query = None
        
        # Search memories for context
        embeddings = {}
        try:
            from jinx.micro.brain import search_all_memories
            memories = await search_all_memories(user_text, k=5)
            embeddings['memories'] = [
                {'content': m.content, 'importance': m.importance}
                for m in memories
            ]
        except Exception:
            embeddings = {}
        
        # Get brain suggestions
        brain_suggestions = []
        try:
            from jinx.micro.brain import get_knowledge_graph
            kg = await get_knowledge_graph()
            nodes = await kg.query_patterns(user_text, 'similar')
            brain_suggestions = [
                (str(node.get('data')), node.get('confidence', 0.5))
                for node in nodes[:3]
            ]
        except Exception:
            brain_suggestions = []
        
        # ML parameters
        ml_params = {}
        try:
            from jinx.micro.brain import select_retrieval_params
            k, timeout = await select_retrieval_params(user_text, {})
            ml_params = {'k': k, 'timeout_ms': timeout}
        except Exception:
            ml_params = {}
        
        return ChainContext(
            user_text=user_text,
            expanded_query=expanded_query,
            evidence=None,  # TODO: gather evidence
            embeddings=embeddings,
            ml_params=ml_params,
            brain_suggestions=brain_suggestions,
            timestamp=time.time()
        )
    
    async def _assess_quality(
        self,
        result: Optional[Dict[str, Any]],
        context: ChainContext
    ) -> float:
        """Assess quality of chain result."""
        if not result:
            return 0.0
        
        score = 0.5  # Base score
        
        # Check sub-queries quality
        if result.get('sub_queries'):
            score += 0.3
            if len(result['sub_queries']) >= 3:
                score += 0.1
        
        # Check if expanded query was used successfully
        if context.expanded_query and result.get('sub_queries'):
            score += 0.1
        
        return min(1.0, score)
    
    async def _record_execution(
        self,
        success: bool,
        latency_ms: float,
        quality: float
    ) -> None:
        """Record execution for learning."""
        async with self._lock:
            self._execution_history.append((time.time(), success, latency_ms))
            
            # Keep last 100 executions
            if len(self._execution_history) > 100:
                self._execution_history = self._execution_history[-100:]
        
        # Record to brain systems
        try:
            from jinx.micro.brain import record_outcome
            await record_outcome(
                'llm_chain',
                success,
                {'quality': quality},
                latency_ms=latency_ms
            )
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        if not self._execution_history:
            return {'total': 0}
        
        total = len(self._execution_history)
        successful = sum(1 for _, success, _ in self._execution_history if success)
        avg_latency = sum(lat for _, _, lat in self._execution_history) / total
        
        return {
            'total_executions': total,
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_latency_ms': avg_latency,
            'recent_success': self._execution_history[-1][1] if self._execution_history else False
        }


# Singleton
_orchestrator: Optional[ChainOrchestrator] = None
_orch_lock = asyncio.Lock()


async def get_chain_orchestrator() -> ChainOrchestrator:
    """Get singleton chain orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        async with _orch_lock:
            if _orchestrator is None:
                _orchestrator = ChainOrchestrator()
    return _orchestrator


async def execute_planning_chain_smart(
    user_text: str,
    **kwargs
) -> ChainResult:
    """Execute planning chain with full intelligence."""
    orch = await get_chain_orchestrator()
    return await orch.execute_planning_chain(user_text, **kwargs)


async def execute_context_chain_smart(
    user_text: str,
    **kwargs
) -> ChainResult:
    """Execute context chain with full intelligence."""
    orch = await get_chain_orchestrator()
    return await orch.execute_context_chain(user_text, **kwargs)


__all__ = [
    "ChainOrchestrator",
    "ChainContext",
    "ChainResult",
    "get_chain_orchestrator",
    "execute_planning_chain_smart",
    "execute_context_chain_smart",
]
