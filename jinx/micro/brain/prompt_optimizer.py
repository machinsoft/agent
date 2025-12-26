"""Adaptive prompt optimization with ML-driven template selection.

Replaces static prompts with context-aware, outcome-driven prompts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

try:
    from jinx.micro.brain.query_classifier import classify_query
except Exception:
    classify_query = None  # type: ignore


@dataclass
class PromptVariant:
    """A prompt template variant."""
    name: str
    template_id: str
    sections: Dict[str, str]  # section_name -> content
    weight: float = 1.0
    uses: int = 0
    successes: int = 0
    avg_quality: float = 0.5


@dataclass
class PromptOutcome:
    """Outcome of using a prompt variant."""
    variant_id: str
    query_intent: str
    query_complexity: int
    response_quality: float
    execution_success: bool
    timestamp: float


class AdaptivePromptOptimizer:
    """ML-driven prompt optimization with A/B testing and context awareness."""
    
    def __init__(self, state_path: str = "log/prompt_optimizer.json"):
        self.state_path = state_path
        
        # Prompt variants per intent
        self.variants: Dict[str, List[PromptVariant]] = defaultdict(list)
        
        # Outcomes for learning
        self.outcomes: deque[PromptOutcome] = deque(maxlen=500)
        
        # Section importance weights (learned)
        self.section_weights: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 0.5)
        )
        
        # Template registry: template_id -> sections
        self.templates: Dict[str, Dict[str, List[str]]] = {}
        
        # Exploration rate
        self.exploration_rate = 0.15
        
        self._lock = asyncio.Lock()
        self._load_state()
        self._register_default_templates()
    
    def _register_default_templates(self) -> None:
        """Register default prompt template sections using modular system."""
        # Get available modules from burning_logic
        try:
            from jinx.prompts.burning_logic import list_modules
            available = list_modules()
        except Exception:
            available = ['identity', 'format', 'pulse', 'code', 'agents']
        
        # Base sections for burning_logic - now modular
        self.templates['burning_logic'] = {
            'core': ['identity', 'format', 'pulse'],  # Always included
            'code': ['code'],  # For code tasks
            'reasoning': ['agents'],  # For complex reasoning
            'context': ['embeddings', 'priority', 'tokens', 'budget'],
            'modes': ['error_recovery', 'architecture', 'api_design'],
        }
        
        # Recovery uses error_recovery module
        self.templates['burning_logic_recovery'] = {
            'core': ['identity', 'format', 'pulse'],
            'mode': ['error_recovery', 'code'],
        }
        
        # Planner variants
        self.templates['planner'] = {
            'mode': ['advisory', 'directive'],
            'depth': ['shallow', 'deep'],
            'format': ['structured', 'freeform'],
        }
    
    def _load_state(self) -> None:
        """Load persisted state."""
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restore variants
                for intent, variants_data in data.get('variants', {}).items():
                    for v_data in variants_data:
                        variant = PromptVariant(
                            name=v_data['name'],
                            template_id=v_data['template_id'],
                            sections=v_data['sections'],
                            weight=v_data.get('weight', 1.0),
                            uses=v_data.get('uses', 0),
                            successes=v_data.get('successes', 0),
                            avg_quality=v_data.get('avg_quality', 0.5)
                        )
                        self.variants[intent].append(variant)
                
                # Restore section weights
                section_weights_data = data.get('section_weights', {})
                for intent, weights in section_weights_data.items():
                    self.section_weights[intent] = defaultdict(lambda: 0.5, weights)
        except Exception:
            pass
    
    async def _save_state(self) -> None:
        """Persist state."""
        try:
            def _write():
                os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
                
                # Serialize variants
                variants_data = {}
                for intent, variants in self.variants.items():
                    variants_data[intent] = [
                        {
                            'name': v.name,
                            'template_id': v.template_id,
                            'sections': v.sections,
                            'weight': v.weight,
                            'uses': v.uses,
                            'successes': v.successes,
                            'avg_quality': v.avg_quality
                        }
                        for v in variants
                    ]
                
                data = {
                    'variants': variants_data,
                    'section_weights': {k: dict(v) for k, v in self.section_weights.items()},
                    'timestamp': time.time()
                }
                
                with open(self.state_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            await asyncio.to_thread(_write)
        except Exception:
            pass
    
    def _estimate_query_complexity(self, query: str) -> int:
        """Estimate query complexity (0-10)."""
        if not query:
            return 3
        
        q = query.strip()
        
        # Length
        len_score = min(5, len(q) / 100)
        
        # Structural
        code_chars = sum(1 for c in q if c in '(){}[]<>=+-*/|&%^~')
        struct_score = min(3, code_chars / 20)
        
        # Lines
        line_score = min(2, len(q.split('\n')) / 5)
        
        return int(min(10, len_score + struct_score + line_score))
    
    async def select_prompt_variant(
        self,
        prompt_type: str,
        query: str,
        context: Optional[Dict[str, object]] = None
    ) -> Dict[str, str]:
        """Select optimal prompt variant for query.
        
        Returns dict of section_name -> content
        """
        async with self._lock:
            complexity = self._estimate_query_complexity(query)
            
            # Get query intent
            intent = 'code_exec'  # default
            if classify_query:
                try:
                    intent_obj = await classify_query(query)
                    intent = intent_obj.intent
                except Exception:
                    pass
            
            # Check if we have learned variants for this intent
            existing_variants = self.variants.get(intent, [])
            
            # Exploration vs exploitation
            explore = (hash(query) % 100) < (self.exploration_rate * 100)
            
            if explore or not existing_variants:
                # Explore: create new variant by mixing sections
                variant = self._create_random_variant(prompt_type, intent, complexity)
            else:
                # Exploit: use best variant
                best_variant = max(existing_variants, key=lambda v: v.weight * v.avg_quality)
                variant = best_variant
            
            # Track usage
            variant.uses += 1
            
            # Ensure variant is in registry
            if intent not in self.variants:
                self.variants[intent] = []
            if variant not in self.variants[intent]:
                self.variants[intent].append(variant)
            
            return variant.sections
    
    def _create_random_variant(
        self,
        prompt_type: str,
        intent: str,
        complexity: int
    ) -> PromptVariant:
        """Create a random variant by selecting sections."""
        template = self.templates.get(prompt_type, {})
        
        sections = {}
        for section_name, options in template.items():
            if options:
                # Weighted random (prefer higher-weight sections)
                weights = [
                    self.section_weights[intent].get(opt, 0.5)
                    for opt in options
                ]
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]
                    import random
                    selected = random.choices(options, weights=weights, k=1)[0]
                else:
                    selected = options[0]
                
                sections[section_name] = selected
        
        # Generate unique variant ID
        variant_id = hashlib.md5(
            f"{prompt_type}|{intent}|{json.dumps(sections, sort_keys=True)}".encode()
        ).hexdigest()[:12]
        
        return PromptVariant(
            name=f"{prompt_type}_{variant_id}",
            template_id=prompt_type,
            sections=sections,
            weight=1.0,
            uses=1,
            successes=0,
            avg_quality=0.5
        )
    
    async def record_outcome(
        self,
        prompt_type: str,
        query: str,
        sections: Dict[str, str],
        response_quality: float,
        execution_success: bool
    ) -> None:
        """Record outcome and update weights."""
        async with self._lock:
            complexity = self._estimate_query_complexity(query)
            
            # Get intent
            intent = 'code_exec'
            if classify_query:
                try:
                    intent_obj = await classify_query(query)
                    intent = intent_obj.intent
                except Exception:
                    pass
            
            # Find matching variant
            variant_id = hashlib.md5(
                f"{prompt_type}|{intent}|{json.dumps(sections, sort_keys=True)}".encode()
            ).hexdigest()[:12]
            
            # Record outcome
            outcome = PromptOutcome(
                variant_id=variant_id,
                query_intent=intent,
                query_complexity=complexity,
                response_quality=response_quality,
                execution_success=execution_success,
                timestamp=time.time()
            )
            
            self.outcomes.append(outcome)
            
            # Update variant stats
            for variant in self.variants.get(intent, []):
                if variant.name.endswith(variant_id):
                    # Update success count
                    if execution_success:
                        variant.successes += 1
                    
                    # Update average quality (EMA)
                    alpha = 0.2
                    variant.avg_quality = alpha * response_quality + (1 - alpha) * variant.avg_quality
                    
                    # Update weight (higher quality = higher weight)
                    variant.weight = variant.avg_quality * (1 + variant.successes / max(1, variant.uses))
                    
                    break
            
            # Update section weights
            for section_name, section_value in sections.items():
                current_weight = self.section_weights[intent][section_value]
                
                # Reward successful sections
                if execution_success and response_quality > 0.6:
                    new_weight = min(1.0, current_weight + 0.05)
                elif not execution_success or response_quality < 0.4:
                    new_weight = max(0.1, current_weight - 0.03)
                else:
                    new_weight = current_weight
                
                self.section_weights[intent][section_value] = new_weight
            
            # Periodically save
            if len(self.outcomes) % 20 == 0:
                await self._save_state()
    
    def get_stats(self) -> Dict[str, object]:
        """Get optimizer statistics."""
        total_variants = sum(len(v) for v in self.variants.values())
        
        if not self.outcomes:
            return {
                'total_variants': total_variants,
                'outcomes_count': 0
            }
        
        avg_quality = sum(o.response_quality for o in self.outcomes) / len(self.outcomes)
        success_rate = sum(1 for o in self.outcomes if o.execution_success) / len(self.outcomes)
        
        # Best variants per intent
        best_variants = {}
        for intent, variants in self.variants.items():
            if variants:
                best = max(variants, key=lambda v: v.avg_quality)
                best_variants[intent] = {
                    'name': best.name,
                    'quality': best.avg_quality,
                    'uses': best.uses
                }
        
        return {
            'total_variants': total_variants,
            'outcomes_count': len(self.outcomes),
            'avg_quality': avg_quality,
            'success_rate': success_rate,
            'best_variants': best_variants,
            'intents_learned': len(self.variants)
        }


# Singleton
_optimizer: Optional[AdaptivePromptOptimizer] = None
_optimizer_lock = asyncio.Lock()


async def get_prompt_optimizer() -> AdaptivePromptOptimizer:
    """Get singleton prompt optimizer."""
    global _optimizer
    if _optimizer is None:
        async with _optimizer_lock:
            if _optimizer is None:
                _optimizer = AdaptivePromptOptimizer()
    return _optimizer


async def select_optimal_prompt(
    prompt_type: str,
    query: str,
    context: Optional[Dict[str, object]] = None
) -> Dict[str, str]:
    """Select optimal prompt variant."""
    optimizer = await get_prompt_optimizer()
    return await optimizer.select_prompt_variant(prompt_type, query, context)


async def record_prompt_outcome(
    prompt_type: str,
    query: str,
    sections: Dict[str, str],
    quality: float,
    success: bool
) -> None:
    """Record prompt outcome."""
    optimizer = await get_prompt_optimizer()
    await optimizer.record_outcome(prompt_type, query, sections, quality, success)


__all__ = [
    "AdaptivePromptOptimizer",
    "PromptVariant",
    "get_prompt_optimizer",
    "select_optimal_prompt",
    "record_prompt_outcome",
]
