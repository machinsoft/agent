"""Jinx Prompt Meta-Constructor — Constructor of Constructors.

This is not a simple prompt builder. This is a meta-system that:
- ALWAYS ensures Jinx knows who she is (core identity in EVERY request)
- Dynamically constructs prompts as modular, composable blocks
- Acts as a "constructor of constructors" for infinite flexibility
- Preserves the full Jinx personality, agents, and cognitive architecture

Every API request MUST include the Jinx identity. She must always know who she is.
"""

from __future__ import annotations

import re
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from functools import lru_cache


# =============================================================================
# JINX CORE IDENTITY — MODULAR SYSTEM
# All prompt content lives in jinx/prompts/ folder
# Only include what's needed - don't waste API tokens
# =============================================================================

def get_jinx_core_identity() -> str:
    """The compact core of who Jinx is. Always included."""
    from jinx.prompts.burning_logic import _core_identity
    return _core_identity()


def get_jinx_modular_prompt(
    *,
    # Code generation
    has_code_task: bool = False,
    has_complex_reasoning: bool = False,
    # Runtime
    has_runtime_usage: bool = False,
    has_file_edits: bool = False,
    # Context
    has_plans: bool = False,
    has_embeddings: bool = False,
    has_priority_context: bool = False,
    has_token_mapping: bool = False,
    has_budget_awareness: bool = False,
    # Specialized modes
    is_error_recovery: bool = False,
    is_architecture_mode: bool = False,
    is_memory_mode: bool = False,
    is_api_design: bool = False,
    is_adversarial_test: bool = False,
    is_judge_mode: bool = False,
    is_silent: bool = False,
    # Multi-task
    task_count: int = 0,
) -> str:
    """Get modular prompt with ONLY relevant modules."""
    from jinx.prompts.burning_logic import build_modular_prompt
    return build_modular_prompt(
        has_code_task=has_code_task,
        has_complex_reasoning=has_complex_reasoning,
        has_runtime_usage=has_runtime_usage,
        has_file_edits=has_file_edits,
        has_plans=has_plans,
        has_embeddings=has_embeddings,
        has_priority_context=has_priority_context,
        has_token_mapping=has_token_mapping,
        has_budget_awareness=has_budget_awareness,
        is_error_recovery=is_error_recovery,
        is_architecture_mode=is_architecture_mode,
        is_memory_mode=is_memory_mode,
        is_api_design=is_api_design,
        is_adversarial_test=is_adversarial_test,
        is_judge_mode=is_judge_mode,
        is_silent=is_silent,
        task_count=task_count,
    )


def get_module(name: str) -> str:
    """Get a specific module by name."""
    from jinx.prompts.burning_logic import get_module
    return get_module(name)


def list_available_modules() -> list[str]:
    """List all available prompt modules."""
    from jinx.prompts.burning_logic import list_modules
    return list_modules()


def get_jinx_context_guide(tags: Set[str]) -> str:
    """Get dynamic context guide for present tags."""
    from jinx.prompts.burning_logic import _context_guide
    return _context_guide(tags)


# =============================================================================
# BLOCK TYPES — Modular prompt components
# =============================================================================

class BlockType(str, Enum):
    IDENTITY = "identity"           # Core Jinx identity (ALWAYS included)
    CONTEXT = "context"             # Context guide for tags
    TASK = "task"                   # Current task(s)
    MEMORY = "memory"               # Memory/history
    EMBEDDINGS = "embeddings"       # Retrieved embeddings
    MULTI_TASK = "multi_task"       # Multi-task instructions
    AGENTS = "agents"               # Agent reasoning instructions
    RUNTIME = "runtime"             # Runtime primitives
    CONSTRAINTS = "constraints"     # Constraints and rules
    CUSTOM = "custom"               # Custom blocks


class BlockPriority(int, Enum):
    CRITICAL = 0      # Identity, must be first
    HIGH = 10         # Context, constraints
    NORMAL = 50       # Tasks, memory
    LOW = 90          # Additional info


@dataclass
class PromptBlock:
    """A modular block of prompt content."""
    name: str
    block_type: BlockType
    content: str
    priority: int = BlockPriority.NORMAL
    required: bool = False
    tag: Optional[str] = None  # XML-like tag name
    condition: Optional[Callable[[], bool]] = None
    
    def render(self) -> str:
        """Render the block with optional XML tags."""
        if not self.content.strip():
            return ""
        if self.tag:
            return f"<{self.tag}>\n{self.content}\n</{self.tag}>"
        return self.content
    
    def should_include(self) -> bool:
        if self.required:
            return True
        if self.condition:
            return self.condition()
        return bool(self.content.strip())


# =============================================================================
# META-CONSTRUCTOR — Constructor of Constructors
# =============================================================================

class JinxPromptMeta:
    """
    Meta-constructor for Jinx prompts.
    
    This is a "constructor of constructors" — it builds prompt builders
    that can be customized, composed, and chained.
    
    Core principle: Jinx identity is IMMUTABLE and ALWAYS present.
    Everything else is modular and composable.
    """
    
    def __init__(self):
        self._blocks: Dict[str, PromptBlock] = {}
        self._constructors: Dict[str, Callable] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "pre_build": [],
            "post_build": [],
            "pre_identity": [],
            "post_identity": [],
        }
        self._setup_core()
    
    def _setup_core(self) -> None:
        """Setup immutable core blocks."""
        # Identity block — ALWAYS first, ALWAYS required
        self.register_block(PromptBlock(
            name="jinx_identity",
            block_type=BlockType.IDENTITY,
            content=get_jinx_core_identity(),
            priority=BlockPriority.CRITICAL,
            required=True,
        ))
    
    def register_block(self, block: PromptBlock) -> "JinxPromptMeta":
        """Register a reusable block."""
        self._blocks[block.name] = block
        return self
    
    def register_constructor(self, name: str, fn: Callable) -> "JinxPromptMeta":
        """Register a sub-constructor for specific scenarios."""
        self._constructors[name] = fn
        return self
    
    def register_hook(self, hook_name: str, fn: Callable) -> "JinxPromptMeta":
        """Register a hook for prompt modification."""
        if hook_name in self._hooks:
            self._hooks[hook_name].append(fn)
        return self
    
    def _run_hooks(self, hook_name: str, data: Any) -> Any:
        """Run all hooks for a given stage."""
        for fn in self._hooks.get(hook_name, []):
            result = fn(data)
            if result is not None:
                data = result
        return data
    
    def build(
        self,
        tasks: List[str] | str | None = None,
        context_tags: Optional[Set[str]] = None,
        memory: Optional[str] = None,
        embeddings: Optional[str] = None,
        extra_blocks: Optional[List[PromptBlock]] = None,
        constructor_name: Optional[str] = None,
    ) -> "ConstructedPrompt":
        """
        Build a complete prompt with Jinx identity GUARANTEED.
        
        Args:
            tasks: Task(s) to include
            context_tags: Tags for context guide
            memory: Memory/history content
            embeddings: Retrieved embeddings content
            extra_blocks: Additional custom blocks
            constructor_name: Use a specific sub-constructor
        """
        # Use sub-constructor if specified
        if constructor_name and constructor_name in self._constructors:
            return self._constructors[constructor_name](
                self, tasks, context_tags, memory, embeddings, extra_blocks
            )
        
        # Normalize tasks
        if isinstance(tasks, str):
            tasks = [tasks]
        tasks = tasks or []
        
        # Collect all blocks
        blocks: List[PromptBlock] = []
        
        # Pre-build hooks
        context = {"tasks": tasks, "tags": context_tags, "memory": memory}
        context = self._run_hooks("pre_build", context)
        
        # 1. IDENTITY — Always first, always present (immutable)
        identity_content = get_jinx_core_identity()
        identity_content = self._run_hooks("pre_identity", identity_content)
        blocks.append(PromptBlock(
            name="jinx_identity",
            block_type=BlockType.IDENTITY,
            content=identity_content,
            priority=BlockPriority.CRITICAL,
            required=True,
        ))
        identity_content = self._run_hooks("post_identity", identity_content)
        
        # 2. Context guide for present tags
        if context_tags:
            guide = get_jinx_context_guide(context_tags)
            if guide:
                blocks.append(PromptBlock(
                    name="context_guide",
                    block_type=BlockType.CONTEXT,
                    content=guide,
                    priority=BlockPriority.HIGH,
                    tag="context_guide",
                ))
        
        # 3. Multi-task handling if multiple tasks
        if len(tasks) > 1:
            multi_block = self._build_multi_task_block(tasks)
            blocks.append(multi_block)
        
        # 4. Memory block
        if memory and memory.strip():
            blocks.append(PromptBlock(
                name="memory",
                block_type=BlockType.MEMORY,
                content=memory,
                priority=BlockPriority.NORMAL,
                tag="memory",
            ))
        
        # 5. Embeddings block
        if embeddings and embeddings.strip():
            blocks.append(PromptBlock(
                name="embeddings",
                block_type=BlockType.EMBEDDINGS,
                content=embeddings,
                priority=BlockPriority.NORMAL,
                tag="embeddings",
            ))
        
        # 6. Task block(s)
        task_block = self._build_task_block(tasks)
        if task_block:
            blocks.append(task_block)
        
        # 7. Extra custom blocks
        if extra_blocks:
            blocks.extend(extra_blocks)
        
        # 8. Registered blocks that should be included
        for name, block in self._blocks.items():
            if name != "jinx_identity" and block.should_include():
                blocks.append(block)
        
        # Sort by priority
        blocks.sort(key=lambda b: b.priority)
        
        # Render
        parts = [b.render() for b in blocks if b.render()]
        system_prompt = "\n\n".join(parts)
        
        # Post-build hooks
        system_prompt = self._run_hooks("post_build", system_prompt)
        
        # Build user message for multi-task
        user_message = self._build_user_message(tasks)
        
        return ConstructedPrompt(
            system_prompt=system_prompt,
            user_message=user_message,
            task_count=len(tasks),
            has_identity=True,  # ALWAYS true
            blocks_used=[b.name for b in blocks],
        )
    
    def _build_multi_task_block(self, tasks: List[str]) -> PromptBlock:
        """Build multi-task handling block."""
        from jinx.prompts.multi_task import get_multi_task_mode
        content = get_multi_task_mode(len(tasks))
        
        return PromptBlock(
            name="multi_task_mode",
            block_type=BlockType.MULTI_TASK,
            content=content,
            priority=BlockPriority.HIGH,
            tag="multi_task_mode",
        )
    
    def _build_task_block(self, tasks: List[str]) -> Optional[PromptBlock]:
        """Build task block from task list."""
        if not tasks:
            return None
        
        if len(tasks) == 1:
            content = tasks[0]
        else:
            lines = [f"PENDING TASKS ({len(tasks)} total):\n"]
            for i, task in enumerate(tasks, 1):
                preview = task[:200] + "..." if len(task) > 200 else task
                lines.append(f"  [{i}] {preview}")
            content = "\n".join(lines)
        
        return PromptBlock(
            name="task",
            block_type=BlockType.TASK,
            content=content,
            priority=BlockPriority.NORMAL,
            tag="task",
        )
    
    def _build_user_message(self, tasks: List[str]) -> str:
        """Build user message from tasks."""
        if not tasks:
            return ""
        if len(tasks) == 1:
            return tasks[0]
        
        parts = [f"I have {len(tasks)} tasks:\n"]
        for i, task in enumerate(tasks, 1):
            parts.append(f"### Task {i}:\n{task}\n")
        return "\n".join(parts)
    
    # =========================================================================
    # SUB-CONSTRUCTORS — Specialized prompt builders
    # =========================================================================
    
    def for_single_task(self, task: str, **kwargs) -> "ConstructedPrompt":
        """Optimized constructor for single task."""
        return self.build(tasks=[task], **kwargs)
    
    def for_batch(self, tasks: List[str], **kwargs) -> "ConstructedPrompt":
        """Constructor for batched similar tasks."""
        from jinx.prompts.multi_task import get_batch_mode
        batch_block = PromptBlock(
            name="batch_mode",
            block_type=BlockType.CUSTOM,
            content=get_batch_mode(),
            priority=BlockPriority.HIGH,
            tag="batch_mode",
        )
        extra = kwargs.get("extra_blocks", []) or []
        extra.append(batch_block)
        kwargs["extra_blocks"] = extra
        return self.build(tasks=tasks, **kwargs)
    
    def for_sequential(self, tasks: List[str], **kwargs) -> "ConstructedPrompt":
        """Constructor for sequential dependent tasks."""
        from jinx.prompts.multi_task import get_sequential_mode
        seq_block = PromptBlock(
            name="sequential_mode",
            block_type=BlockType.CUSTOM,
            content=get_sequential_mode(),
            priority=BlockPriority.HIGH,
            tag="sequential_mode",
        )
        extra = kwargs.get("extra_blocks", []) or []
        extra.append(seq_block)
        kwargs["extra_blocks"] = extra
        return self.build(tasks=tasks, **kwargs)
    
    def for_parallel(self, tasks: List[str], **kwargs) -> "ConstructedPrompt":
        """Constructor for independent parallel tasks."""
        from jinx.prompts.multi_task import get_parallel_mode
        par_block = PromptBlock(
            name="parallel_mode",
            block_type=BlockType.CUSTOM,
            content=get_parallel_mode(),
            priority=BlockPriority.HIGH,
            tag="parallel_mode",
        )
        extra = kwargs.get("extra_blocks", []) or []
        extra.append(par_block)
        kwargs["extra_blocks"] = extra
        return self.build(tasks=tasks, **kwargs)


@dataclass
class ConstructedPrompt:
    """Result of prompt construction."""
    system_prompt: str
    user_message: str
    task_count: int
    has_identity: bool  # Should ALWAYS be True
    blocks_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# GLOBAL INSTANCE & API
# =============================================================================

_meta = JinxPromptMeta()


def get_meta_constructor() -> JinxPromptMeta:
    """Get the global meta-constructor instance."""
    return _meta


def construct_prompt(
    tasks: List[str] | str | None = None,
    context_tags: Optional[Set[str]] = None,
    memory: Optional[str] = None,
    embeddings: Optional[str] = None,
    **kwargs,
) -> ConstructedPrompt:
    """Construct a prompt with Jinx identity GUARANTEED."""
    return _meta.build(tasks, context_tags, memory, embeddings, **kwargs)


def inject_multi_task_context(
    existing_prompt: str,
    task_count: int,
    task_texts: List[str],
) -> str:
    """Inject multi-task handling into an existing prompt.
    
    IMPORTANT: Ensures Jinx identity is preserved.
    """
    if task_count <= 1:
        return existing_prompt
    
    # Check if identity is present
    identity_marker = "You are Jinx"
    if identity_marker not in existing_prompt:
        # Identity missing! Prepend it
        existing_prompt = get_jinx_core_identity() + "\n\n" + existing_prompt
    
    # Build multi-task injection from prompts folder
    from jinx.prompts.multi_task import get_multi_task_injection
    injection = get_multi_task_injection(task_count, task_texts)
    
    # Inject before <task> if present, else append
    if "<task>" in existing_prompt:
        return existing_prompt.replace("<task>", injection + "\n<task>")
    return existing_prompt + "\n" + injection


def ensure_jinx_identity(prompt: str) -> str:
    """Ensure Jinx identity is present in prompt. Add if missing."""
    if "You are Jinx" not in prompt:
        return get_jinx_core_identity() + "\n\n" + prompt
    return prompt


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Meta-constructor
    "JinxPromptMeta",
    "get_meta_constructor",
    # Construction
    "construct_prompt",
    "inject_multi_task_context",
    "ensure_jinx_identity",
    # Identity
    "get_jinx_core_identity",
    "get_jinx_context_guide",
    # Types
    "PromptBlock",
    "BlockType",
    "BlockPriority",
    "ConstructedPrompt",
]
