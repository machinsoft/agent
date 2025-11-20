from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from jinx.micro.protocol.common import AuthMode
from jinx.micro.protocol.config_types import ReasoningEffort


@dataclass
class ReasoningEffortPreset:
    effort: ReasoningEffort
    description: str


@dataclass
class ModelUpgrade:
    id: str
    reasoning_effort_mapping: Optional[Dict[ReasoningEffort, ReasoningEffort]]
    migration_config_key: str


@dataclass
class ModelPreset:
    id: str
    model: str
    display_name: str
    description: str
    default_reasoning_effort: ReasoningEffort
    supported_reasoning_efforts: List[ReasoningEffortPreset]
    is_default: bool
    upgrade: Optional[ModelUpgrade]
    show_in_picker: bool


PRESETS: List[ModelPreset] = [
    ModelPreset(
        id="gpt-5.1-codex-max",
        model="gpt-5.1-codex-max",
        display_name="gpt-5.1-codex-max",
        description="Latest Codex-optimized flagship for deep and fast reasoning.",
        default_reasoning_effort="medium",
        supported_reasoning_efforts=[
            ReasoningEffortPreset(effort="low", description="Fast responses with lighter reasoning"),
            ReasoningEffortPreset(effort="medium", description="Balances speed and reasoning depth for everyday tasks"),
            ReasoningEffortPreset(effort="high", description="Maximizes reasoning depth for complex problems"),
            ReasoningEffortPreset(effort="xhigh", description="Extra high reasoning depth for complex problems"),
        ],
        is_default=True,
        upgrade=None,
        show_in_picker=True,
    ),
    ModelPreset(
        id="gpt-5.1-codex",
        model="gpt-5.1-codex",
        display_name="gpt-5.1-codex",
        description="Optimized for codex.",
        default_reasoning_effort="medium",
        supported_reasoning_efforts=[
            ReasoningEffortPreset(effort="low", description="Fastest responses with limited reasoning"),
            ReasoningEffortPreset(effort="medium", description="Dynamically adjusts reasoning based on the task"),
            ReasoningEffortPreset(effort="high", description="Maximizes reasoning depth for complex or ambiguous problems"),
        ],
        is_default=False,
        upgrade=ModelUpgrade(
            id="gpt-5.1-codex-max",
            reasoning_effort_mapping=None,
            migration_config_key="hide_gpt-5.1-codex-max_migration_prompt",
        ),
        show_in_picker=True,
    ),
    ModelPreset(
        id="gpt-5.1-codex-mini",
        model="gpt-5.1-codex-mini",
        display_name="gpt-5.1-codex-mini",
        description="Optimized for codex. Cheaper, faster, but less capable.",
        default_reasoning_effort="medium",
        supported_reasoning_efforts=[
            ReasoningEffortPreset(effort="medium", description="Dynamically adjusts reasoning based on the task"),
            ReasoningEffortPreset(effort="high", description="Maximizes reasoning depth for complex or ambiguous problems"),
        ],
        is_default=False,
        upgrade=ModelUpgrade(
            id="gpt-5.1-codex-max",
            reasoning_effort_mapping=None,
            migration_config_key="hide_gpt-5.1-codex-max_migration_prompt",
        ),
        show_in_picker=True,
    ),
    ModelPreset(
        id="gpt-5.1",
        model="gpt-5.1",
        display_name="gpt-5.1",
        description="Broad world knowledge with strong general reasoning.",
        default_reasoning_effort="medium",
        supported_reasoning_efforts=[
            ReasoningEffortPreset(
                effort="low",
                description=(
                    "Balances speed with some reasoning; useful for straightforward queries and short explanations"
                ),
            ),
            ReasoningEffortPreset(
                effort="medium",
                description=(
                    "Provides a solid balance of reasoning depth and latency for general-purpose tasks"
                ),
            ),
            ReasoningEffortPreset(
                effort="high",
                description=(
                    "Maximizes reasoning depth for complex or ambiguous problems"
                ),
            ),
        ],
        is_default=False,
        upgrade=ModelUpgrade(
            id="gpt-5.1-codex-max",
            reasoning_effort_mapping=None,
            migration_config_key="hide_gpt-5.1-codex-max_migration_prompt",
        ),
        show_in_picker=True,
    ),
    # Deprecated models below remain hidden from picker where applicable.
    ModelPreset(
        id="gpt-5-codex",
        model="gpt-5-codex",
        display_name="gpt-5-codex",
        description="Optimized for codex.",
        default_reasoning_effort="medium",
        supported_reasoning_efforts=[
            ReasoningEffortPreset(effort="low", description="Fastest responses with limited reasoning"),
            ReasoningEffortPreset(effort="medium", description="Dynamically adjusts reasoning based on the task"),
            ReasoningEffortPreset(effort="high", description="Maximizes reasoning depth for complex or ambiguous problems"),
        ],
        is_default=False,
        upgrade=ModelUpgrade(
            id="gpt-5.1-codex-max",
            reasoning_effort_mapping=None,
            migration_config_key="hide_gpt-5.1-codex-max_migration_prompt",
        ),
        show_in_picker=False,
    ),
    ModelPreset(
        id="gpt-5-codex-mini",
        model="gpt-5-codex-mini",
        display_name="gpt-5-codex-mini",
        description="Optimized for codex. Cheaper, faster, but less capable.",
        default_reasoning_effort="medium",
        supported_reasoning_efforts=[
            ReasoningEffortPreset(effort="medium", description="Dynamically adjusts reasoning based on the task"),
            ReasoningEffortPreset(effort="high", description="Maximizes reasoning depth for complex or ambiguous problems"),
        ],
        is_default=False,
        upgrade=ModelUpgrade(
            id="gpt-5.1-codex-mini",
            reasoning_effort_mapping=None,
            migration_config_key="hide_gpt5_1_migration_prompt",
        ),
        show_in_picker=False,
    ),
    ModelPreset(
        id="gpt-5",
        model="gpt-5",
        display_name="gpt-5",
        description="Broad world knowledge with strong general reasoning.",
        default_reasoning_effort="medium",
        supported_reasoning_efforts=[
            ReasoningEffortPreset(effort="minimal", description="Fastest responses with little reasoning"),
            ReasoningEffortPreset(
                effort="low",
                description=(
                    "Balances speed with some reasoning; useful for straightforward queries and short explanations"
                ),
            ),
            ReasoningEffortPreset(
                effort="medium",
                description=(
                    "Provides a solid balance of reasoning depth and latency for general-purpose tasks"
                ),
            ),
            ReasoningEffortPreset(effort="high", description="Maximizes reasoning depth for complex or ambiguous problems"),
        ],
        is_default=False,
        upgrade=ModelUpgrade(
            id="gpt-5.1-codex-max",
            reasoning_effort_mapping=None,
            migration_config_key="hide_gpt-5.1-codex-max_migration_prompt",
        ),
        show_in_picker=False,
    ),
]


def builtin_model_presets(auth_mode: Optional[AuthMode]) -> List[ModelPreset]:
    """Return presets visible in picker, applying AuthMode-specific filtering.

    For ApiKey mode, hide `gpt-5.1-codex-max` (mirrors Rust logic).
    """
    items = [p for p in PRESETS if p.show_in_picker]
    if auth_mode == AuthMode.apiKey:
        items = [p for p in items if p.id != "gpt-5.1-codex-max"]
    return items


def all_model_presets() -> List[ModelPreset]:
    return list(PRESETS)
