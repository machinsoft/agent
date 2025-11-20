from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

from jinx.micro.protocol.common import AuthMode
from jinx.micro.protocol.v2 import Model, ReasoningEffortOption


def supported_models(auth_mode: Optional[AuthMode]) -> List[Model]:
    """
    Return the list of supported models for the given auth mode.

    Uses a small built-in set; no external configuration is required.
    """
    # Fallback defaults
    return [
        Model(
            id="gpt-4o-mini",
            model="gpt-4o-mini",
            display_name="GPT-4o Mini",
            description="Fast general-purpose reasoning model.",
            supported_reasoning_efforts=[
                ReasoningEffortOption(reasoning_effort="low", description="Low effort"),
                ReasoningEffortOption(reasoning_effort="medium", description="Balanced"),
            ],
            default_reasoning_effort="low",
            is_default=True,
        ),
        Model(
            id="gpt-4o",
            model="gpt-4o",
            display_name="GPT-4o",
            description="High quality reasoning model.",
            supported_reasoning_efforts=[
                ReasoningEffortOption(reasoning_effort="medium", description="Balanced"),
                ReasoningEffortOption(reasoning_effort="high", description="Maximum effort"),
            ],
            default_reasoning_effort="medium",
            is_default=False,
        ),
    ]
