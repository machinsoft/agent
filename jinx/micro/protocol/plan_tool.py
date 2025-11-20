from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal

# Snake_case tags to match Rust serde(rename_all = "snake_case")
StepStatus = Literal["pending", "in_progress", "completed"]


@dataclass
class PlanItemArg:
    step: str
    status: StepStatus


@dataclass
class UpdatePlanArgs:
    explanation: Optional[str]
    plan: List[PlanItemArg]
