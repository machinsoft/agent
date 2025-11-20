from __future__ import annotations

from typing import Literal

# Plan types (lowercase to mirror serde(rename_all = "lowercase"))
PlanType = Literal[
    "free",
    "plus",
    "pro",
    "team",
    "business",
    "enterprise",
    "edu",
    "unknown",
]
