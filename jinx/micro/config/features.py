from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set

Stage = str  # "experimental" | "beta" | "stable" | "deprecated" | "removed"


@dataclass(frozen=True)
class FeatureSpec:
    id: str
    key: str
    stage: Stage
    default_enabled: bool


# Registry mirroring Rust FEATURES list
FEATURES: tuple[FeatureSpec, ...] = (
    # Stable
    FeatureSpec(id="ghost_commit", key="undo", stage="stable", default_enabled=True),
    FeatureSpec(id="view_image_tool", key="view_image_tool", stage="stable", default_enabled=True),
    FeatureSpec(id="web_search_request", key="web_search_request", stage="stable", default_enabled=False),
    FeatureSpec(id="exec_policy", key="exec_policy", stage="experimental", default_enabled=True),
    FeatureSpec(id="shell_tool", key="shell_tool", stage="stable", default_enabled=True),
    # Experimental / Beta
    FeatureSpec(id="unified_exec", key="unified_exec", stage="experimental", default_enabled=False),
    FeatureSpec(id="shell_command_tool", key="shell_command_tool", stage="experimental", default_enabled=False),
    FeatureSpec(id="rmcp_client", key="rmcp_client", stage="experimental", default_enabled=False),
    FeatureSpec(id="apply_patch_freeform", key="apply_patch_freeform", stage="beta", default_enabled=False),
    FeatureSpec(
        id="sandbox_command_assessment",
        key="experimental_sandbox_command_assessment",
        stage="experimental",
        default_enabled=False,
    ),
    FeatureSpec(
        id="windows_sandbox",
        key="enable_experimental_windows_sandbox",
        stage="experimental",
        default_enabled=False,
    ),
    FeatureSpec(id="remote_compaction", key="remote_compaction", stage="experimental", default_enabled=True),
    FeatureSpec(id="parallel_tool_calls", key="parallel", stage="experimental", default_enabled=False),
)


_KEY_TO_SPEC: Dict[str, FeatureSpec] = {spec.key: spec for spec in FEATURES}


def is_known_feature_key(key: str) -> bool:
    return key in _KEY_TO_SPEC


class Features:
    def __init__(self) -> None:
        self._enabled: Set[str] = {spec.id for spec in FEATURES if spec.default_enabled}

    @staticmethod
    def with_defaults() -> "Features":
        return Features()

    def enabled(self, feature_id: str) -> bool:
        return feature_id in self._enabled

    def enable(self, feature_id: str) -> "Features":
        self._enabled.add(feature_id)
        return self

    def disable(self, feature_id: str) -> "Features":
        self._enabled.discard(feature_id)
        return self

    def apply_map(self, m: Dict[str, bool]) -> None:
        for key, on in m.items():
            spec = _KEY_TO_SPEC.get(key)
            if not spec:
                # unknown key → ignore (or log upstream)
                continue
            if on:
                self.enable(spec.id)
            else:
                self.disable(spec.id)

    def keys(self) -> Iterable[str]:
        return (spec.key for spec in FEATURES)
