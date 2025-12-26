from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from jinx.micro.protocol.config_types import ReasoningEffort, Verbosity

# --- Local equivalents of core config/tool types ---
ReasoningSummaryFormat = Literal["none", "experimental"]
ApplyPatchToolType = Literal["function", "freeform"]
ConfigShellToolType = Literal["default", "local", "shell_command"]


@dataclass
class TruncationPolicyTokens:
    type: Literal["tokens"] = "tokens"
    value: int = 10_000


@dataclass
class TruncationPolicyBytes:
    type: Literal["bytes"] = "bytes"
    value: int = 10_000


TruncationPolicy = Union[TruncationPolicyTokens, TruncationPolicyBytes]


# --- Model family ---
@dataclass
class ModelFamily:
    slug: str
    family: str

    needs_special_apply_patch_instructions: bool = False
    supports_reasoning_summaries: bool = False
    default_reasoning_effort: Optional[ReasoningEffort] = None
    reasoning_summary_format: ReasoningSummaryFormat = "none"

    supports_parallel_tool_calls: bool = False
    apply_patch_tool_type: Optional[ApplyPatchToolType] = None

    base_instructions: str = ""
    experimental_supported_tools: List[str] = field(default_factory=list)

    effective_context_window_percent: int = 95
    support_verbosity: bool = False
    default_verbosity: Optional[Verbosity] = None

    shell_type: ConfigShellToolType = "default"
    truncation_policy: TruncationPolicy = field(default_factory=lambda: TruncationPolicyBytes())


def _make(slug: str, family: str, **kw) -> ModelFamily:
    mf = ModelFamily(slug=slug, family=family)
    for k, v in kw.items():
        setattr(mf, k, v)
    return mf


def find_family_for_model(slug: str) -> Optional[ModelFamily]:
    s = slug
    if s.startswith("o3"):
        return _make(
            s,
            "o3",
            supports_reasoning_summaries=True,
            needs_special_apply_patch_instructions=True,
        )
    if s.startswith("o4-mini"):
        return _make(
            s,
            "o4-mini",
            supports_reasoning_summaries=True,
            needs_special_apply_patch_instructions=True,
        )
    if s.startswith("codex-mini-latest"):
        return _make(
            s,
            "codex-mini-latest",
            supports_reasoning_summaries=True,
            needs_special_apply_patch_instructions=True,
            shell_type="local",
        )
    if s.startswith("gpt-4.1"):
        return _make(s, "gpt-4.1", needs_special_apply_patch_instructions=True)
    if s.startswith("gpt-4o"):
        return _make(s, "gpt-4o", needs_special_apply_patch_instructions=True)
    if s.startswith("gpt-3.5"):
        return _make(s, "gpt-3.5", needs_special_apply_patch_instructions=True)
    if s.startswith("test-gpt-5"):
        return _make(
            s,
            s,
            supports_reasoning_summaries=True,
            reasoning_summary_format="experimental",
            experimental_supported_tools=["grep_files", "list_dir", "read_file", "test_sync_tool"],
            supports_parallel_tool_calls=True,
            shell_type="shell_command",
            support_verbosity=True,
            truncation_policy=TruncationPolicyTokens(value=10_000),
        )
    if s.startswith("codex-exp-"):
        return _make(
            s,
            s,
            supports_reasoning_summaries=True,
            reasoning_summary_format="experimental",
            apply_patch_tool_type="freeform",
            experimental_supported_tools=["grep_files", "list_dir", "read_file"],
            shell_type="shell_command",
            supports_parallel_tool_calls=True,
            support_verbosity=True,
            truncation_policy=TruncationPolicyTokens(value=10_000),
        )
    if s.startswith("gpt-5.1-codex-max"):
        return _make(
            s,
            s,
            supports_reasoning_summaries=True,
            reasoning_summary_format="experimental",
            apply_patch_tool_type="freeform",
            shell_type="shell_command",
            supports_parallel_tool_calls=True,
            truncation_policy=TruncationPolicyTokens(value=10_000),
        )
    if s.startswith("gpt-5-codex") or s.startswith("gpt-5.1-codex") or s.startswith("codex-"):
        return _make(
            s,
            s,
            supports_reasoning_summaries=True,
            reasoning_summary_format="experimental",
            apply_patch_tool_type="freeform",
            shell_type="shell_command",
            supports_parallel_tool_calls=True,
            truncation_policy=TruncationPolicyTokens(value=10_000),
        )
    if s.startswith("gpt-5.1"):
        return _make(
            s,
            "gpt-5.1",
            supports_reasoning_summaries=True,
            apply_patch_tool_type="freeform",
            support_verbosity=True,
            default_verbosity="low",
            default_reasoning_effort="medium",
            truncation_policy=TruncationPolicyBytes(value=10_000),
            shell_type="shell_command",
            supports_parallel_tool_calls=True,
        )
    if s.startswith("gpt-5"):
        return _make(
            s,
            "gpt-5",
            supports_reasoning_summaries=True,
            needs_special_apply_patch_instructions=True,
            shell_type="default",
            support_verbosity=True,
            truncation_policy=TruncationPolicyBytes(value=10_000),
        )
    return None


def derive_default_model_family(model: str) -> ModelFamily:
    return ModelFamily(
        slug=model,
        family=model,
        needs_special_apply_patch_instructions=False,
        supports_reasoning_summaries=False,
        reasoning_summary_format="none",
        supports_parallel_tool_calls=False,
        apply_patch_tool_type=None,
        base_instructions="",
        experimental_supported_tools=[],
        effective_context_window_percent=95,
        support_verbosity=False,
        default_verbosity=None,
        default_reasoning_effort=None,
        truncation_policy=TruncationPolicyBytes(value=10_000),
        shell_type="default",
    )
