from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal

from .file_change import FileChange
from .custom_prompts import CustomPrompt
from .conversation_id import ConversationId
from .v2 import AskForApprovalType, SandboxPolicy


# --- Basic events used by Turn/Items legacy mapping ---
@dataclass
class AgentMessageEvent:
    message: str


@dataclass
class UserMessageEvent:
    message: str
    images: Optional[List[str]] = None


@dataclass
class AgentReasoningEvent:
    text: str


@dataclass
class AgentReasoningRawContentEvent:
    text: str


# --- Web search ---
@dataclass
class WebSearchBeginEvent:
    call_id: str


@dataclass
class WebSearchEndEvent:
    call_id: str
    query: str


# --- Tool call: view image ---
@dataclass
class ViewImageToolCallEvent:
    call_id: str
    path: str


# --- Patch apply lifecycle ---
@dataclass
class PatchApplyBeginEvent:
    call_id: str
    turn_id: str
    auto_approved: bool
    changes: Dict[str, FileChange]


@dataclass
class PatchApplyEndEvent:
    call_id: str
    turn_id: str
    stdout: str
    stderr: str
    success: bool
    changes: Dict[str, FileChange]


# --- Turn abort ---
TurnAbortReason = Literal["interrupted", "replaced", "review_ended"]


@dataclass
class TurnAbortedEvent:
    reason: TurnAbortReason


# --- Diffs / history ---
@dataclass
class TurnDiffEvent:
    unified_diff: str


@dataclass
class GetHistoryEntryResponseEvent:
    offset: int
    log_id: int
    entry: Optional[Dict[str, object]] = None  # minimal, opaque


# --- MCP: tools and startup ---
McpAuthStatus = Literal["unsupported", "not_logged_in", "bearer_token", "oauth"]


@dataclass
class McpListToolsResponseEvent:
    tools: Dict[str, Dict[str, object]]
    resources: Dict[str, List[Dict[str, object]]]
    resource_templates: Dict[str, List[Dict[str, object]]]
    auth_statuses: Dict[str, McpAuthStatus]


@dataclass
class McpStartupStatusStarting:
    state: Literal["starting"] = "starting"


@dataclass
class McpStartupStatusReady:
    state: Literal["ready"] = "ready"


@dataclass
class McpStartupStatusFailed:
    state: Literal["failed"] = "failed"
    error: str = ""


@dataclass
class McpStartupStatusCancelled:
    state: Literal["cancelled"] = "cancelled"


McpStartupStatus = (
    McpStartupStatusStarting
    | McpStartupStatusReady
    | McpStartupStatusFailed
    | McpStartupStatusCancelled
)


@dataclass
class McpStartupUpdateEvent:
    server: str
    status: McpStartupStatus


@dataclass
class McpStartupFailure:
    server: str
    error: str


@dataclass
class McpStartupCompleteEvent:
    ready: List[str]
    failed: List[McpStartupFailure]
    cancelled: List[str]


# --- Custom prompts list ---
@dataclass
class ListCustomPromptsResponseEvent:
    custom_prompts: List[CustomPrompt]


# --- Session configured ---
@dataclass
class SessionConfiguredEvent:
    session_id: ConversationId
    model: str
    model_provider_id: str
    approval_policy: AskForApprovalType
    sandbox_policy: SandboxPolicy
    cwd: str
    reasoning_effort: Optional[str]
    history_log_id: int
    history_entry_count: int
    initial_messages: Optional[List[Dict[str, object]]]
    rollout_path: str


# --- Exec output stream ---
ExecOutputStream = Literal["stdout", "stderr"]


@dataclass
class ExecCommandOutputDeltaEvent:
    call_id: str
    stream: ExecOutputStream
    # Base64-encoded bytes chunk per wire format
    chunk: str


# --- Misc background/info events ---
@dataclass
class BackgroundEventEvent:
    message: str


@dataclass
class DeprecationNoticeEvent:
    summary: str
    details: Optional[str] = None


@dataclass
class UndoStartedEvent:
    message: Optional[str] = None


@dataclass
class UndoCompletedEvent:
    success: bool
    message: Optional[str] = None


# --- Envelope ---
@dataclass
class StreamErrorEvent:
    message: str


@dataclass
class StreamInfoEvent:
    message: str

EventMsg = (
    AgentMessageEvent
    | UserMessageEvent
    | AgentReasoningEvent
    | AgentReasoningRawContentEvent
    | WebSearchBeginEvent
    | WebSearchEndEvent
    | StreamErrorEvent
    | StreamInfoEvent
    | PatchApplyBeginEvent
    | PatchApplyEndEvent
    | TurnDiffEvent
    | GetHistoryEntryResponseEvent
    | McpListToolsResponseEvent
    | McpStartupUpdateEvent
    | McpStartupCompleteEvent
    | ListCustomPromptsResponseEvent
    | SessionConfiguredEvent
    | ExecCommandOutputDeltaEvent
    | BackgroundEventEvent
    | DeprecationNoticeEvent
    | UndoStartedEvent
    | UndoCompletedEvent
)


@dataclass
class Event:
    id: str
    msg: EventMsg
