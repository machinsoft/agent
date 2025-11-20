from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
from jinx.micro.backend.types import RateLimitSnapshot
from jinx.micro.protocol.common import AuthMode


# --- User input ---
@dataclass
class UserInput:
    type: Literal["text", "image", "localImage"]
    text: Optional[str] = None
    url: Optional[str] = None
    path: Optional[str] = None

    @staticmethod
    def text_msg(text: str) -> "UserInput":
        return UserInput(type="text", text=text)

    @staticmethod
    def image_url(url: str) -> "UserInput":
        return UserInput(type="image", url=url)


# --- Thread items ---
@dataclass
class ThreadItemUserMessage:
    id: str
    content: List[UserInput]

@dataclass
class ThreadItemAgentMessage:
    id: str
    text: str

@dataclass
class ThreadItemReasoning:
    id: str
    summary: List[str] = field(default_factory=list)
    content: List[str] = field(default_factory=list)

ThreadItem = ThreadItemUserMessage | ThreadItemAgentMessage | ThreadItemReasoning


# --- Turn status ---
@dataclass
class TurnError:
    message: str

@dataclass
class TurnStatusCompleted:
    status: Literal["completed"] = "completed"

@dataclass
class TurnStatusInterrupted:
    status: Literal["interrupted"] = "interrupted"

@dataclass
class TurnStatusFailed:
    status: Literal["failed"] = "failed"
    error: TurnError | None = None

@dataclass
class TurnStatusInProgress:
    status: Literal["inProgress"] = "inProgress"

TurnStatus = (
    TurnStatusCompleted | TurnStatusInterrupted | TurnStatusFailed | TurnStatusInProgress
)


# --- Turn / Thread ---
@dataclass
class Turn:
    id: str
    items: List[ThreadItem]
    status: TurnStatus

@dataclass
class Thread:
    id: str
    preview: str
    model_provider: str
    created_at: int
    path: str
    turns: List[Turn] = field(default_factory=list)


# --- helpers to construct items ---

def make_user_message_item(item_id: str, content: List[UserInput]) -> ThreadItemUserMessage:
    return ThreadItemUserMessage(id=item_id, content=content)


def make_agent_message_item(item_id: str, text: str) -> ThreadItemAgentMessage:
    return ThreadItemAgentMessage(id=item_id, text=text)


def make_reasoning_item(item_id: str, summary: List[str] | None = None, content: List[str] | None = None) -> ThreadItemReasoning:
    return ThreadItemReasoning(id=item_id, summary=summary or [], content=content or [])


# --- Accounts and Login ---
@dataclass
class AccountApiKey:
    type: Literal["apiKey"] = "apiKey"


@dataclass
class AccountChatgpt:
    type: Literal["chatgpt"] = "chatgpt"
    email: str = ""
    plan_type: str = ""


Account = AccountApiKey | AccountChatgpt


@dataclass
class LoginAccountParamsApiKey:
    type: Literal["apiKey"] = "apiKey"
    api_key: str = ""


@dataclass
class LoginAccountParamsChatgpt:
    type: Literal["chatgpt"] = "chatgpt"


LoginAccountParams = LoginAccountParamsApiKey | LoginAccountParamsChatgpt


@dataclass
class LoginAccountResponseApiKey:
    type: Literal["apiKey"] = "apiKey"


@dataclass
class LoginAccountResponseChatgpt:
    type: Literal["chatgpt"] = "chatgpt"
    login_id: str = ""
    auth_url: str = ""


LoginAccountResponse = LoginAccountResponseApiKey | LoginAccountResponseChatgpt


@dataclass
class CancelLoginAccountParams:
    login_id: str


@dataclass
class CancelLoginAccountResponse:
    pass


@dataclass
class LogoutAccountResponse:
    pass


@dataclass
class GetAccountRateLimitsResponse:
    rate_limits: RateLimitSnapshot


@dataclass
class GetAccountParams:
    refresh_token: bool = False


@dataclass
class GetAccountResponse:
    account: Optional[Account]
    requires_openai_auth: bool


# --- Models ---
@dataclass
class ReasoningEffortOption:
    reasoning_effort: str
    description: str


@dataclass
class Model:
    id: str
    model: str
    display_name: str
    description: str
    supported_reasoning_efforts: List[ReasoningEffortOption]
    default_reasoning_effort: str
    is_default: bool


@dataclass
class ModelListParams:
    cursor: Optional[str] = None
    limit: Optional[int] = None


@dataclass
class ModelListResponse:
    data: List[Model]
    next_cursor: Optional[str] = None


# --- Policies & Approvals ---
AskForApprovalType = Literal["unlessTrusted", "onFailure", "onRequest", "never"]
SandboxModeType = Literal["readOnly", "workspaceWrite", "dangerFullAccess"]


@dataclass
class SandboxPolicyWorkspaceWrite:
    type: Literal["workspaceWrite"] = "workspaceWrite"
    writable_roots: List[str] = field(default_factory=list)
    network_access: bool = False
    exclude_tmpdir_env_var: bool = False
    exclude_slash_tmp: bool = False


@dataclass
class SandboxPolicyDangerFullAccess:
    type: Literal["dangerFullAccess"] = "dangerFullAccess"


@dataclass
class SandboxPolicyReadOnly:
    type: Literal["readOnly"] = "readOnly"


SandboxPolicy = SandboxPolicyWorkspaceWrite | SandboxPolicyDangerFullAccess | SandboxPolicyReadOnly


@dataclass
class SandboxCommandAssessment:
    description: str
    risk_level: Literal["low", "medium", "high"]


# --- Thread APIs ---
@dataclass
class ThreadStartParams:
    model: Optional[str] = None
    model_provider: Optional[str] = None
    cwd: Optional[str] = None
    approval_policy: Optional[AskForApprovalType] = None
    sandbox: Optional[SandboxModeType] = None
    config: Optional[Dict[str, Any]] = None
    base_instructions: Optional[str] = None
    developer_instructions: Optional[str] = None


@dataclass
class ThreadStartResponse:
    thread: "Thread"
    model: str
    model_provider: str
    cwd: str
    approval_policy: AskForApprovalType
    sandbox: SandboxPolicy
    reasoning_effort: Optional[str] = None


@dataclass
class ThreadResumeParams:
    thread_id: str
    history: Optional[List[Dict[str, Any]]] = None
    path: Optional[str] = None
    model: Optional[str] = None
    model_provider: Optional[str] = None
    cwd: Optional[str] = None
    approval_policy: Optional[AskForApprovalType] = None
    sandbox: Optional[SandboxModeType] = None
    config: Optional[Dict[str, Any]] = None
    base_instructions: Optional[str] = None
    developer_instructions: Optional[str] = None


@dataclass
class ThreadResumeResponse:
    thread: "Thread"
    model: str
    model_provider: str
    cwd: str
    approval_policy: AskForApprovalType
    sandbox: SandboxPolicy
    reasoning_effort: Optional[str] = None


@dataclass
class ThreadArchiveParams:
    thread_id: str


@dataclass
class ThreadArchiveResponse:
    pass


@dataclass
class ThreadListParams:
    cursor: Optional[str] = None
    limit: Optional[int] = None
    model_providers: Optional[List[str]] = None


@dataclass
class ThreadListResponse:
    data: List[Thread]
    next_cursor: Optional[str] = None


@dataclass
class ThreadCompactParams:
    thread_id: str


@dataclass
class ThreadCompactResponse:
    pass


# --- Turn APIs ---
@dataclass
class TurnStartParams:
    thread_id: str
    input: List[UserInput]
    cwd: Optional[str] = None
    approval_policy: Optional[AskForApprovalType] = None
    sandbox_policy: Optional[SandboxPolicy] = None
    model: Optional[str] = None
    effort: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class ReviewTargetBaseBranch:
    type: Literal["baseBranch"] = "baseBranch"
    branch: str = ""


@dataclass
class ReviewTargetCommit:
    type: Literal["commit"] = "commit"
    sha: str = ""
    title: Optional[str] = None


@dataclass
class ReviewTargetUncommittedChanges:
    type: Literal["uncommittedChanges"] = "uncommittedChanges"


@dataclass
class ReviewTargetCustom:
    type: Literal["custom"] = "custom"
    instructions: str = ""


ReviewTarget = (
    ReviewTargetBaseBranch | ReviewTargetCommit | ReviewTargetUncommittedChanges | ReviewTargetCustom
)


@dataclass
class ReviewStartParams:
    thread_id: str
    target: ReviewTarget
    append_to_original_thread: bool = False


@dataclass
class TurnStartResponse:
    turn: Turn


@dataclass
class TurnInterruptParams:
    thread_id: str
    turn_id: str


@dataclass
class TurnInterruptResponse:
    pass


# --- Items and execution ---
@dataclass
class CommandActionRead:
    type: Literal["read"] = "read"
    command: str = ""
    name: str = ""
    path: str = ""


@dataclass
class CommandActionListFiles:
    type: Literal["listFiles"] = "listFiles"
    command: str = ""
    path: Optional[str] = None


@dataclass
class CommandActionSearch:
    type: Literal["search"] = "search"
    command: str = ""
    query: Optional[str] = None
    path: Optional[str] = None


@dataclass
class CommandActionUnknown:
    type: Literal["unknown"] = "unknown"
    command: str = ""


CommandAction = CommandActionRead | CommandActionListFiles | CommandActionSearch | CommandActionUnknown


CommandExecutionStatus = Literal["inProgress", "completed", "failed"]


@dataclass
class FileUpdateChange:
    path: str
    kind: Literal["add", "delete", "update"]
    diff: str
    move_path: Optional[str] = None


PatchApplyStatus = Literal["inProgress", "completed", "failed", "declined"]
McpToolCallStatus = Literal["inProgress", "completed", "failed"]


@dataclass
class McpToolCallResult:
    content: List[Dict[str, Any]]
    structured_content: Optional[Dict[str, Any]] = None


@dataclass
class McpToolCallError:
    message: str


@dataclass
class TodoItem:
    id: str
    text: str
    completed: bool


# --- Notifications ---
@dataclass
class ThreadStartedNotification:
    thread: Thread


@dataclass
class TurnStartedNotification:
    turn: Turn


@dataclass
class TurnCompletedNotification:
    turn: Turn


@dataclass
class ItemStartedNotification:
    item: Dict[str, Any]


@dataclass
class ItemCompletedNotification:
    item: Dict[str, Any]


@dataclass
class AgentMessageDeltaNotification:
    item_id: str
    delta: str


@dataclass
class ReasoningSummaryTextDeltaNotification:
    item_id: str
    delta: str
    summary_index: int


@dataclass
class ReasoningSummaryPartAddedNotification:
    item_id: str
    summary_index: int


@dataclass
class ReasoningTextDeltaNotification:
    item_id: str
    delta: str
    content_index: int


@dataclass
class CommandExecutionOutputDeltaNotification:
    item_id: str
    delta: str


@dataclass
class McpToolCallProgressNotification:
    item_id: str
    message: str


@dataclass
class AccountUpdatedNotification:
    auth_mode: Optional[AuthMode]


@dataclass
class WindowsWorldWritableWarningNotification:
    sample_paths: List[str]
    extra_count: int
    failed_scan: bool


# --- Feedback ---
@dataclass
class FeedbackUploadResponse:
    thread_id: str
