from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from jinx.micro.protocol.common import AuthMode, GitSha
from jinx.micro.protocol.v2 import SandboxCommandAssessment  # reuse
from jinx.micro.protocol.v2 import AskForApprovalType, SandboxModeType

# ---- Initialize ----
@dataclass
class ClientInfo:
    name: str
    version: str
    title: Optional[str] = None


@dataclass
class InitializeParams:
    client_info: ClientInfo


@dataclass
class InitializeResponse:
    user_agent: str


# ---- Conversation lifecycle (v1) ----
ReasoningEffort = str
ReasoningSummary = str
Verbosity = str
SessionSource = str
ReviewDecision = str
TurnAbortReason = str
ForcedLoginMethod = str


@dataclass
class NewConversationParams:
    model: Optional[str] = None
    model_provider: Optional[str] = None
    profile: Optional[str] = None
    cwd: Optional[str] = None
    approval_policy: Optional[AskForApprovalType] = None
    sandbox: Optional[SandboxModeType] = None
    config: Optional[Dict[str, Any]] = None
    base_instructions: Optional[str] = None
    developer_instructions: Optional[str] = None
    compact_prompt: Optional[str] = None
    include_apply_patch_tool: Optional[bool] = None


@dataclass
class NewConversationResponse:
    conversation_id: str
    model: str
    reasoning_effort: Optional[ReasoningEffort]
    rollout_path: str


@dataclass
class ResumeConversationResponse:
    conversation_id: str
    model: str
    initial_messages: Optional[List[Dict[str, Any]]] = None
    rollout_path: str = ""


# Two-branch params for summary
@dataclass
class GetConversationSummaryParamsRolloutPath:
    rollout_path: str


@dataclass
class GetConversationSummaryParamsConversationId:
    conversation_id: str


GetConversationSummaryParams = Union[
    GetConversationSummaryParamsRolloutPath, GetConversationSummaryParamsConversationId
]


@dataclass
class ConversationGitInfo:
    sha: Optional[str]
    branch: Optional[str]
    origin_url: Optional[str]


@dataclass
class ConversationSummary:
    conversation_id: str
    path: str
    preview: str
    timestamp: Optional[str]
    model_provider: str
    cwd: str
    cli_version: str
    source: SessionSource
    git_info: Optional[ConversationGitInfo]


@dataclass
class GetConversationSummaryResponse:
    summary: ConversationSummary


@dataclass
class ListConversationsParams:
    page_size: Optional[int] = None
    cursor: Optional[str] = None
    model_providers: Optional[List[str]] = None


@dataclass
class ListConversationsResponse:
    items: List[ConversationSummary]
    next_cursor: Optional[str] = None


@dataclass
class ResumeConversationParams:
    path: Optional[str] = None
    conversation_id: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None
    overrides: Optional[NewConversationParams] = None


@dataclass
class AddConversationSubscriptionResponse:
    subscription_id: str


@dataclass
class ArchiveConversationParams:
    conversation_id: str
    rollout_path: str


@dataclass
class ArchiveConversationResponse:
    pass


@dataclass
class RemoveConversationSubscriptionResponse:
    pass


# ---- Auth / Login (legacy) ----
@dataclass
class LoginApiKeyParams:
    api_key: str


@dataclass
class LoginApiKeyResponse:
    pass


@dataclass
class LoginChatGptResponse:
    login_id: str
    auth_url: str


@dataclass
class CancelLoginChatGptParams:
    login_id: str


@dataclass
class CancelLoginChatGptResponse:
    pass


@dataclass
class LogoutChatGptParams:
    pass


@dataclass
class LogoutChatGptResponse:
    pass


# ---- Git / Diff ----
@dataclass
class GitDiffToRemoteParams:
    cwd: str


@dataclass
class GitDiffToRemoteResponse:
    sha: GitSha | str
    diff: str


@dataclass
class ExecOneOffCommandResponse:
    exit_code: int
    stdout: str
    stderr: str


# ---- Approvals ----
@dataclass
class ApplyPatchApprovalParams:
    conversation_id: str
    call_id: str
    file_changes: Dict[str, Any]
    reason: Optional[str] = None
    grant_root: Optional[str] = None


@dataclass
class ApplyPatchApprovalResponse:
    decision: ReviewDecision


@dataclass
class ExecCommandApprovalParams:
    conversation_id: str
    call_id: str
    command: List[str]
    cwd: str
    reason: Optional[str]
    risk: Optional[SandboxCommandAssessment]
    parsed_cmd: List[Dict[str, Any]]


@dataclass
class ExecCommandApprovalResponse:
    decision: ReviewDecision


# ---- Turn / Messaging (legacy) ----
@dataclass
class InputItemText:
    type: str = field(default="text", init=False)
    text: str = ""


@dataclass
class InputItemImage:
    type: str = field(default="image", init=False)
    image_url: str = ""


@dataclass
class InputItemLocalImage:
    type: str = field(default="localImage", init=False)
    path: str = ""


InputItem = Union[InputItemText, InputItemImage, InputItemLocalImage]


@dataclass
class SendUserMessageParams:
    conversation_id: str
    items: List[InputItem]


@dataclass
class SendUserMessageResponse:
    pass


@dataclass
class SendUserTurnParams:
    conversation_id: str
    items: List[InputItem]
    cwd: str
    approval_policy: AskForApprovalType
    sandbox_policy: str
    model: str
    effort: Optional[ReasoningEffort]
    summary: ReasoningSummary


@dataclass
class SendUserTurnResponse:
    pass


@dataclass
class InterruptConversationParams:
    conversation_id: str


@dataclass
class InterruptConversationResponse:
    abort_reason: TurnAbortReason


# ---- Misc ----
@dataclass
class GetAuthStatusParams:
    include_token: Optional[bool] = None
    refresh_token: Optional[bool] = None


@dataclass
class GetAuthStatusResponse:
    auth_method: Optional[AuthMode]
    auth_token: Optional[str]
    requires_openai_auth: Optional[bool]


@dataclass
class GetUserAgentResponse:
    user_agent: str


@dataclass
class UserInfoResponse:
    alleged_user_email: Optional[str]


@dataclass
class Tools:
    web_search: Optional[bool]
    view_image: Optional[bool]


@dataclass
class Profile:
    model: Optional[str]
    model_provider: Optional[str]
    approval_policy: Optional[AskForApprovalType]
    model_reasoning_effort: Optional[ReasoningEffort]
    model_reasoning_summary: Optional[ReasoningSummary]
    model_verbosity: Optional[Verbosity]
    chatgpt_base_url: Optional[str]


@dataclass
class SandboxSettings:
    writable_roots: List[str] = field(default_factory=list)
    network_access: Optional[bool] = None
    exclude_tmpdir_env_var: Optional[bool] = None
    exclude_slash_tmp: Optional[bool] = None


@dataclass
class UserSavedConfig:
    approval_policy: Optional[AskForApprovalType]
    sandbox_mode: Optional[SandboxModeType]
    sandbox_settings: Optional[SandboxSettings]
    forced_chatgpt_workspace_id: Optional[str]
    forced_login_method: Optional[ForcedLoginMethod]
    model: Optional[str]
    model_reasoning_effort: Optional[ReasoningEffort]
    model_reasoning_summary: Optional[ReasoningSummary]
    model_verbosity: Optional[Verbosity]
    tools: Optional[Tools]
    profile: Optional[str]
    profiles: Dict[str, Profile] = field(default_factory=dict)


@dataclass
class GetUserSavedConfigResponse:
    config: UserSavedConfig
