from __future__ import annotations

from dataclasses import asdict
import os
from typing import Any, Dict, List, Optional

from jinx.micro.app_server.outgoing import OutgoingMessageSender
from jinx.micro.net.jsonrpc import JSONRPCErrorError
from jinx.micro.app_server.error_codes import INTERNAL_ERROR_CODE
from jinx.micro.protocol.v2 import (
    Model,
    ModelListParams,
    Thread,
    Turn,
    TurnStatusInProgress,
    TurnStatusCompleted,
    TurnStatusInterrupted,
    SandboxPolicyReadOnly,
    SandboxPolicyWorkspaceWrite,
    SandboxPolicyDangerFullAccess,
    LoginAccountParamsApiKey,
    LoginAccountParamsChatgpt,
    LoginAccountResponseApiKey,
    LoginAccountResponseChatgpt,
    LogoutAccountResponse,
)
from jinx.micro.app_server.models import supported_models
from jinx.micro.protocol.common import AuthMode
from jinx.micro.backend.client import Client
from jinx.micro.protocol.v1 import (
    GetAuthStatusResponse,
    GetUserSavedConfigResponse,
    UserSavedConfig,
    SetDefaultModelResponse,
    UserInfoResponse,
    NewConversationResponse,
    ResumeConversationResponse,
    ListConversationsResponse,
    ConversationSummary,
    ConversationGitInfo,
    ArchiveConversationResponse,
    SendUserMessageResponse,
    SendUserTurnResponse,
    InterruptConversationResponse,
    AddConversationSubscriptionResponse,
    RemoveConversationSubscriptionResponse,
    ExecOneOffCommandResponse,
    GitDiffToRemoteResponse,
)


class CodexMessageProcessor:
    """Handle higher-level ClientRequest methods.

    Incremental port: implements a small subset and returns error for others.
    """

    def __init__(self, outgoing: OutgoingMessageSender) -> None:
        self._outgoing = outgoing
        # In-memory state (ephemeral) for threads and turns
        self._threads: Dict[str, Thread] = {}
        # Map thread_id -> last turn id
        self._last_turn: Dict[str, str] = {}

    async def process_request(self, method: str, params: Any, request_id: Any) -> None:
        if method == "model/list":
            await self._handle_model_list(params, request_id)
            return
        if method == "account/rateLimits/read":
            await self._handle_account_rate_limits(request_id)
            return
        if method == "thread/start":
            await self._handle_thread_start(params, request_id)
            return
        if method == "thread/resume":
            await self._handle_thread_resume(params, request_id)
            return
        if method == "thread/archive":
            # Remove from local state if present
            if isinstance(params, dict):
                tid = params.get("thread_id") or params.get("threadId")
                if tid and tid in self._threads:
                    self._threads.pop(tid, None)
                    self._last_turn.pop(tid, None)
            await self._outgoing.send_response(request_id, {})
            return
        if method == "thread/list":
            await self._handle_thread_list(request_id)
            return
        if method == "thread/compact":
            await self._outgoing.send_response(request_id, {})
            return
        if method == "turn/start":
            await self._handle_turn_start(params, request_id)
            return
        if method == "turn/interrupt":
            # Update status to interrupted if the turn is known
            if isinstance(params, dict):
                tid = params.get("thread_id") or params.get("threadId")
                turn_id = params.get("turn_id") or params.get("turnId") or (tid and self._last_turn.get(tid))
                if tid and turn_id and tid in self._threads:
                    t = self._threads[tid]
                    for tr in t.turns:
                        if tr.id == turn_id:
                            tr.status = TurnStatusInterrupted()
                            # Notify client of completion
                            await self._outgoing.send_server_notification("turn/completed", {"turn": asdict(tr)})
                            break
            await self._outgoing.send_response(request_id, {})
            return
        if method == "review/start":
            await self._handle_turn_start(params, request_id)
            return
        if method == "account/login/start":
            await self._handle_account_login(params, request_id)
            return
        if method == "account/logout":
            await self._outgoing.send_response(request_id, asdict(LogoutAccountResponse()))
            return
        if method == "account/read":
            await self._handle_account_read(request_id)
            return
        if method == "account/login/cancel":
            await self._outgoing.send_response(request_id, {"loginId": None})
            return
        if method == "getAccountRateLimits":
            await self._handle_account_rate_limits(request_id)
            return
        if method == "newConversation":
            await self._handle_new_conversation(params, request_id)
            return
        if method == "resumeConversation":
            await self._handle_resume_conversation(params, request_id)
            return
        if method == "listConversations":
            await self._outgoing.send_response(request_id, asdict(ListConversationsResponse(items=[], next_cursor=None)))
            return
        if method == "getConversationSummary":
            await self._handle_get_conversation_summary(params, request_id)
            return
        if method == "archiveConversation":
            await self._outgoing.send_response(request_id, asdict(ArchiveConversationResponse()))
            return
        if method == "sendUserMessage":
            await self._outgoing.send_response(request_id, asdict(SendUserMessageResponse()))
            return
        if method == "sendUserTurn":
            await self._outgoing.send_response(request_id, asdict(SendUserTurnResponse()))
            return
        if method == "interruptConversation":
            await self._outgoing.send_response(request_id, asdict(InterruptConversationResponse(abort_reason="replaced")))
            return
        if method == "addConversationListener":
            import uuid
            await self._outgoing.send_response(request_id, asdict(AddConversationSubscriptionResponse(subscription_id=str(uuid.uuid4()))))
            return
        if method == "removeConversationListener":
            await self._outgoing.send_response(request_id, asdict(RemoveConversationSubscriptionResponse()))
            return
        if method == "gitDiffToRemote":
            from jinx.micro.git.info import git_diff_to_remote
            import os as _os
            try:
                cwd = _os.getcwd()
                sha, diff = await git_diff_to_remote(cwd)
            except Exception:
                sha, diff = ("", "")
            await self._outgoing.send_response(request_id, asdict(GitDiffToRemoteResponse(sha=sha, diff=diff)))
            return
        if method == "execOneOffCommand":
            await self._outgoing.send_response(request_id, asdict(ExecOneOffCommandResponse(exit_code=0, stdout="", stderr="")))
            return
        if method == "feedback/upload":
            from jinx.micro.protocol.v2 import FeedbackUploadResponse
            import uuid
            await self._outgoing.send_response(request_id, asdict(FeedbackUploadResponse(thread_id=str(uuid.uuid4()))))
            return
        if method == "getAuthStatus":
            await self._handle_get_auth_status(request_id)
            return
        if method == "getUserSavedConfig":
            await self._handle_get_user_saved_config(request_id)
            return
        if method == "setDefaultModel":
            await self._outgoing.send_response(request_id, asdict(SetDefaultModelResponse()))
            return
        if method == "userInfo":
            await self._outgoing.send_response(request_id, asdict(UserInfoResponse(alleged_user_email=None)))
            return
        # Default: not implemented yet
        await self._outgoing.send_error(
            request_id,
            JSONRPCErrorError(
                code=INTERNAL_ERROR_CODE,
                message=f"{method} is not implemented yet",
                data=None,
            ),
        )

    async def _handle_model_list(self, params: Any, request_id: Any) -> None:
        auth_mode: AuthMode | None = None
        models: List[Model] = supported_models(auth_mode)
        payload: Dict[str, Any] = {
            "data": [asdict(m) for m in models],
            "nextCursor": None,
        }
        await self._outgoing.send_response(request_id, payload)

    async def _handle_account_rate_limits(self, request_id: Any) -> None:
        try:
            client = Client("https://chatgpt.com/backend-api")
            snapshot = await client.get_rate_limits()
            await client.aclose()
            payload = asdict(snapshot)
        except Exception:
            payload = {"primary": None, "secondary": None, "credits": None}
        await self._outgoing.send_response(request_id, payload)

    # --- helpers ---
    async def _handle_thread_start(self, params: Any, request_id: Any) -> None:
        import time, uuid, os as _os
        model = (params or {}).get("model") if isinstance(params, dict) else None
        model = model or os.getenv("OPENAI_MODEL", "gpt-4.1")
        model_provider = None
        if isinstance(params, dict):
            model_provider = params.get("model_provider") or params.get("modelProvider")
        model_provider = model_provider or "openai"
        cwd = (params or {}).get("cwd") if isinstance(params, dict) else None
        cwd = cwd or _os.getcwd()
        approval_policy: Optional[str] = None
        if isinstance(params, dict):
            approval_policy = params.get("approval_policy") or params.get("approvalPolicy")
        approval_policy = approval_policy or "never"
        sandbox_mode: Optional[str] = (params or {}).get("sandbox") if isinstance(params, dict) else None

        if sandbox_mode == "workspaceWrite":
            sandbox = SandboxPolicyWorkspaceWrite()
        elif sandbox_mode == "dangerFullAccess":
            sandbox = SandboxPolicyDangerFullAccess()
        else:
            sandbox = SandboxPolicyReadOnly()

        thread_id = str(uuid.uuid4())
        thread = Thread(
            id=thread_id,
            preview="",
            model_provider=model_provider,
            created_at=int(time.time()),
            path=cwd,
            turns=[],
        )
        # Save state
        self._threads[thread_id] = thread
        self._last_turn.pop(thread_id, None)
        # Notify client
        await self._outgoing.send_server_notification("thread/started", {"thread": asdict(thread)})
        resp = {
            "thread": asdict(thread),
            "model": model,
            "modelProvider": model_provider,
            "cwd": cwd,
            "approvalPolicy": approval_policy,
            "sandbox": asdict(sandbox),
            "reasoningEffort": None,
        }
        await self._outgoing.send_response(request_id, resp)

    async def _handle_thread_resume(self, params: Any, request_id: Any) -> None:
        # Resume by thread_id when provided; otherwise behave like start
        thread_id = None
        if isinstance(params, dict):
            thread_id = params.get("thread_id") or params.get("threadId")
        if thread_id and thread_id in self._threads:
            thread = self._threads[thread_id]
            import os as _os
            resp = {
                "thread": asdict(thread),
                "model": os.getenv("OPENAI_MODEL", "gpt-4.1"),
                "modelProvider": thread.model_provider,
                "cwd": thread.path,
                "approvalPolicy": "never",
                "sandbox": asdict(SandboxPolicyReadOnly()),
                "reasoningEffort": None,
            }
            await self._outgoing.send_response(request_id, resp)
            return
        await self._handle_thread_start(params, request_id)

    async def _handle_turn_start(self, params: Any, request_id: Any) -> None:
        import uuid
        thread_id = None
        if isinstance(params, dict):
            thread_id = params.get("thread_id") or params.get("threadId")
        turn = Turn(id=str(uuid.uuid4()), items=[], status=TurnStatusInProgress())
        if thread_id and thread_id in self._threads:
            t = self._threads[thread_id]
            t.turns.append(turn)
            self._last_turn[thread_id] = turn.id
        # Notify client
        await self._outgoing.send_server_notification("turn/started", {"turn": asdict(turn)})
        await self._outgoing.send_response(request_id, {"turn": asdict(turn)})

    async def _handle_thread_list(self, request_id: Any) -> None:
        # Return shallow thread list (turns empty to keep payload small)
        data = []
        for th in self._threads.values():
            item = asdict(Thread(
                id=th.id,
                preview=th.preview,
                model_provider=th.model_provider,
                created_at=th.created_at,
                path=th.path,
                turns=[],
            ))
            data.append(item)
        await self._outgoing.send_response(request_id, {"data": data, "nextCursor": None})

    async def _handle_account_login(self, params: Any, request_id: Any) -> None:
        import uuid
        if isinstance(params, dict) and params.get("type") == "apiKey":
            await self._outgoing.send_response(request_id, asdict(LoginAccountResponseApiKey()))
            return
        # chatgpt flow: return login id and default auth url
        auth_url = "https://chat.openai.com"
        payload = LoginAccountResponseChatgpt(login_id=str(uuid.uuid4()), auth_url=auth_url)
        await self._outgoing.send_response(request_id, asdict(payload))

    async def _handle_account_read(self, request_id: Any) -> None:
        # Determine auth from OPENAI_API_KEY presence
        has_key = bool(os.getenv("OPENAI_API_KEY"))
        resp = {"account": None, "requiresOpenaiAuth": not has_key}
        await self._outgoing.send_response(request_id, resp)

    async def _handle_get_auth_status(self, request_id: Any) -> None:
        auth_method = "apiKey" if os.getenv("OPENAI_API_KEY") else None
        resp = GetAuthStatusResponse(
            auth_method=auth_method,  # type: ignore[arg-type]
            auth_token=None,
            requires_openai_auth=False,
        )
        await self._outgoing.send_response(request_id, asdict(resp))

    async def _handle_get_user_saved_config(self, request_id: Any) -> None:
        cfg = UserSavedConfig(
            approval_policy=None,
            sandbox_mode=None,
            sandbox_settings=None,
            forced_chatgpt_workspace_id=None,
            forced_login_method=None,
            model=None,
            model_reasoning_effort=None,
            model_reasoning_summary=None,
            model_verbosity=None,
            tools=None,
            profile=None,
            profiles={},
        )
        resp = GetUserSavedConfigResponse(config=cfg)
        await self._outgoing.send_response(request_id, asdict(resp))

    async def _handle_new_conversation(self, params: Any, request_id: Any) -> None:
        import uuid, os as _os
        model = None
        if isinstance(params, dict):
            model = params.get("model")
        model = model or os.getenv("OPENAI_MODEL", "gpt-4.1")
        conv_id = str(uuid.uuid4())
        resp = NewConversationResponse(
            conversation_id=conv_id,
            model=model,
            reasoning_effort=None,
            rollout_path=_os.getcwd(),
        )
        await self._outgoing.send_response(request_id, asdict(resp))

    async def _handle_resume_conversation(self, params: Any, request_id: Any) -> None:
        import uuid, os as _os
        model = None
        if isinstance(params, dict):
            overrides = params.get("overrides") or {}
            if isinstance(overrides, dict):
                model = overrides.get("model")
        model = model or os.getenv("OPENAI_MODEL", "gpt-4.1")
        conv_id = (params or {}).get("conversation_id") if isinstance(params, dict) else None
        conv_id = conv_id or str(uuid.uuid4())
        resp = ResumeConversationResponse(
            conversation_id=conv_id,
            model=model,
            initial_messages=None,
            rollout_path=_os.getcwd(),
        )
        await self._outgoing.send_response(request_id, asdict(resp))

    async def _handle_get_conversation_summary(self, params: Any, request_id: Any) -> None:
        import os as _os
        from datetime import datetime
        conv_id = None
        if isinstance(params, dict):
            conv_id = params.get("conversation_id") or params.get("conversationId")
        conv_id = conv_id or "00000000-0000-0000-0000-000000000000"
        summary = ConversationSummary(
            conversation_id=conv_id,
            path=_os.getcwd(),
            preview="",
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_provider="openai",
            cwd=_os.getcwd(),
            cli_version="0.0.0",
            source="CLI",
            git_info=ConversationGitInfo(sha=None, branch=None, origin_url=None),
        )
        await self._outgoing.send_response(request_id, asdict(ListConversationsResponse(items=[summary], next_cursor=None)))
