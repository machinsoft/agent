from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from jinx.micro.app_server.error_codes import (
    INVALID_REQUEST_ERROR_CODE,
)
from jinx.micro.app_server.outgoing import OutgoingMessageSender
from jinx.micro.net.jsonrpc import (
    JSONRPCErrorError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
)
from jinx.micro.protocol.v1 import (
    ClientInfo,
    InitializeResponse,
)
from jinx.micro.protocol.common import (
    FuzzyFileSearchParams,
    FuzzyFileSearchResponse,
    FuzzyFileSearchResult,
)
from jinx.micro.file_search import run_fuzzy_file_search
from jinx.micro.app_server.codex_message_processor import CodexMessageProcessor


 


class MessageProcessor:
    """Process JSON-RPC messages. Handles initialize, then forwards to CodexMessageProcessor."""

    def __init__(
        self,
        outgoing: OutgoingMessageSender,
        codex_linux_sandbox_exe: Optional[str] = None,
        config: Optional[Any] = None,
        feedback: Optional[Any] = None,
    ) -> None:
        self._outgoing = outgoing
        self._initialized = False
        self._codex = CodexMessageProcessor(outgoing)
        self._codex_linux_sandbox_exe = codex_linux_sandbox_exe
        self._config = config
        self._feedback = feedback
        self._user_agent: str | None = None

    async def process_request(self, request: JSONRPCRequest) -> None:
        rid = request.id
        method = request.method
        params = request.params

        if method == "initialize":
            if self._initialized:
                err = JSONRPCErrorError(
                    code=INVALID_REQUEST_ERROR_CODE,
                    message="Already initialized",
                    data=None,
                )
                await self._outgoing.send_error(rid, err)
                return
            # Decode client info if present
            name, version = None, None
            if isinstance(params, dict):
                info = params.get("client_info") or params.get("clientInfo")
                if isinstance(info, dict):
                    name = info.get("name")
                    version = info.get("version")
            # Build user agent (fallbacks allowed)
            ua = self._build_user_agent(name=name, version=version)
            self._user_agent = ua
            resp = InitializeResponse(user_agent=ua)
            await self._outgoing.send_response(rid, asdict(resp))
            self._initialized = True
            return

        if not self._initialized:
            err = JSONRPCErrorError(
                code=INVALID_REQUEST_ERROR_CODE,
                message="Not initialized",
                data=None,
            )
            await self._outgoing.send_error(rid, err)
            return

        # Route selected methods directly until full codex processor is ported
        if method == "fuzzyFileSearch" and isinstance(params, dict):
            query = str(params.get("query", ""))
            roots = params.get("roots") or []
            if not isinstance(roots, list):
                roots = []
            files = run_fuzzy_file_search(query, [str(r) for r in roots])
            resp = FuzzyFileSearchResponse(files=files)
            # Convert dataclasses to dicts
            payload = {"files": [asdict(f) for f in resp.files]}
            await self._outgoing.send_response(rid, payload)
            return

        if method == "getUserAgent":
            ua = self._user_agent or self._build_user_agent(name=None, version=None)
            await self._outgoing.send_response(rid, {"userAgent": ua})
            return

        await self._codex.process_request(method, params, rid)

    async def process_notification(self, notification: JSONRPCNotification) -> None:
        # No notifications expected currently; could log if desired
        return

    async def process_response(self, response: JSONRPCResponse) -> None:
        await self._outgoing.notify_client_response(response.id, response.result)

    def process_error(self, err: Any) -> None:
        # Placeholder hook for logging
        return

    def _build_user_agent(self, *, name: Optional[str], version: Optional[str]) -> str:
        name = name or "jinx-app-server"
        version = version or "0.0.0"
        return f"{name}; {version}"
