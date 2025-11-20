"""Async backend client for Codex Cloud Tasks API.

Rewrites the Rust backend-client in Python, aligned with Jinx micro-architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse
import importlib

from jinx.bootstrap import ensure_optional, package

from .types import (
    CodeTaskDetailsResponse,
    TurnAttemptsSiblingTurnsResponse,
    RateLimitSnapshot,
    RateLimitWindow,
    CreditsSnapshot,
)


@dataclass(frozen=True)
class PathStyle:
    CodexApi: str = "CodexApi"
    ChatGptApi: str = "ChatGptApi"

    @staticmethod
    def from_base_url(base_url: str) -> str:
        return PathStyle.ChatGptApi if "/backend-api" in base_url else PathStyle.CodexApi


class Client:
    def __init__(self, base_url: str):
        base_url = base_url.rstrip('/')
        if (
            (base_url.startswith("https://chatgpt.com") or base_url.startswith("https://chat.openai.com"))
            and "/backend-api" not in base_url
        ):
            base_url = f"{base_url}/backend-api"
        self._base_url = base_url
        self._path_style = PathStyle.from_base_url(base_url)
        self._bearer_token: Optional[str] = None
        self._user_agent: Optional[str] = None
        self._chatgpt_account_id: Optional[str] = None
        self._http = self._make_http_client()

    @staticmethod
    def _ensure_httpx():
        try:
            return importlib.import_module("httpx")
        except Exception:
            package("httpx")
            return importlib.import_module("httpx")

    @staticmethod
    def _pick_proxy_env() -> Optional[str]:
        import os
        for key in ("PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            val = os.getenv(key)
            if val:
                return val
        return None

    def _make_http_client(self):
        httpx = self._ensure_httpx()
        proxy = self._pick_proxy_env()
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0)
        if proxy:
            try:
                scheme = (urlparse(proxy).scheme or "").lower()
                if scheme.startswith("socks"):
                    try:
                        socks = importlib.import_module("httpx_socks")
                    except ImportError:
                        package("httpx-socks")
                        socks = importlib.import_module("httpx_socks")
                    transport = socks.AsyncProxyTransport.from_url(proxy)
                    return httpx.AsyncClient(timeout=timeout, limits=limits, transport=transport)
                return httpx.AsyncClient(timeout=timeout, limits=limits, proxies=proxy)
            except Exception:
                # Fallback to direct client
                return httpx.AsyncClient(timeout=timeout, limits=limits)
        return httpx.AsyncClient(timeout=timeout, limits=limits)

    def with_bearer_token(self, token: str) -> "Client":
        self._bearer_token = token
        return self

    def with_user_agent(self, ua: str) -> "Client":
        self._user_agent = ua
        return self

    def with_chatgpt_account_id(self, account_id: str) -> "Client":
        self._chatgpt_account_id = account_id
        return self

    def with_path_style(self, style: str) -> "Client":
        self._path_style = style
        return self

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        h["User-Agent"] = self._user_agent or "codex-cli"
        if self._bearer_token:
            h["Authorization"] = f"Bearer {self._bearer_token}"
        if self._chatgpt_account_id:
            h["ChatGPT-Account-Id"] = self._chatgpt_account_id
        return h

    async def _exec_request(self, req, method: str, url: str) -> Tuple[str, str]:
        res = await req
        ct = res.headers.get("content-type", "")
        try:
            body_bytes = await res.aread()
        except Exception:
            # Fallback: some transports may already buffer content
            body_bytes = getattr(res, "content", b"") or b""
        try:
            text = body_bytes.decode("utf-8", "replace")
        except Exception:
            text = ""
        if res.is_error:
            raise RuntimeError(f"{method} {url} failed: {res.status_code}; content-type={ct}; body={text}")
        return text, ct

    @staticmethod
    def _decode_json(httpx_mod, url: str, ct: str, body: str) -> Any:
        try:
            return httpx_mod.Response(200, text=body).json()
        except Exception as e:
            raise RuntimeError(f"Decode error for {url}: {e}; content-type={ct}; body={body}")

    # API methods
    async def get_rate_limits(self) -> RateLimitSnapshot:
        httpx = self._ensure_httpx()
        if self._path_style == PathStyle.CodexApi:
            url = f"{self._base_url}/api/codex/usage"
        else:
            url = f"{self._base_url}/wham/usage"
        req = self._http.get(url, headers=self._headers())
        body, ct = await self._exec_request(req, "GET", url)
        payload = self._decode_json(httpx, url, ct, body)
        return self._rate_limit_snapshot_from_payload(payload)

    async def list_tasks(
        self,
        limit: Optional[int] = None,
        task_filter: Optional[str] = None,
        environment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        httpx = self._ensure_httpx()
        if self._path_style == PathStyle.CodexApi:
            url = f"{self._base_url}/api/codex/tasks/list"
        else:
            url = f"{self._base_url}/wham/tasks/list"
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = int(limit)
        if task_filter is not None:
            params["task_filter"] = task_filter
        if environment_id is not None:
            params["environment_id"] = environment_id
        req = self._http.get(url, headers=self._headers(), params=params or None)
        body, ct = await self._exec_request(req, "GET", url)
        return self._decode_json(httpx, url, ct, body)

    async def get_task_details(self, task_id: str) -> CodeTaskDetailsResponse:
        parsed, _, _ = await self.get_task_details_with_body(task_id)
        return parsed

    async def get_task_details_with_body(self, task_id: str) -> Tuple[CodeTaskDetailsResponse, str, str]:
        httpx = self._ensure_httpx()
        if self._path_style == PathStyle.CodexApi:
            url = f"{self._base_url}/api/codex/tasks/{task_id}"
        else:
            url = f"{self._base_url}/wham/tasks/{task_id}"
        req = self._http.get(url, headers=self._headers())
        body, ct = await self._exec_request(req, "GET", url)
        data = self._decode_json(httpx, url, ct, body)
        parsed = CodeTaskDetailsResponse.from_dict(data)
        return parsed, body, ct

    async def list_sibling_turns(self, task_id: str, turn_id: str) -> TurnAttemptsSiblingTurnsResponse:
        httpx = self._ensure_httpx()
        if self._path_style == PathStyle.CodexApi:
            url = f"{self._base_url}/api/codex/tasks/{task_id}/turns/{turn_id}/sibling_turns"
        else:
            url = f"{self._base_url}/wham/tasks/{task_id}/turns/{turn_id}/sibling_turns"
        req = self._http.get(url, headers=self._headers())
        body, ct = await self._exec_request(req, "GET", url)
        data = self._decode_json(httpx, url, ct, body)
        return TurnAttemptsSiblingTurnsResponse.from_dict(data)

    async def create_task(self, request_body: Dict[str, Any]) -> str:
        httpx = self._ensure_httpx()
        if self._path_style == PathStyle.CodexApi:
            url = f"{self._base_url}/api/codex/tasks"
        else:
            url = f"{self._base_url}/wham/tasks"
        req = self._http.post(url, headers={**self._headers(), "content-type": "application/json"}, json=request_body)
        body, ct = await self._exec_request(req, "POST", url)
        data = self._decode_json(httpx, url, ct, body)
        # prefer nested task.id
        task = data.get("task") if isinstance(data, dict) else None
        if isinstance(task, dict) and isinstance(task.get("id"), str):
            return task["id"]
        if isinstance(data, dict) and isinstance(data.get("id"), str):
            return data["id"]
        raise RuntimeError(f"POST {url} succeeded but no task id found; content-type={ct}; body={body}")

    # Mapping helpers
    @staticmethod
    def _rate_limit_snapshot_from_payload(payload: Dict[str, Any]) -> RateLimitSnapshot:
        rl = payload.get("rate_limit") if isinstance(payload, dict) else None
        details = None
        if isinstance(rl, dict):
            details = rl
        elif isinstance(rl, (list, tuple)) and rl:
            details = rl[0]
        # windows
        primary = Client._map_rate_limit_window(details.get("primary_window") if isinstance(details, dict) else None)
        secondary = Client._map_rate_limit_window(details.get("secondary_window") if isinstance(details, dict) else None)
        credits = Client._map_credits(payload.get("credits") if isinstance(payload, dict) else None)
        return RateLimitSnapshot(primary=primary, secondary=secondary, credits=credits)

    @staticmethod
    def _map_rate_limit_window(snapshot: Any) -> Optional[RateLimitWindow]:
        if not isinstance(snapshot, dict):
            return None
        used_percent = snapshot.get("used_percent")
        try:
            used_percent = float(used_percent) if used_percent is not None else None
        except Exception:
            used_percent = None
        seconds = snapshot.get("limit_window_seconds")
        try:
            seconds_i = int(seconds) if seconds is not None else None
        except Exception:
            seconds_i = None
        window_minutes = ((seconds_i + 59) // 60) if isinstance(seconds_i, int) and seconds_i > 0 else None
        resets_at = snapshot.get("reset_at")
        try:
            resets_at = int(resets_at) if resets_at is not None else None
        except Exception:
            resets_at = None
        return RateLimitWindow(used_percent=used_percent, window_minutes=window_minutes, resets_at=resets_at)

    @staticmethod
    def _map_credits(details: Any) -> Optional[CreditsSnapshot]:
        if not isinstance(details, dict):
            return None
        has = details.get("has_credits")
        unlim = details.get("unlimited")
        bal = details.get("balance")
        try:
            bal = float(bal) if bal is not None else None
        except Exception:
            bal = None
        return CreditsSnapshot(has_credits=bool(has) if has is not None else None, unlimited=bool(unlim) if unlim is not None else None, balance=bal)

    async def aclose(self) -> None:
        try:
            await self._http.aclose()
        except Exception:
            pass

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
