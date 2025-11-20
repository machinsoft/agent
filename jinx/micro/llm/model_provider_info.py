from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal
import os

from jinx.micro.protocol.common import AuthMode

WireApi = Literal["responses", "chat"]


@dataclass
class ModelProviderInfo:
    name: str
    base_url: Optional[str] = None
    env_key: Optional[str] = None
    env_key_instructions: Optional[str] = None
    experimental_bearer_token: Optional[str] = None
    wire_api: WireApi = "chat"
    query_params: Optional[Dict[str, str]] = None
    http_headers: Optional[Dict[str, str]] = None
    env_http_headers: Optional[Dict[str, str]] = None
    request_max_retries: Optional[int] = None
    stream_max_retries: Optional[int] = None
    stream_idle_timeout_ms: Optional[int] = None
    requires_openai_auth: bool = False

    # ---- helpers ----
    def _query_string(self) -> str:
        if not self.query_params:
            return ""
        parts = [f"{k}={v}" for k, v in self.query_params.items()]
        return "?" + "&".join(parts)

    def get_full_url(self, auth_mode: Optional[AuthMode]) -> str:
        default_base = (
            "https://chatgpt.com/backend-api/codex"
            if auth_mode == AuthMode.chatgpt
            else "https://api.openai.com/v1"
        )
        base = self.base_url or default_base
        qs = self._query_string()
        if self.wire_api == "responses":
            return f"{base}/responses{qs}"
        return f"{base}/chat/completions{qs}"

    def get_compact_url(self, auth_mode: Optional[AuthMode]) -> Optional[str]:
        if self.wire_api != "responses":
            return None
        full = self.get_full_url(auth_mode)
        if "?" in full:
            path, qs = full.split("?", 1)
            return f"{path}/compact?{qs}"
        return f"{full}/compact"

    def is_azure_responses_endpoint(self) -> bool:
        if self.wire_api != "responses":
            return False
        if self.name.lower() == "azure":
            return True
        base = (self.base_url or "").lower()
        markers = [
            "openai.azure.",
            "cognitiveservices.azure.",
            "aoai.azure.",
            "azure-api.",
            "azurefd.",
        ]
        return any(m in base for m in markers)


DEFAULT_LMSTUDIO_PORT = 1234
DEFAULT_OLLAMA_PORT = 11434
LMSTUDIO_OSS_PROVIDER_ID = "lmstudio"
OLLAMA_OSS_PROVIDER_ID = "ollama"


def create_oss_provider_with_base_url(base_url: str, wire_api: WireApi) -> ModelProviderInfo:
    return ModelProviderInfo(
        name="gpt-oss",
        base_url=base_url,
        wire_api=wire_api,
        requires_openai_auth=False,
    )


def create_oss_provider(default_port: int, wire_api: WireApi) -> ModelProviderInfo:
    base = f"http://localhost:{default_port}/v1"
    return create_oss_provider_with_base_url(base, wire_api)


def built_in_model_providers(auth_mode: Optional[AuthMode]) -> Dict[str, ModelProviderInfo]:
    openai_entry = ModelProviderInfo(
        name="OpenAI",
        base_url=None,
        wire_api="responses",
        http_headers={"version": "0.0.0"},
        requires_openai_auth=True,
    )

    providers: Dict[str, ModelProviderInfo] = {
        "openai": openai_entry,
        OLLAMA_OSS_PROVIDER_ID: create_oss_provider(DEFAULT_OLLAMA_PORT, "chat"),
        LMSTUDIO_OSS_PROVIDER_ID: create_oss_provider(DEFAULT_LMSTUDIO_PORT, "responses"),
    }
    return providers
