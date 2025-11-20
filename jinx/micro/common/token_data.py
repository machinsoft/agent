from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "TokenData",
    "IdTokenInfo",
    "parse_id_token",
]


def _b64url_no_pad_decode(s: str) -> bytes:
    s = s.strip()
    # Add padding for urlsafe decode
    rem = len(s) % 4
    if rem:
        s += "=" * (4 - rem)
    return base64.urlsafe_b64decode(s.encode("ascii"))


@dataclass(frozen=True)
class IdTokenInfo:
    email: Optional[str]
    chatgpt_plan_type: Optional[str]
    chatgpt_account_id: Optional[str]
    raw_jwt: str

    def get_chatgpt_plan_type(self) -> Optional[str]:
        return self.chatgpt_plan_type


@dataclass(frozen=True)
class TokenData:
    id_token: IdTokenInfo
    access_token: str
    refresh_token: str
    account_id: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id_token": self.id_token.raw_jwt,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "account_id": self.account_id,
        }


_KNOWN_PLAN_PRETTY = {
    "free": "Free",
    "plus": "Plus",
    "pro": "Pro",
    "team": "Team",
    "business": "Business",
    "enterprise": "Enterprise",
    "edu": "Edu",
}


def parse_id_token(id_token: str) -> IdTokenInfo:
    parts = (id_token or "").split(".")
    if len(parts) != 3 or not all(parts):
        raise ValueError("invalid ID token format")
    _header_b64, payload_b64, _sig_b64 = parts
    try:
        payload_bytes = _b64url_no_pad_decode(payload_b64)
        claims = json.loads(payload_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"invalid ID token payload: {e}")

    email = claims.get("email")
    auth = claims.get("https://api.openai.com/auth") or {}
    plan = auth.get("chatgpt_plan_type")
    if isinstance(plan, str):
        pretty = _KNOWN_PLAN_PRETTY.get(plan.lower()) or plan
    elif isinstance(plan, dict):
        # Untagged enum style (unlikely in JSON) — best effort
        key = next(iter(plan.keys()), None)
        pretty = _KNOWN_PLAN_PRETTY.get(str(key).lower()) if key else None
    else:
        pretty = None

    chatgpt_account_id = auth.get("chatgpt_account_id")

    return IdTokenInfo(
        email=email if isinstance(email, str) else None,
        chatgpt_plan_type=pretty,
        chatgpt_account_id=chatgpt_account_id if isinstance(chatgpt_account_id, str) else None,
        raw_jwt=id_token,
    )
