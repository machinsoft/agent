from __future__ import annotations

# Re-export directly from the local micro-module to avoid circular imports
# when the top-level facade `jinx.net` imports from `jinx.micro.net.client`.
from .client import get_openai_client
from .jsonrpc import (
    JSONRPC_VERSION,
    JSONRPCRequest,
    JSONRPCNotification,
    JSONRPCResponse,
    JSONRPCError,
    JSONRPCErrorError,
    RequestId,
    from_obj,
)

__all__ = [
    "get_openai_client",
    "JSONRPC_VERSION",
    "JSONRPCRequest",
    "JSONRPCNotification",
    "JSONRPCResponse",
    "JSONRPCError",
    "JSONRPCErrorError",
    "RequestId",
    "from_obj",
]
