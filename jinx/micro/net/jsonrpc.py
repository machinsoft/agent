from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

__all__ = [
    "JSONRPC_VERSION",
    "RequestId",
    "JSONRPCMessage",
    "JSONRPCRequest",
    "JSONRPCNotification",
    "JSONRPCResponse",
    "JSONRPCError",
    "JSONRPCErrorError",
    "from_obj",
]

JSONRPC_VERSION: str = "2.0"
RequestId = Union[str, int]


@dataclass
class JSONRPCRequest:
    id: RequestId
    method: str
    params: Optional[Any] = None

    def to_obj(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"id": self.id, "method": self.method}
        if self.params is not None:
            obj["params"] = self.params
        return obj


@dataclass
class JSONRPCNotification:
    method: str
    params: Optional[Any] = None

    def to_obj(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"method": self.method}
        if self.params is not None:
            obj["params"] = self.params
        return obj


@dataclass
class JSONRPCResponse:
    id: RequestId
    result: Any

    def to_obj(self) -> Dict[str, Any]:
        return {"id": self.id, "result": self.result}


@dataclass
class JSONRPCErrorError:
    code: int
    message: str
    data: Optional[Any] = None

    def to_obj(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            obj["data"] = self.data
        return obj


@dataclass
class JSONRPCError:
    error: JSONRPCErrorError
    id: RequestId

    def to_obj(self) -> Dict[str, Any]:
        return {"error": self.error.to_obj(), "id": self.id}


JSONRPCMessage = Union[JSONRPCRequest, JSONRPCNotification, JSONRPCResponse, JSONRPCError]


def from_obj(obj: Dict[str, Any]) -> JSONRPCMessage:
    if "error" in obj and "id" in obj:
        err = obj["error"] or {}
        return JSONRPCError(
            error=JSONRPCErrorError(
                code=int(err.get("code", 0)),
                message=str(err.get("message", "")),
                data=err.get("data"),
            ),
            id=obj["id"],
        )
    if "result" in obj and "id" in obj:
        return JSONRPCResponse(id=obj["id"], result=obj.get("result"))
    if "id" in obj and "method" in obj:
        return JSONRPCRequest(id=obj["id"], method=str(obj["method"]), params=obj.get("params"))
    if "method" in obj:
        return JSONRPCNotification(method=str(obj["method"]), params=obj.get("params"))
    raise ValueError("Invalid JSON-RPC object")
