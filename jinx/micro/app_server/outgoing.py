from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from jinx.micro.net.jsonrpc import JSONRPCErrorError
from .error_codes import INTERNAL_ERROR_CODE

SendCallable = Callable[[Dict[str, Any]], Awaitable[None]]


class OutgoingMessageSender:
    """
    Async sender for JSON-RPC messages with request/response correlation.

    - Uses an asyncio.Queue or async callable to emit JSON-RPC messages
    - Correlates responses to requests via a future map keyed by id
    """

    def __init__(self, sender: Union[asyncio.Queue, SendCallable]) -> None:
        self._next_request_id: int = 0
        self._lock = asyncio.Lock()
        self._request_futures: Dict[Union[int, str], asyncio.Future] = {}
        self._queue: Optional[asyncio.Queue] = sender if isinstance(sender, asyncio.Queue) else None
        self._send_cb: Optional[SendCallable] = None if self._queue is not None else sender  # type: ignore[assignment]

    async def _send(self, obj: Dict[str, Any]) -> None:
        if self._queue is not None:
            await self._queue.put(obj)
        elif self._send_cb is not None:
            await self._send_cb(obj)
        else:
            # No-op fallback
            pass

    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> asyncio.Future:
        async with self._lock:
            rid = self._next_request_id
            self._next_request_id += 1
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._request_futures[rid] = fut
        req: Dict[str, Any] = {"id": rid, "method": method}
        if params is not None:
            req["params"] = params
        await self._send(req)
        return fut

    async def notify_client_response(self, id_value: Union[int, str], result: Any) -> None:
        fut = self._request_futures.pop(id_value, None)
        if fut and not fut.done():
            fut.set_result(result)

    async def send_response(self, id_value: Union[int, str], response: Any) -> None:
        obj = {"id": id_value, "result": response}
        await self._send(obj)

    async def send_server_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        obj: Dict[str, Any] = {"method": method}
        if params is not None:
            obj["params"] = params
        await self._send(obj)

    async def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        # Compatibility alias for legacy notifications
        await self.send_server_notification(method, params)

    async def send_error(
        self,
        id_value: Union[int, str],
        error: Optional[JSONRPCErrorError] = None,
    ) -> None:
        err_obj: Dict[str, Any]
        if error is None:
            err_obj = {
                "code": INTERNAL_ERROR_CODE,
                "message": "internal error",
            }
        else:
            err_obj = {
                "code": int(error.code),
                "message": str(error.message),
            }
            if getattr(error, "data", None) is not None:
                err_obj["data"] = error.data
        obj = {"id": id_value, "error": err_obj}
        await self._send(obj)
