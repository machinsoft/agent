from __future__ import annotations

import asyncio
import json
import sys
import contextlib
from typing import Any, Optional

from jinx.micro.app_server.message_processor import MessageProcessor
from jinx.micro.app_server.outgoing import OutgoingMessageSender
from jinx.micro.net.jsonrpc import (
    from_obj as jsonrpc_from_obj,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCNotification,
    JSONRPCError,
)


async def _stdin_lines() -> None:
    # Placeholder to hint type checkers
    return None


async def _read_stdin_lines(queue: asyncio.Queue[str]) -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            break
        await queue.put(line)


async def _writer_task(outgoing_queue: asyncio.Queue[dict]) -> None:
    while True:
        obj = await outgoing_queue.get()
        try:
            sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception:
            # Best-effort; continue
            pass


async def run_main(
    codex_linux_sandbox_exe: Optional[str] = None,
    cli_config_overrides: Optional[dict[str, Any]] = None,
) -> None:
    outgoing_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=128)
    sender = OutgoingMessageSender(outgoing_queue)
    processor = MessageProcessor(
        sender,
        codex_linux_sandbox_exe=codex_linux_sandbox_exe,
        config=cli_config_overrides,
        feedback=None,
    )

    stdin_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=128)

    writer = asyncio.create_task(_writer_task(outgoing_queue))
    reader = asyncio.create_task(_read_stdin_lines(stdin_queue))

    try:
        while True:
            try:
                line = await asyncio.wait_for(stdin_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if reader.done():
                    break
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            try:
                msg = jsonrpc_from_obj(obj)
            except Exception:
                # Unknown payload; ignore
                continue

            if isinstance(msg, JSONRPCRequest):
                await processor.process_request(msg)
            elif isinstance(msg, JSONRPCResponse):
                await processor.process_response(msg)
            elif isinstance(msg, JSONRPCNotification):
                await processor.process_notification(msg)
            elif isinstance(msg, JSONRPCError):
                processor.process_error(msg)
            else:
                # ignore
                pass
    finally:
        # Allow writer to drain; cancel after a brief delay
        await asyncio.sleep(0.05)
        for task in (reader, writer):
            if not task.done():
                task.cancel()
                with contextlib.suppress(Exception):
                    await task
