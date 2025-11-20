from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

__all__ = [
    "ExecCommandSession",
    "SpawnedPty",
    "spawn_pty_process",
]


class _Broadcast:
    def __init__(self, max_receivers: int = 64, max_queue: int = 256) -> None:
        self._subs: list[asyncio.Queue[bytes]] = []
        self._max_receivers = max_receivers
        self._max_queue = max_queue
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue(self._max_queue)
        async with self._lock:
            if len(self._subs) < self._max_receivers:
                self._subs.append(q)
            else:
                # Drop oldest subscriber to keep memory bounded
                self._subs.pop(0)
                self._subs.append(q)
        return q

    async def send(self, data: bytes) -> None:
        # Non-blocking best-effort broadcast with drop-oldest policy
        async with self._lock:
            for q in list(self._subs):
                if q.full():
                    try:
                        _ = q.get_nowait()
                    except Exception:
                        pass
                try:
                    q.put_nowait(data)
                except Exception:
                    # Subscriber likely cancelled; remove lazily next GC
                    pass


@dataclass
class ExecCommandSession:
    _proc: asyncio.subprocess.Process
    _writer_q: asyncio.Queue[bytes]
    _broadcast: _Broadcast
    _reader_task: asyncio.Task[None]
    _writer_task: asyncio.Task[None]
    _wait_task: asyncio.Task[int]
    _exit_code: Optional[int] = None

    def writer_sender(self) -> asyncio.Queue[bytes]:
        return self._writer_q

    async def output_receiver(self) -> asyncio.Queue[bytes]:
        return await self._broadcast.subscribe()

    def has_exited(self) -> bool:
        return self._exit_code is not None

    def exit_code(self) -> Optional[int]:
        if self._exit_code is not None:
            return self._exit_code
        if self._proc.returncode is not None:
            return int(self._proc.returncode)
        return None

    async def kill(self) -> None:
        try:
            if os.name == "nt":
                self._proc.terminate()
            else:
                self._proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass

    async def close(self) -> None:
        for t in (self._reader_task, self._writer_task):
            if not t.done():
                t.cancel()
        try:
            await asyncio.gather(self._reader_task, self._writer_task, return_exceptions=True)
        finally:
            if self._proc.returncode is None:
                await self.kill()


@dataclass
class SpawnedPty:
    session: ExecCommandSession
    output_rx: asyncio.Queue[bytes]
    exit_rx: asyncio.Future[int]


async def spawn_pty_process(
    program: str,
    args: list[str],
    cwd: str | Path,
    env: Dict[str, str],
    arg0: Optional[str] = None,
) -> SpawnedPty:
    if not program:
        raise ValueError("missing program for PTY spawn")

    cmd = [arg0 or program, *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env or None,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        limit=8192,
    )

    writer_q: asyncio.Queue[bytes] = asyncio.Queue(128)
    broadcast = _Broadcast(max_receivers=64, max_queue=256)

    async def _reader() -> None:
        assert proc.stdout is not None
        while True:
            try:
                chunk = await proc.stdout.read(8192)
            except Exception:
                break
            if not chunk:
                break
            await broadcast.send(chunk)

    async def _writer() -> None:
        assert proc.stdin is not None
        while True:
            try:
                data = await writer_q.get()
            except asyncio.CancelledError:
                break
            try:
                proc.stdin.write(data)
                await proc.stdin.drain()
            except Exception:
                break

    async def _waiter() -> int:
        rc = await proc.wait()
        return int(rc)

    reader_task = asyncio.create_task(_reader())
    writer_task = asyncio.create_task(_writer())
    wait_task: asyncio.Task[int] = asyncio.create_task(_waiter())

    # Track exit code
    def _set_exit(task: asyncio.Task[int]) -> None:
        try:
            code = task.result()
        except Exception:
            code = -1
        session._exit_code = code  # type: ignore[name-defined]

    session = ExecCommandSession(
        _proc=proc,
        _writer_q=writer_q,
        _broadcast=broadcast,
        _reader_task=reader_task,
        _writer_task=writer_task,
        _wait_task=wait_task,
    )
    wait_task.add_done_callback(_set_exit)

    initial_rx = await session.output_receiver()
    exit_future: asyncio.Future[int] = asyncio.ensure_future(wait_task)

    return SpawnedPty(session=session, output_rx=initial_rx, exit_rx=exit_future)
