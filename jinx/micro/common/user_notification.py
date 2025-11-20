from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = [
    "UserNotifier",
    "UserNotification",
]


@dataclass(frozen=True)
class UserNotification:
    type: str
    thread_id: str
    turn_id: str
    cwd: str
    input_messages: List[str]
    last_assistant_message: Optional[str] = None

    def to_json(self) -> str:
        # Serialize with kebab-case keys to mirror Codex payload
        payload = {
            "type": self.type,
            "thread-id": self.thread_id,
            "turn-id": self.turn_id,
            "cwd": self.cwd,
            "input-messages": self.input_messages,
        }
        if self.last_assistant_message is not None:
            payload["last-assistant-message"] = self.last_assistant_message
        return json.dumps(payload, ensure_ascii=False)


class UserNotifier:
    def __init__(self, notify_command: Optional[Sequence[str]] = None) -> None:
        self._cmd = list(notify_command) if notify_command else None

    def notify(self, notification: UserNotification) -> None:
        if not self._cmd:
            return
        try:
            argv = self._cmd[:]
            argv.append(notification.to_json())
            # Fire-and-forget; do not wait
            subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            # best-effort only
            pass
