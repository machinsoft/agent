from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .v2 import (
    Turn,
    TurnStatusCompleted,
    TurnStatusInterrupted,
    UserInput,
    make_user_message_item,
    make_agent_message_item,
    make_reasoning_item,
    ThreadItem,
)


@dataclass
class _PendingTurn:
    id: str
    items: List[ThreadItem]
    status: object


class _ThreadHistoryBuilder:
    def __init__(self) -> None:
        self.turns: List[Turn] = []
        self.current: Optional[_PendingTurn] = None
        self._next_turn_index = 1
        self._next_item_index = 1

    def finish(self) -> List[Turn]:
        self._finish_current_turn()
        return self.turns

    def handle_event(self, event: Dict[str, Any]) -> None:
        etype = str(event.get("type", ""))
        if etype == "UserMessage":
            self._handle_user_message(event)
        elif etype == "AgentMessage":
            self._handle_agent_message(event)
        elif etype == "AgentReasoning":
            self._handle_agent_reasoning(event)
        elif etype == "AgentReasoningRawContent":
            self._handle_agent_reasoning_raw(event)
        elif etype == "TurnAborted":
            self._handle_turn_aborted(event)
        else:
            # Ignore unknown events for forward-compatibility
            pass

    # --- handlers ---
    def _handle_user_message(self, payload: Dict[str, Any]) -> None:
        self._finish_current_turn()
        turn = self._new_turn()
        iid = self._next_item_id()
        content: List[UserInput] = []
        message = str(payload.get("message", ""))
        if message.strip():
            content.append(UserInput.text_msg(message))
        images = payload.get("images")
        if isinstance(images, list):
            for img in images:
                try:
                    url = str(img)
                except Exception:
                    continue
                if url:
                    content.append(UserInput.image_url(url))
        turn.items.append(make_user_message_item(iid, content))
        self.current = turn

    def _handle_agent_message(self, payload: Dict[str, Any]) -> None:
        text = str(payload.get("message") or payload.get("text") or "")
        if not text:
            return
        iid = self._next_item_id()
        self._ensure_turn().items.append(make_agent_message_item(iid, text))

    def _handle_agent_reasoning(self, payload: Dict[str, Any]) -> None:
        text = str(payload.get("text") or "").strip()
        if not text:
            return
        turn = self._ensure_turn()
        if turn.items and isinstance(turn.items[-1], type(make_reasoning_item("_", [], []))):
            # last item is reasoning -> append to summary
            reasoning = turn.items[-1]  # type: ignore[assignment]
            try:
                reasoning.summary.append(text)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        iid = self._next_item_id()
        turn.items.append(make_reasoning_item(iid, summary=[text]))

    def _handle_agent_reasoning_raw(self, payload: Dict[str, Any]) -> None:
        text = str(payload.get("text") or "").strip()
        if not text:
            return
        turn = self._ensure_turn()
        if turn.items and isinstance(turn.items[-1], type(make_reasoning_item("_", [], []))):
            reasoning = turn.items[-1]
            try:
                reasoning.content.append(text)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        iid = self._next_item_id()
        turn.items.append(make_reasoning_item(iid, content=[text]))

    def _handle_turn_aborted(self, _payload: Dict[str, Any]) -> None:
        if self.current is None:
            return
        self.current.status = TurnStatusInterrupted()

    # --- utilities ---
    def _finish_current_turn(self) -> None:
        if self.current is None:
            return
        if not self.current.items:
            self.current = None
            return
        turn = Turn(id=self.current.id, items=self.current.items, status=self.current.status)
        self.turns.append(turn)
        self.current = None

    def _new_turn(self) -> _PendingTurn:
        return _PendingTurn(id=self._next_turn_id(), items=[], status=TurnStatusCompleted())

    def _ensure_turn(self) -> _PendingTurn:
        if self.current is None:
            self.current = self._new_turn()
        return self.current

    def _next_turn_id(self) -> str:
        tid = f"turn-{self._next_turn_index}"
        self._next_turn_index += 1
        return tid

    def _next_item_id(self) -> str:
        iid = f"item-{self._next_item_index}"
        self._next_item_index += 1
        return iid


def build_turns_from_event_msgs(events: List[Dict[str, Any]]) -> List[Turn]:
    builder = _ThreadHistoryBuilder()
    for ev in events:
        if isinstance(ev, dict):
            builder.handle_event(ev)
    return builder.finish()
