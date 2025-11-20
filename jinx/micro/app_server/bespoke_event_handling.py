from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from jinx.micro.app_server.outgoing import OutgoingMessageSender


def _send_notification(sender: OutgoingMessageSender, method: str, params: Dict[str, Any]) -> None:
    # Fire and forget helper (compat with async)
    import asyncio

    async def _run() -> None:
        await sender.send_server_notification(method, params)

    asyncio.create_task(_run())


def _turn_completed(sender: OutgoingMessageSender, event_id: str, status: str, error_msg: Optional[str] = None) -> None:
    turn: Dict[str, Any]
    if status == "failed" and error_msg:
        turn = {"id": event_id, "items": [], "status": "failed", "error": {"message": error_msg}}
    else:
        turn = {"id": event_id, "items": [], "status": status}
    _send_notification(sender, "turn/completed", {"turn": turn})


def apply_bespoke_event_handling(
    event: Dict[str, Any],
    conversation_id: str,
    outgoing: OutgoingMessageSender,
) -> None:
    """Translate core events into server notifications (v2 namespace).

    Expects events as dicts with at least a 'type' key and event-specific fields.
    This is a minimal, best-effort port and can be extended incrementally.
    """
    et = str(event.get("type", ""))

    # Token usage -> Account rate limits updated
    if et == "TokenCount":
        rl = event.get("rate_limits")
        if isinstance(rl, dict):
            _send_notification(outgoing, "account/rateLimits/updated", {"rateLimits": rl})
        return

    # Agent message delta
    if et == "AgentMessageContentDelta":
        _send_notification(
            outgoing,
            "item/agentMessage/delta",
            {"itemId": event.get("item_id"), "delta": event.get("delta", "")},
        )
        return

    # Reasoning summary text deltas
    if et == "ReasoningContentDelta":
        _send_notification(
            outgoing,
            "item/reasoning/summaryTextDelta",
            {
                "itemId": event.get("item_id"),
                "delta": event.get("delta", ""),
                "summaryIndex": int(event.get("summary_index", 0)),
            },
        )
        return

    if et == "ReasoningRawContentDelta":
        _send_notification(
            outgoing,
            "item/reasoning/textDelta",
            {
                "itemId": event.get("item_id"),
                "delta": event.get("delta", ""),
                "contentIndex": int(event.get("content_index", 0)),
            },
        )
        return

    if et == "AgentReasoningSectionBreak":
        _send_notification(
            outgoing,
            "item/reasoning/summaryPartAdded",
            {"itemId": event.get("item_id"), "summaryIndex": int(event.get("summary_index", 0))},
        )
        return

    # Patch apply lifecycle -> FileChange item start/complete
    if et == "PatchApplyBegin":
        item_id = str(event.get("call_id", ""))
        changes = event.get("changes") or []
        _send_notification(
            outgoing,
            "item/started",
            {"item": {"type": "fileChange", "id": item_id, "changes": changes, "status": "inProgress"}},
        )
        return

    if et == "PatchApplyEnd":
        item_id = str(event.get("call_id", ""))
        changes = event.get("changes") or []
        success = bool(event.get("success", False))
        status = "completed" if success else "failed"
        _send_notification(
            outgoing,
            "item/completed",
            {"item": {"type": "fileChange", "id": item_id, "changes": changes, "status": status}},
        )
        return

    # Command execution lifecycle
    if et == "ExecCommandBegin":
        _send_notification(
            outgoing,
            "item/started",
            {
                "item": {
                    "type": "commandExecution",
                    "id": str(event.get("call_id", "")),
                    "command": str(event.get("command", "")),
                    "cwd": str(event.get("cwd", "")),
                    "status": "inProgress",
                    "commandActions": event.get("parsed_cmd") or [],
                    "aggregatedOutput": None,
                    "exitCode": None,
                    "durationMs": None,
                }
            },
        )
        return

    if et == "ExecCommandOutputDelta":
        _send_notification(
            outgoing,
            "item/commandExecution/outputDelta",
            {"itemId": str(event.get("call_id", "")), "delta": str(event.get("chunk", b""), "utf-8", "ignore")},
        )
        return

    if et == "ExecCommandEnd":
        exit_code = int(event.get("exit_code", -1))
        status = "completed" if exit_code == 0 else "failed"
        _send_notification(
            outgoing,
            "item/completed",
            {
                "item": {
                    "type": "commandExecution",
                    "id": str(event.get("call_id", "")),
                    "command": str(event.get("command", "")),
                    "cwd": str(event.get("cwd", "")),
                    "status": status,
                    "commandActions": event.get("parsed_cmd") or [],
                    "aggregatedOutput": event.get("aggregated_output") or None,
                    "exitCode": exit_code,
                    "durationMs": int(event.get("duration_ms", 0)) if event.get("duration_ms") is not None else None,
                }
            },
        )
        return

    # Turn completion/interrupt/error mapping
    if et == "TurnCompleted":
        _turn_completed(outgoing, str(event.get("event_id", "")), "completed")
        return

    if et == "TurnInterrupted":
        _turn_completed(outgoing, str(event.get("event_id", "")), "interrupted")
        return

    if et == "Error":
        _turn_completed(outgoing, str(event.get("event_id", "")), "failed", error_msg=str(event.get("message", "")))
        return
