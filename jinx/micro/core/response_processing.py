from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from jinx.micro.protocol.models import (
    FunctionCallOutputContentItem,
    FunctionCallOutputPayload,
    ResponseInputItem,
    ResponseInputItemCustomToolCallOutput,
    ResponseInputItemFunctionCallOutput,
    ResponseInputItemMcpToolCallOutput,
    ResponseItem,
    ResponseItemFunctionCallOutput,
    ResponseItemCustomToolCallOutput,
)


@dataclass
class ProcessedResponseItem:
    item: ResponseItem
    response: Optional[ResponseInputItem]


async def process_items(
    processed_items: List[ProcessedResponseItem],
    sess: Any = None,
    turn_context: Any = None,
) -> Tuple[List[ResponseInputItem], List[ResponseItem]]:
    """Transform streamed items into (responses_to_send, items_to_record).

    - Records original output items and any synthetic input items derived from responses.
    - Compatible with Jinx async runtime; session recording is optional.
    """
    outputs_to_record: List[ResponseItem] = []
    new_inputs_to_record: List[ResponseItem] = []
    responses: List[ResponseInputItem] = []

    for pri in processed_items:
        item, response = pri.item, pri.response
        if response is not None:
            responses.append(response)

        if isinstance(response, ResponseInputItemFunctionCallOutput):
            new_inputs_to_record.append(
                ResponseItemFunctionCallOutput(call_id=response.call_id, output=response.output)
            )
        elif isinstance(response, ResponseInputItemCustomToolCallOutput):
            new_inputs_to_record.append(
                ResponseItemCustomToolCallOutput(call_id=response.call_id, output=response.output)
            )
        elif isinstance(response, ResponseInputItemMcpToolCallOutput):
            result = response.result
            if isinstance(result, Exception):
                output = FunctionCallOutputPayload(content=str(result), success=False)
            else:
                # Best-effort: if a structured content-like object is present, turn into items
                content_items: Optional[List[FunctionCallOutputContentItem]] = None
                output_content = str(result)
                output = FunctionCallOutputPayload(
                    content=output_content,
                    content_items=content_items,
                    success=True,
                )
            new_inputs_to_record.append(
                ResponseItemFunctionCallOutput(call_id=response.call_id, output=output)
            )

        outputs_to_record.append(item)

    all_items_to_record = [*outputs_to_record, *new_inputs_to_record]

    # Optional session recording hook if provided
    if all_items_to_record and sess is not None and hasattr(sess, "record_conversation_items"):
        try:
            await sess.record_conversation_items(turn_context, all_items_to_record)
        except Exception:
            pass

    return responses, all_items_to_record
