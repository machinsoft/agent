from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal


# --- Content items (snake_case to match wire format) ---
@dataclass
class ContentItemInputText:
    type: Literal["input_text"] = "input_text"
    text: str = ""


@dataclass
class ContentItemInputImage:
    type: Literal["input_image"] = "input_image"
    image_url: str = ""


@dataclass
class ContentItemOutputText:
    type: Literal["output_text"] = "output_text"
    text: str = ""


ContentItem = ContentItemInputText | ContentItemInputImage | ContentItemOutputText


# --- Response input items ---
@dataclass
class ResponseInputItemMessage:
    type: Literal["message"] = "message"
    role: str = "user"
    content: List[ContentItem] = field(default_factory=list)


@dataclass
class ResponseInputItemFunctionCallOutput:
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str = ""
    output: "FunctionCallOutputPayload" = field(default_factory=lambda: FunctionCallOutputPayload(content=""))


@dataclass
class ResponseInputItemMcpToolCallOutput:
    type: Literal["mcp_tool_call_output"] = "mcp_tool_call_output"
    call_id: str = ""
    result: Any = None  # Result<CallToolResult, String> analogue


@dataclass
class ResponseInputItemCustomToolCallOutput:
    type: Literal["custom_tool_call_output"] = "custom_tool_call_output"
    call_id: str = ""
    output: str = ""


ResponseInputItem = (
    ResponseInputItemMessage
    | ResponseInputItemFunctionCallOutput
    | ResponseInputItemMcpToolCallOutput
    | ResponseInputItemCustomToolCallOutput
)


# --- Local shell actions ---
LocalShellStatus = Literal["completed", "in_progress", "incomplete"]


@dataclass
class LocalShellExecAction:
    type: Literal["exec"] = "exec"
    command: List[str] = field(default_factory=list)
    timeout_ms: Optional[int] = None
    working_directory: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    user: Optional[str] = None


LocalShellAction = LocalShellExecAction


# --- Web search actions ---
@dataclass
class WebSearchActionSearch:
    type: Literal["search"] = "search"
    query: str = ""


WebSearchAction = WebSearchActionSearch


# --- Reasoning item details ---
@dataclass
class ReasoningItemReasoningSummary:
    type: Literal["summary_text"] = "summary_text"
    text: str = ""


@dataclass
class ReasoningItemContentReasoningText:
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str = ""


@dataclass
class ReasoningItemContentText:
    type: Literal["text"] = "text"
    text: str = ""


ReasoningItemContent = ReasoningItemContentReasoningText | ReasoningItemContentText


# --- Function call output payload ---
@dataclass
class FunctionCallOutputContentItemInputText:
    type: Literal["input_text"] = "input_text"
    text: str = ""


@dataclass
class FunctionCallOutputContentItemInputImage:
    type: Literal["input_image"] = "input_image"
    image_url: str = ""


FunctionCallOutputContentItem = (
    FunctionCallOutputContentItemInputText | FunctionCallOutputContentItemInputImage
)


@dataclass
class FunctionCallOutputPayload:
    content: str
    content_items: Optional[List[FunctionCallOutputContentItem]] = None
    success: Optional[bool] = None


# --- Response items (unified) ---
@dataclass
class ResponseItemMessage:
    type: Literal["message"] = "message"
    id: Optional[str] = None
    role: str = "assistant"
    content: List[ContentItem] = field(default_factory=list)


@dataclass
class ResponseItemReasoning:
    type: Literal["reasoning"] = "reasoning"
    id: str = ""
    summary: List[ReasoningItemReasoningSummary] = field(default_factory=list)
    content: Optional[List[ReasoningItemContent]] = None
    encrypted_content: Optional[str] = None


@dataclass
class ResponseItemLocalShellCall:
    type: Literal["local_shell_call"] = "local_shell_call"
    id: Optional[str] = None
    call_id: Optional[str] = None
    status: LocalShellStatus = "in_progress"
    action: LocalShellAction = field(default_factory=LocalShellExecAction)


@dataclass
class ResponseItemFunctionCall:
    type: Literal["function_call"] = "function_call"
    id: Optional[str] = None
    name: str = ""
    arguments: str = ""  # raw string JSON
    call_id: str = ""


@dataclass
class ResponseItemFunctionCallOutput:
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str = ""
    output: FunctionCallOutputPayload = field(default_factory=lambda: FunctionCallOutputPayload(content=""))


@dataclass
class ResponseItemCustomToolCall:
    type: Literal["custom_tool_call"] = "custom_tool_call"
    id: Optional[str] = None
    status: Optional[str] = None
    call_id: str = ""
    name: str = ""
    input: str = ""


@dataclass
class ResponseItemCustomToolCallOutput:
    type: Literal["custom_tool_call_output"] = "custom_tool_call_output"
    call_id: str = ""
    output: str = ""


@dataclass
class ResponseItemWebSearchCall:
    type: Literal["web_search_call"] = "web_search_call"
    id: Optional[str] = None
    status: Optional[str] = None
    action: WebSearchAction = field(default_factory=WebSearchActionSearch)


@dataclass
class ResponseItemGhostSnapshot:
    type: Literal["ghost_snapshot"] = "ghost_snapshot"
    ghost_commit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseItemCompactionSummary:
    type: Literal["compaction_summary"] = "compaction_summary"
    encrypted_content: str = ""


ResponseItem = (
    ResponseItemMessage
    | ResponseItemReasoning
    | ResponseItemLocalShellCall
    | ResponseItemFunctionCall
    | ResponseItemFunctionCallOutput
    | ResponseItemCustomToolCall
    | ResponseItemCustomToolCallOutput
    | ResponseItemWebSearchCall
    | ResponseItemGhostSnapshot
    | ResponseItemCompactionSummary
)
