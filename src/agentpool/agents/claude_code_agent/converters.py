"""Claude Agent SDK to native event converters.

This module provides conversion from Claude Agent SDK message types to native
agentpool streaming events, enabling ClaudeCodeAgent to yield the same
event types as native agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai import PartDeltaEvent
from pydantic_ai.messages import TextPartDelta, ThinkingPartDelta

from agentpool.agents.events import (
    DiffContentItem,
    LocationContentItem,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
)


if TYPE_CHECKING:
    from claude_agent_sdk import ContentBlock, Message, ToolUseBlock

    from agentpool.agents.events import RichAgentStreamEvent, ToolCallContentItem
    from agentpool.tools.base import ToolKind


@dataclass
class RichToolInfo:
    """Rich display information derived from tool name and input."""

    title: str
    """Human-readable title for the tool call."""
    kind: ToolKind = "other"
    """Category of tool operation."""
    locations: list[LocationContentItem] = field(default_factory=list)
    """File locations involved in the operation."""
    content: list[ToolCallContentItem] = field(default_factory=list)
    """Rich content items (diffs, text, etc.)."""


def derive_rich_tool_info(name: str, input_data: dict[str, Any]) -> RichToolInfo:  # noqa: PLR0911
    """Derive rich display info from tool name and input arguments.

    Maps MCP tool names and their inputs to human-readable titles, kinds,
    and location information for rich UI display. Handles both Claude Code
    built-in tools and MCP bridge tools.

    Args:
        name: The tool name (e.g., "Read", "mcp__server__read_file")
        input_data: The tool input arguments

    Returns:
        RichToolInfo with derived display information
    """
    # Extract the actual tool name if it's an MCP bridge tool
    # Format: mcp__{server_name}__{tool_name}
    actual_name = name
    if name.startswith("mcp__") and "__" in name[5:]:
        parts = name.split("__")
        if len(parts) >= 3:  # noqa: PLR2004
            actual_name = parts[-1]  # Get the last part (actual tool name)

    # Normalize to lowercase for matching
    tool_lower = actual_name.lower()

    # Read operations
    if tool_lower in ("read", "read_file"):
        path = input_data.get("file_path") or input_data.get("path", "")
        offset = input_data.get("offset") or input_data.get("line")
        limit = input_data.get("limit")

        suffix = ""
        if limit:
            start = (offset or 0) + 1
            end = (offset or 0) + limit
            suffix = f" ({start}-{end})"
        elif offset:
            suffix = f" (from line {offset + 1})"

        return RichToolInfo(
            title=f"Read {path}{suffix}" if path else "Read File",
            kind="read",
            locations=[LocationContentItem(path=path, line=offset or 0)] if path else [],
        )

    # Write operations
    if tool_lower in ("write", "write_file"):
        path = input_data.get("file_path") or input_data.get("path", "")
        content = input_data.get("content", "")
        return RichToolInfo(
            title=f"Write {path}" if path else "Write File",
            kind="edit",
            locations=[LocationContentItem(path=path)] if path else [],
            content=[DiffContentItem(path=path, old_text=None, new_text=content)] if path else [],
        )

    # Edit operations
    if tool_lower in ("edit", "edit_file"):
        path = input_data.get("file_path") or input_data.get("path", "")
        old_string = input_data.get("old_string") or input_data.get("old_text", "")
        new_string = input_data.get("new_string") or input_data.get("new_text", "")
        return RichToolInfo(
            title=f"Edit {path}" if path else "Edit File",
            kind="edit",
            locations=[LocationContentItem(path=path)] if path else [],
            content=[DiffContentItem(path=path, old_text=old_string, new_text=new_string)]
            if path
            else [],
        )

    # Delete operations
    if tool_lower in ("delete", "delete_path", "delete_file"):
        path = input_data.get("file_path") or input_data.get("path", "")
        return RichToolInfo(
            title=f"Delete {path}" if path else "Delete",
            kind="delete",
            locations=[LocationContentItem(path=path)] if path else [],
        )

    # Bash/terminal operations
    if tool_lower in ("bash", "execute", "run_command"):
        command = input_data.get("command", "")
        # Escape backticks in command
        escaped_cmd = command.replace("`", "\\`") if command else ""
        title = f"`{escaped_cmd}`" if escaped_cmd else "Terminal"
        return RichToolInfo(title=title, kind="execute")

    # Search operations
    if tool_lower in ("grep", "search", "glob", "find"):
        pattern = input_data.get("pattern") or input_data.get("query", "")
        path = input_data.get("path", "")
        title = f"Search for '{pattern}'" if pattern else "Search"
        if path:
            title += f" in {path}"
        return RichToolInfo(
            title=title,
            kind="search",
            locations=[LocationContentItem(path=path)] if path else [],
        )

    # List directory
    if tool_lower in ("ls", "list", "list_directory"):
        path = input_data.get("path", ".")
        return RichToolInfo(
            title=f"List {path}" if path != "." else "List current directory",
            kind="search",
            locations=[LocationContentItem(path=path)] if path else [],
        )

    # Web operations
    if tool_lower in ("webfetch", "web_fetch", "fetch"):
        url = input_data.get("url", "")
        return RichToolInfo(title=f"Fetch {url}" if url else "Web Fetch", kind="fetch")

    if tool_lower in ("websearch", "web_search", "search_web"):
        query = input_data.get("query", "")
        return RichToolInfo(title=f"Search: {query}" if query else "Web Search", kind="fetch")

    # Task/subagent operations
    if tool_lower == "task":
        description = input_data.get("description", "")
        return RichToolInfo(title=description if description else "Task", kind="think")

    # Notebook operations
    if tool_lower in ("notebookread", "notebook_read"):
        path = input_data.get("notebook_path", "")
        return RichToolInfo(
            title=f"Read Notebook {path}" if path else "Read Notebook",
            kind="read",
            locations=[LocationContentItem(path=path)] if path else [],
        )

    if tool_lower in ("notebookedit", "notebook_edit"):
        path = input_data.get("notebook_path", "")
        return RichToolInfo(
            title=f"Edit Notebook {path}" if path else "Edit Notebook",
            kind="edit",
            locations=[LocationContentItem(path=path)] if path else [],
        )

    # Default: use the tool name as title
    return RichToolInfo(title=actual_name, kind="other")


def content_block_to_event(
    block: ContentBlock,
    index: int = 0,
) -> RichAgentStreamEvent[Any] | None:
    """Convert a Claude SDK ContentBlock to a streaming event.

    Args:
        block: Claude SDK content block
        index: Part index for the event

    Returns:
        Corresponding streaming event, or None if not mappable
    """
    from claude_agent_sdk import TextBlock, ThinkingBlock, ToolUseBlock

    match block:
        case TextBlock(text=text):
            return PartDeltaEvent(index=index, delta=TextPartDelta(content_delta=text))
        case ThinkingBlock(thinking=thinking):
            return PartDeltaEvent(index=index, delta=ThinkingPartDelta(content_delta=thinking))
        case ToolUseBlock(id=tool_id, name=name, input=input_data):
            rich_info = derive_rich_tool_info(name, input_data)
            return ToolCallStartEvent(
                tool_call_id=tool_id,
                tool_name=name,
                title=rich_info.title,
                kind=rich_info.kind,
                locations=rich_info.locations,
                content=rich_info.content,
                raw_input=input_data,
            )
        case _:
            return None


def claude_message_to_events(
    message: Message,
    agent_name: str = "",
    pending_tool_calls: dict[str, ToolUseBlock] | None = None,
) -> list[RichAgentStreamEvent[Any]]:
    """Convert a Claude SDK Message to a list of streaming events.

    Args:
        message: Claude SDK message (UserMessage, AssistantMessage, etc.)
        agent_name: Name of the agent for event attribution
        pending_tool_calls: Dict to track tool calls awaiting results

    Returns:
        List of corresponding streaming events
    """
    from claude_agent_sdk import AssistantMessage, ToolResultBlock, ToolUseBlock

    events: list[RichAgentStreamEvent[Any]] = []

    match message:
        case AssistantMessage(content=content):
            for idx, block in enumerate(content):
                # Track tool use blocks for later pairing with results
                if isinstance(block, ToolUseBlock) and pending_tool_calls is not None:
                    pending_tool_calls[block.id] = block

                # Handle tool results - pair with pending tool call
                if isinstance(block, ToolResultBlock) and pending_tool_calls is not None:
                    tool_use = pending_tool_calls.pop(block.tool_use_id, None)
                    if tool_use:
                        events.append(
                            ToolCallCompleteEvent(
                                tool_name=tool_use.name,
                                tool_call_id=block.tool_use_id,
                                tool_input=tool_use.input,
                                tool_result=block.content,
                                agent_name=agent_name,
                                message_id="",
                            )
                        )
                    continue

                # Convert other blocks to events
                if event := content_block_to_event(block, index=idx):
                    events.append(event)

        case _:
            # UserMessage, SystemMessage, ResultMessage - no events to emit
            pass

    return events


def extract_text_from_message(message: Message) -> str:
    """Extract text content from a Claude SDK message.

    Args:
        message: Claude SDK message

    Returns:
        Concatenated text content from all text blocks
    """
    from claude_agent_sdk import AssistantMessage, TextBlock

    if not isinstance(message, AssistantMessage):
        return ""
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))
