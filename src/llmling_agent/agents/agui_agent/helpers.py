"""AG-UI agent helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import TypeAdapter

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ag_ui.core import Event, ToolMessage
    import httpx

    from llmling_agent.tools import Tool


logger = get_logger(__name__)


async def execute_tool_calls(
    tool_calls: list[tuple[str, str, dict[str, Any]]],
    tools_by_name: dict[str, Tool],
) -> list[ToolMessage]:
    """Execute tool calls locally and return results.

    Args:
        tool_calls: List of (tool_call_id, tool_name, args) tuples
        tools_by_name: Dictionary mapping tool names to Tool instances

    Returns:
        List of ToolMessage with execution results
    """
    from ag_ui.core import ToolMessage as AGUIToolMessage

    results: list[AGUIToolMessage] = []
    for tc_id, tool_name, args in tool_calls:
        if tool_name not in tools_by_name:
            logger.warning("Unknown tool requested", tool=tool_name)
            result_msg = AGUIToolMessage(
                id=str(uuid4()),
                tool_call_id=tc_id,
                content=f"Error: Unknown tool {tool_name!r}",
                error=f"Tool {tool_name!r} not found",
            )
        else:
            tool = tools_by_name[tool_name]
            logger.info("Executing tool", tool=tool_name, args=args)
            try:
                result = await tool.execute(**args)
                result_str = str(result) if not isinstance(result, str) else result
                id_ = str(uuid4())
                result_msg = AGUIToolMessage(id=id_, tool_call_id=tc_id, content=result_str)
                logger.debug("Tool executed", tool=tool_name, result=result_str[:100])
            except Exception as e:
                logger.exception("Tool execution failed", tool=tool_name)
                result_msg = AGUIToolMessage(
                    id=str(uuid4()),
                    tool_call_id=tc_id,
                    content=f"Error executing tool: {e}",
                    error=str(e),
                )
        results.append(result_msg)
    return results


async def parse_sse_stream(response: httpx.Response) -> AsyncIterator[Event]:
    """Parse Server-Sent Events stream.

    Args:
        response: HTTP response with SSE stream

    Yields:
        Parsed AG-UI events
    """
    from ag_ui.core import Event

    event_adapter: TypeAdapter[Event] = TypeAdapter(Event)
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        # Process complete SSE events
        while "\n\n" in buffer:
            event_text, buffer = buffer.split("\n\n", 1)
            # Parse SSE format: "data: {json}\n"
            for line in event_text.split("\n"):
                if not line.startswith("data: "):
                    continue
                json_str = line[6:]  # Remove "data: " prefix
                try:
                    event = event_adapter.validate_json(json_str)
                    yield event
                except (ValueError, TypeError) as e:
                    logger.warning("Failed to parse AG-UI event", json=json_str[:100], error=str(e))
