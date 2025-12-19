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

    from llmling_agent.agents.context import ConfirmationResult
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.tools import Tool
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.nodes import ToolConfirmationMode


logger = get_logger(__name__)


async def _should_confirm_tool(
    tool: Tool,
    confirmation_mode: ToolConfirmationMode,
) -> bool:
    """Determine if a tool requires confirmation based on mode.

    Args:
        tool: The tool to check
        confirmation_mode: Current confirmation mode

    Returns:
        True if confirmation is required
    """
    if confirmation_mode == "never":
        return False
    if confirmation_mode == "always":
        return True
    # "per_tool" mode - check tool's requires_confirmation attribute
    return tool.requires_confirmation


async def _get_tool_confirmation(
    tool: Tool,
    args: dict[str, Any],
    input_provider: InputProvider,
    context: NodeContext[Any],
) -> ConfirmationResult:
    """Get confirmation for tool execution.

    Args:
        tool: Tool to confirm
        args: Tool arguments
        input_provider: Provider for confirmation UI
        context: Node context

    Returns:
        Confirmation result
    """
    return await input_provider.get_tool_confirmation(
        context=context,
        tool=tool,
        args=args,
        message_history=None,
    )


async def execute_tool_calls(
    tool_calls: list[tuple[str, str, dict[str, Any]]],
    tools_by_name: dict[str, Tool],
    *,
    confirmation_mode: ToolConfirmationMode = "never",
    input_provider: InputProvider | None = None,
    context: NodeContext[Any] | None = None,
) -> list[ToolMessage]:
    """Execute tool calls locally and return results.

    Args:
        tool_calls: List of (tool_call_id, tool_name, args) tuples
        tools_by_name: Dictionary mapping tool names to Tool instances
        confirmation_mode: Tool confirmation mode
        input_provider: Optional input provider for confirmation requests
        context: Optional node context for confirmation

    Returns:
        List of ToolMessage with execution results
    """
    from ag_ui.core import ToolMessage as AGUIToolMessage

    results: list[AGUIToolMessage] = []
    for tc_id, tool_name, args in tool_calls:
        if tool_name not in tools_by_name:
            # Tool not registered locally - this is a server-side tool, skip it
            logger.debug("Skipping server-side tool", tool=tool_name)
            continue

        tool = tools_by_name[tool_name]

        # Check if confirmation is required
        if await _should_confirm_tool(tool, confirmation_mode):
            if input_provider is None or context is None:
                logger.warning(
                    "Tool requires confirmation but no input provider available",
                    tool=tool_name,
                )
                result_msg = AGUIToolMessage(
                    id=str(uuid4()),
                    tool_call_id=tc_id,
                    content="Error: Tool requires confirmation but no provider available",
                    error="No input provider for confirmation",
                )
                results.append(result_msg)
                continue

            confirmation = await _get_tool_confirmation(tool, args, input_provider, context)
            if confirmation == "skip":
                logger.info("Tool execution skipped by user", tool=tool_name)
                result_msg = AGUIToolMessage(
                    id=str(uuid4()),
                    tool_call_id=tc_id,
                    content="Tool execution was skipped by user",
                    error=None,
                )
                results.append(result_msg)
                continue
            if confirmation in ("abort_run", "abort_chain"):
                logger.info("Tool execution aborted by user", tool=tool_name, action=confirmation)
                result_msg = AGUIToolMessage(
                    id=str(uuid4()),
                    tool_call_id=tc_id,
                    content="Tool execution was aborted by user",
                    error="Execution aborted",
                )
                results.append(result_msg)
                # Could raise an exception here to abort the entire run
                continue

        # Execute the tool
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
