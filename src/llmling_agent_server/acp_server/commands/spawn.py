"""Spawn subagent slash command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import uuid

from pydantic_ai import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPartDelta,
    ToolReturnPart,
)
from slashed import CommandContext, CommandError  # noqa: TC002

from llmling_agent.agents.events import StreamCompleteEvent, ToolCallProgressEvent
from llmling_agent.log import get_logger
from llmling_agent.messaging.context import NodeContext  # noqa: TC001
from llmling_agent_commands.base import NodeCommand
from llmling_agent_server.acp_server.session import ACPSession  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent.agents.events import RichAgentStreamEvent


logger = get_logger(__name__)


class SpawnSubagentCommand(NodeCommand):
    """Spawn a subagent to execute a specific task.

    The subagent runs concurrently and reports progress in a dedicated tool call box.

    Usage:
      /spawn "agent-name" "prompt for the subagent"
      /spawn "code-reviewer" "Review the main.py file for potential bugs"
    """

    name = "spawn"
    category = "agents"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        agent_name: str,
        task_prompt: str,
    ) -> None:
        """Spawn a subagent to execute a task.

        Args:
            ctx: Command context with ACP session
            agent_name: Name of the agent to spawn
            task_prompt: Task prompt for the subagent
        """
        session = ctx.context.data
        assert session, "ACP session required for spawn command"
        # Generate unique tool call ID
        tool_call_id = f"spawn-{agent_name}-{uuid.uuid4().hex[:8]}"
        try:
            # Check if agent exists in pool
            if not session.agent_pool or agent_name not in session.agent_pool.agents:
                available = list(session.agent_pool.agents.keys())
                error_msg = f"Agent {agent_name!r} not found. Available agents: {available}"
                await ctx.print(f"❌ {error_msg}")
                return

            target_agent = session.agent_pool.get_agent(agent_name)
            await session.notifications.tool_call_start(
                tool_call_id=tool_call_id,
                title=f"Spawning agent: {agent_name}",
                kind="execute",
                raw_input={
                    "agent_name": agent_name,
                    "task_prompt": task_prompt,
                },
            )

            aggregated_content: list[str] = []  # Aggregate output as we stream
            try:
                # Run the subagent and handle events
                async for event in target_agent.run_stream(task_prompt):
                    await _handle_subagent_event(event, tool_call_id, aggregated_content, session)

                final_content = "".join(aggregated_content).strip()
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="completed",
                    content=[final_content] if final_content else None,
                )
            except Exception as e:
                error_msg = f"Subagent execution failed: {e}"
                logger.exception("Subagent execution error", error=str(e))
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="failed",
                    raw_output=error_msg,
                )

        except Exception as e:
            error_msg = f"Failed to spawn agent '{agent_name}': {e}"
            logger.exception("Spawn command error", error=str(e))
            raise CommandError(error_msg) from e


async def _handle_subagent_event(
    event: RichAgentStreamEvent[Any],
    tool_call_id: str,
    aggregated_content: list[str],
    session: ACPSession,
) -> None:
    """Handle events from spawned subagent and convert to tool_call_progress.

    Args:
        event: Event from the subagent stream
        tool_call_id: ID of the tool call box
        aggregated_content: List to accumulate content for final display
        session: ACP session for notifications
    """
    match event:
        case (
            PartStartEvent(part=TextPart(content=delta))
            | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
        ):
            # Subagent text output → accumulate and update progress
            aggregated_content.append(delta)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="in_progress",
                content=["".join(aggregated_content)],
            )

        case (
            PartStartEvent(part=ThinkingPart(content=delta))
            | PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta))
        ):
            # Subagent thinking → show thinking indicator
            if delta:
                thinking_text = f"💭 {delta}"
                aggregated_content.append(thinking_text)
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="in_progress",
                    content=["".join(aggregated_content)],
                )

        case FunctionToolCallEvent(part=part):
            # Subagent calls a tool → show nested tool call
            tool_text = f"\n🔧 Using tool: {part.tool_name}\n"
            aggregated_content.append(tool_text)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="in_progress",
                content=["".join(aggregated_content)],
            )

        case FunctionToolResultEvent(
            result=ToolReturnPart(content=content, tool_name=tool_name),
        ):
            # Subagent tool completes → show tool result
            result_text = f"✅ {tool_name}: {content}\n"
            aggregated_content.append(result_text)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="in_progress",
                content=["".join(aggregated_content)],
            )

        case FunctionToolResultEvent(
            result=RetryPromptPart(tool_name=tool_name) as result,
        ):
            # Tool call failed and needs retry
            error_message = result.model_response()
            error_text = f"❌ {tool_name or 'unknown'}: Error: {error_message}\n"
            aggregated_content.append(error_text)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="in_progress",
                content=["".join(aggregated_content)],
            )

        case ToolCallProgressEvent(message=message, tool_name=tool_name):
            # Progress event from tools
            if message:
                progress_text = f"🔄 {tool_name}: {message}\n"
                aggregated_content.append(progress_text)
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="in_progress",
                    content=["".join(aggregated_content)],
                )

        case (
            PartStartEvent()
            | PartDeltaEvent(delta=ToolCallPartDelta())
            | FinalResultEvent()
            | StreamCompleteEvent()
        ):
            pass  # These events don't need special handling

        case _:
            logger.debug("Unhandled subagent event", event_type=type(event).__name__)
