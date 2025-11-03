"""ACP (Agent Client Protocol) session management for llmling-agent.

This module provides session lifecycle management, state tracking, and coordination
between agents and ACP clients through the JSON-RPC protocol.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    UsageLimitExceeded,
)

from acp.filesystem import ACPFileSystem
from acp.notifications import ACPNotifications
from acp.requests import ACPRequests
from acp.utils import to_acp_content_blocks
from llmling_agent.agent.events import StreamCompleteEvent, ToolCallProgressEvent
from llmling_agent.log import get_logger
from llmling_agent_acp.acp_tools import get_acp_provider
from llmling_agent_acp.command_bridge import SLASH_PATTERN
from llmling_agent_acp.converters import (
    convert_acp_mcp_server_to_config,
    from_content_blocks,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from acp import Client
    from acp.schema import ClientCapabilities, ContentBlock, McpServer, StopReason
    from llmling_agent import Agent, AgentPool
    from llmling_agent.agent.events import RichAgentStreamEvent
    from llmling_agent.models.content import BaseContent
    from llmling_agent.resource_providers.aggregating import AggregatingResourceProvider
    from llmling_agent_acp.acp_agent import LLMlingACPAgent
    from llmling_agent_acp.command_bridge import ACPCommandBridge
    from llmling_agent_acp.session_manager import ACPSessionManager
    from llmling_agent_providers.base import UsageLimits


logger = get_logger(__name__)
# Tools that send their own rich ACP notifications (with ToolCallLocation, etc.)
# These tools are excluded from generic session-level notifications to prevent duplication
ACP_SELF_NOTIFYING_TOOLS = {"read_text_file", "write_text_file", "run_command"}


def _is_slash_command(text: str) -> bool:
    """Check if text starts with a slash command."""
    return bool(SLASH_PATTERN.match(text.strip()))


@dataclass
class ACPSession:
    """Individual ACP session state and management.

    Manages the lifecycle and state of a single ACP session, including:
    - Agent instance and conversation state
    - Working directory and environment
    - MCP server connections
    - File system bridge for client operations
    - Tool execution and streaming updates
    """

    session_id: str
    """Unique session identifier"""

    agent_pool: AgentPool[Any]
    """AgentPool containing available agents"""

    current_agent_name: str
    """Name of currently active agent"""

    cwd: str
    """Working directory for the session"""

    client: Client
    """External library Client interface for operations"""

    acp_agent: LLMlingACPAgent
    """ACP agent instance for capability tools"""

    mcp_servers: Sequence[McpServer] | None = None
    """Optional MCP server configurations"""

    usage_limits: UsageLimits | None = None
    """Optional usage limits for model requests and tokens"""

    command_bridge: ACPCommandBridge | None = None
    """Optional command bridge for slash commands"""

    client_capabilities: ClientCapabilities | None = None
    """Client capabilities for tool registration"""

    manager: ACPSessionManager | None = None
    """Session manager for managing sessions. Used for session management commands."""

    def __post_init__(self) -> None:
        """Initialize session state and set up providers."""
        self.mcp_servers = self.mcp_servers or []
        self.log = logger.bind(session_id=self.session_id)
        self._active = True
        self._task_lock = asyncio.Lock()
        self._cancelled = False
        self._current_tool_inputs: dict[str, dict] = {}
        self.fs = ACPFileSystem(self.client, session_id=self.session_id)
        self._acp_provider: AggregatingResourceProvider | None = None
        self.notifications = ACPNotifications(
            client=self.client,
            session_id=self.session_id,
        )
        self.requests = ACPRequests(client=self.client, session_id=self.session_id)

        if self.client_capabilities:
            self._acp_provider = get_acp_provider(self)
            current_agent = self.agent_pool.get_agent(self.current_agent_name)
            current_agent.tools.add_provider(self._acp_provider)

        # Add cwd context to all agents in the pool
        for agent in self.agent_pool.agents.values():
            agent.sys_prompts.prompts.append(self.get_cwd_context)  # pyright: ignore[reportArgumentType]

        self.log.info("Created ACP session", current_agent=self.current_agent_name)

    async def initialize_mcp_servers(self) -> None:
        """Initialize MCP servers if any are configured."""
        if not self.mcp_servers:
            return
        self.log.info("Initializing MCP servers", server_count=len(self.mcp_servers))
        cfgs = [convert_acp_mcp_server_to_config(s) for s in self.mcp_servers]
        # Define accessible roots for MCP servers
        # root = Path(self.cwd).resolve().as_uri() if self.cwd else None
        for cfg in cfgs:
            try:
                await self.agent_pool.mcp.setup_server(cfg)
                self.log.info("Added MCP servers", server_count=len(cfgs))
                await self._register_mcp_prompts_as_commands()
            except Exception:
                self.log.exception("Failed to initialize MCP manager")
                # Don't fail session creation, just log the error
                self.mcp_manager = None

    async def init_project_context(self) -> None:
        """Load AGENTS.md file and inject project context into all agents.

        TODO: Consider moving this to __aenter__
        """
        if info := await self.requests.read_agent_rules(self.cwd):
            for agent in self.agent_pool.agents.values():
                prompt = f"## Project Information\n\n{info}"
                agent.sys_prompts.prompts.append(prompt)

    @property
    def agent(self) -> Agent[Any, str]:
        """Get the currently active agent."""
        return self.agent_pool.get_agent(self.current_agent_name)

    def get_cwd_context(self) -> str:
        """Get current working directory context for prompts."""
        return f"Working directory: {self.cwd}" if self.cwd else ""

    async def switch_active_agent(self, agent_name: str) -> None:
        """Switch to a different agent in the pool.

        Args:
            agent_name: Name of the agent to switch to

        Raises:
            ValueError: If agent not found in pool
        """
        if agent_name not in self.agent_pool.agents:
            available = list(self.agent_pool.agents.keys())
            msg = f"Agent {agent_name!r} not found. Available: {available}"
            raise ValueError(msg)

        old_agent_name = self.current_agent_name
        self.current_agent_name = agent_name

        if self._acp_provider:  # Move capability provider from old agent to new agent
            old_agent = self.agent_pool.get_agent(old_agent_name)
            new_agent = self.agent_pool.get_agent(agent_name)
            old_agent.tools.remove_provider(self._acp_provider)
            new_agent.tools.add_provider(self._acp_provider)

        self.log.info("Switched agents", from_agent=old_agent_name, to_agent=agent_name)
        # if new_model := new_agent.model_name:
        #     await self.notifications.update_session_model(new_model)
        await self.send_available_commands_update()

    @property
    def active(self) -> bool:
        """Check if session is active."""
        return self._active

    def cancel(self) -> None:
        """Cancel the current prompt turn."""
        self._cancelled = True
        self.log.info("Session cancelled")

    def is_cancelled(self) -> bool:
        """Check if the session is cancelled."""
        return self._cancelled

    async def process_prompt(self, content_blocks: Sequence[ContentBlock]) -> StopReason:  # noqa: PLR0911
        """Process a prompt request and stream responses.

        Args:
            content_blocks: List of content blocks from the prompt request

        Returns:
            Stop reason
        """
        if not self._active:
            self.log.warning("Attempted to process prompt on inactive session")
            return "refusal"

        self._cancelled = False
        contents = from_content_blocks(content_blocks)
        self.log.debug("Converted content", content=contents)
        if not contents:
            self.log.warning("Empty prompt received")
            return "refusal"
        # Check for slash commands in text content
        commands: list[str] = []
        non_command_content: list[str | BaseContent] = []
        for item in contents:
            if isinstance(item, str) and _is_slash_command(item):
                self.log.info("Found slash command", command=item)
                commands.append(item.strip())
            else:
                non_command_content.append(item)

        async with self._task_lock:
            # Process commands if found
            if commands and self.command_bridge:
                for command in commands:
                    self.log.info("Processing slash command", command=command)
                    await self.command_bridge.execute_slash_command(command, self)

                # If only commands, end turn
                if not non_command_content:
                    return "end_turn"

            self.log.debug(
                "Processing prompt",
                content_items=len(non_command_content),
                model=self.agent.model_name,
            )
            event_count = 0
            self._current_tool_inputs.clear()  # Reset tool inputs for new stream

            try:
                async for event in self.agent.run_stream(
                    *non_command_content, usage_limits=self.usage_limits
                ):
                    if self._cancelled:
                        return "cancelled"

                    event_count += 1
                    await self.handle_event(event)
                self.log.info("Streaming finished", events_processed=event_count)

            except UsageLimitExceeded as e:
                self.log.info("Usage limit exceeded", error=str(e))
                error_msg = str(e)  # Determine which limit was hit based on error
                if "request_limit" in error_msg:
                    return "max_turn_requests"
                if any(limit in error_msg for limit in ["tokens_limit", "token_limit"]):
                    return "max_tokens"
                # Tool call limits don't have a direct ACP stop reason, treat as refusal
                if "tool_calls_limit" in error_msg or "tool call" in error_msg:
                    return "refusal"
                return "max_tokens"  # Default to max_tokens for other usage limits
            except Exception as e:
                self.log.exception("Error during streaming")
                await self.notifications.send_agent_text(f"Agent error: {e}")
                return "cancelled"
            else:
                return "end_turn"

    async def handle_event(self, event: RichAgentStreamEvent):
        match event:
            case (
                PartStartEvent(part=TextPart(content=delta))
                | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
            ):
                await self.notifications.send_agent_text(delta)

            case (
                PartStartEvent(part=ThinkingPart(content=delta))
                | PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta))
            ):
                await self.notifications.send_agent_thought(delta or "\n")

            case PartStartEvent(delta=delta):
                self.log.debug("Received unhandled PartStartEvent", delta=delta)

            case PartDeltaEvent(delta=ToolCallPartDelta()):
                self.log.debug("Received ToolCallPartDelta")

            case FunctionToolCallEvent(part=part):
                tool_call_id = part.tool_call_id
                self._current_tool_inputs[tool_call_id] = part.args_as_dict()
                # Skip generic notifications for self-notifying tools
                if part.tool_name not in ACP_SELF_NOTIFYING_TOOLS:
                    await self.notifications.tool_call(
                        tool_name=part.tool_name,
                        tool_input=part.args_as_dict(),
                        tool_output=None,  # Not available yet
                        status="pending",
                        tool_call_id=tool_call_id,
                    )

            case FunctionToolResultEvent(
                result=ToolReturnPart(content=content, tool_name=tool_name) as result,
                tool_call_id=tool_call_id,
            ):
                tool_input = self._current_tool_inputs.get(tool_call_id, {})
                if isinstance(content, AsyncGenerator):
                    full_content = ""
                    async for chunk in content:
                        full_content += str(chunk)
                        # Yield intermediate streaming notification
                        # Skip generic notifications for self-notifying tools
                        if tool_name not in ACP_SELF_NOTIFYING_TOOLS:
                            await self.notifications.tool_call(
                                tool_name=tool_name,
                                tool_input=tool_input,
                                tool_output=chunk,
                                status="in_progress",
                                tool_call_id=tool_call_id,
                            )

                    # Replace the AsyncGenerator with the full content to
                    # prevent errors
                    result.content = full_content
                    final_output = full_content
                else:
                    final_output = result.content

                # Final completion notification
                # Skip generic notifications for self-notifying tools
                if result.tool_name not in ACP_SELF_NOTIFYING_TOOLS:
                    converted_blocks = to_acp_content_blocks(final_output)
                    await self.notifications.tool_call(
                        tool_name=result.tool_name,
                        tool_input=tool_input,
                        tool_output=converted_blocks,
                        status="completed",
                        tool_call_id=tool_call_id,
                    )
                # Clean up stored input
                self._current_tool_inputs.pop(tool_call_id, None)

            case FunctionToolResultEvent(
                result=RetryPromptPart(tool_name=tool_name) as result,
                tool_call_id=tool_call_id,
            ):
                # Tool call failed and needs retry
                tool_name = tool_name or "unknown"
                error_message = result.model_response()
                # Skip generic notifications for self-notifying tools
                if tool_name not in ACP_SELF_NOTIFYING_TOOLS:
                    await self.notifications.tool_call(
                        tool_name=tool_name,
                        tool_input=self._current_tool_inputs.get(tool_call_id, {}),
                        tool_output=f"Error: {error_message}",
                        status="failed",
                        tool_call_id=tool_call_id,
                    )
                self._current_tool_inputs.pop(tool_call_id, None)  # Clean up stored input

            case ToolCallProgressEvent(
                progress=progress,
                total=total,
                message=message,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_input=tool_input,
            ):
                self.log.debug(
                    "Received progress event for tool",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
                output = message if message else f"Progress: {progress}"
                if total:
                    output += f"/{total}"
                try:
                    # Create content from progress message

                    # Create ACP tool call progress notification
                    # await self.notifications.tool_call(
                    #     tool_name=tool_name,
                    #     tool_input=tool_input or {},
                    #     tool_output=output,
                    #     status="in_progress",
                    #     tool_call_id=tool_call_id,
                    # )
                    await self.notifications.tool_call_progress(
                        title=message,
                        raw_output=output,
                        status="in_progress",
                        tool_call_id=tool_call_id,
                    )
                except Exception as e:  # noqa: BLE001
                    self.log.warning(
                        "Failed to convert progress event to ACP notification",
                        error=str(e),
                    )

            case FinalResultEvent():
                self.log.debug("Final result received")

            case StreamCompleteEvent(message=message):
                pass

            case _:
                self.log.debug("Unhandled event", event_type=type(event).__name__)

    async def close(self) -> None:
        """Close the session and cleanup resources."""
        if not self._active:
            return

        self._active = False
        self._current_tool_inputs.clear()

        try:
            # Clean up capability provider if present
            if self._acp_provider:
                current_agent = self.agent_pool.get_agent(self.current_agent_name)
                current_agent.tools.remove_provider(self._acp_provider)

            # Remove cwd context callable from all agents
            for agent in self.agent_pool.agents.values():
                if self.get_cwd_context in agent.sys_prompts.prompts:
                    agent.sys_prompts.prompts.remove(self.get_cwd_context)  # pyright: ignore[reportArgumentType]
                self._acp_provider = None

            # Note: Individual agents are managed by the pool's lifecycle
            # The pool will handle agent cleanup when it's closed
            self.log.info("Closed ACP session")
        except Exception:
            self.log.exception("Error closing session")

    async def send_available_commands_update(self) -> None:
        """Send current available commands to client."""
        if not self.command_bridge:
            return
        try:
            commands = self.command_bridge.get_acp_commands(self.agent.context)
            await self.notifications.update_commands(commands)
        except Exception:
            self.log.exception("Failed to send available commands update")

    async def _register_mcp_prompts_as_commands(self) -> None:
        """Register MCP prompts as slash commands."""
        if not self.command_bridge:
            return

        try:
            # Collect all prompts from all MCP clients concurrently
            clients = list(self.agent_pool.mcp.clients.values())
            if not clients:
                return

            # Use gather to fetch prompts concurrently, with exception handling
            results = await asyncio.gather(*[client.list_prompts() for client in clients])
            if all_prompts := [p for result in results for p in result]:
                self.command_bridge.add_mcp_prompt_commands(all_prompts)
                self.log.info(
                    "Registered MCP prompts as slash commands",
                    prompt_count=len(all_prompts),
                )
                # Send updated command list to client
                await self.send_available_commands_update()

        except Exception:
            self.log.exception("Failed to register MCP prompts as commands")
