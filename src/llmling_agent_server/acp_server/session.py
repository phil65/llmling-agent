"""ACP (Agent Client Protocol) session management for llmling-agent.

This module provides session lifecycle management, state tracking, and coordination
between agents and ACP clients through the JSON-RPC protocol.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
import re
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
from slashed import Command, CommandStore

from acp.acp_requests import ACPRequests
from acp.filesystem import ACPFileSystem
from acp.notifications import ACPNotifications
from acp.schema import AvailableCommand
from acp.utils import to_acp_content_blocks
from llmling_agent.agent import SlashedAgent
from llmling_agent.agent.events import StreamCompleteEvent, ToolCallProgressEvent
from llmling_agent.log import get_logger
from llmling_agent_commands import get_commands
from llmling_agent_server.acp_server.acp_tools import get_acp_provider
from llmling_agent_server.acp_server.converters import (
    convert_acp_mcp_server_to_config,
    from_content_blocks,
)
from llmling_agent_server.acp_server.input_provider import ACPInputProvider


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic_ai.messages import SystemPromptPart, UserPromptPart
    from slashed import CommandContext

    from acp import Client
    from acp.schema import ClientCapabilities, ContentBlock, McpServer, StopReason
    from llmling_agent import Agent, AgentPool
    from llmling_agent.agent.events import RichAgentStreamEvent
    from llmling_agent.models.content import BaseContent
    from llmling_agent.prompts.manager import PromptManager
    from llmling_agent.prompts.prompts import MCPClientPrompt
    from llmling_agent.resource_providers.aggregating import AggregatingResourceProvider
    from llmling_agent_server.acp_server.acp_agent import LLMlingACPAgent
    from llmling_agent_server.acp_server.session_manager import ACPSessionManager

logger = get_logger(__name__)
SLASH_PATTERN = re.compile(r"^/([\w-]+)(?:\s+(.*))?$")
ACP_COMMANDS = {"list-sessions", "load-session", "save-session", "delete-session"}


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
        # Staged prompt parts for context building
        from llmling_agent_server.acp_server.commands.acp_commands import get_acp_commands
        from llmling_agent_server.acp_server.commands.docs_commands import (
            get_docs_commands,
        )
        from llmling_agent_server.acp_server.commands.terminal_commands import (
            get_terminal_commands,
        )

        cmds = [
            *get_commands(
                enable_set_model=False,
                enable_enable_tool=False,
                enable_disable_tool=False,
                enable_list_resources=False,
                enable_add_resource=False,
                enable_show_resource=False,
                enable_copy_clipboard=False,
            ),
            *get_acp_commands(),
            *get_docs_commands(),
            *get_terminal_commands(),
        ]
        self.command_store = CommandStore(enable_system_commands=True, commands=cmds)
        self.command_store._initialize_sync()
        self._update_callbacks: list[Callable[[], None]] = []

        self._staged_parts: list[SystemPromptPart | UserPromptPart] = []
        self.notifications = ACPNotifications(
            client=self.client,
            session_id=self.session_id,
        )
        self.requests = ACPRequests(client=self.client, session_id=self.session_id)
        self.input_provider = ACPInputProvider(self)

        if self.client_capabilities:
            self._acp_provider = get_acp_provider(self)
            current_agent = self.agent
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
        for _cfg in cfgs:
            try:
                # Server will be initialized when MCP manager enters context
                self.log.info("Added MCP servers", server_count=len(cfgs))
                await self._register_mcp_prompts_as_commands()
            except Exception:
                self.log.exception("Failed to initialize MCP manager")
                # Don't fail session creation, just log the error

    async def init_project_context(self) -> None:
        """Load AGENTS.md file and inject project context into all agents.

        TODO: Consider moving this to __aenter__
        """
        if info := await self.requests.read_agent_rules(self.cwd):
            for agent in self.agent_pool.agents.values():
                prompt = f"## Project Information\n\n{info}"
                agent.sys_prompts.prompts.append(prompt)

    @property
    def agent(self) -> Agent[ACPSession, str]:
        """Get the currently active agent."""
        return self.agent_pool.get_agent(self.current_agent_name, deps_type=ACPSession)

    @property
    def slashed_agent(self) -> SlashedAgent[Any, str]:
        """Get the wrapped slashed agent."""
        return SlashedAgent(self.agent, command_store=self.command_store)

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

    def get_staged_parts(self) -> list[SystemPromptPart | UserPromptPart]:
        """Get copy of currently staged prompt parts."""
        return self._staged_parts.copy()

    def add_staged_parts(self, parts: list[SystemPromptPart | UserPromptPart]) -> None:
        """Add prompt parts to staging area.

        Args:
            parts: List of SystemPromptPart or UserPromptPart to stage
        """
        self._staged_parts.extend(parts)

    def clear_staged_parts(self) -> None:
        """Clear all staged prompt parts."""
        self._staged_parts.clear()

    def get_staged_parts_count(self) -> int:
        """Get count of staged parts."""
        return len(self._staged_parts)

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
            if commands:
                for command in commands:
                    self.log.info("Processing slash command", command=command)
                    await self.execute_slash_command(command)

                # If only commands, end turn
                if not non_command_content:
                    return "end_turn"

            self.log.debug("Processing prompt", content_items=len(non_command_content))
            event_count = 0
            self._current_tool_inputs.clear()  # Reset tool inputs for new stream

            try:
                # Use the session's persistent input provider
                async for event in self.agent.run_stream(
                    *non_command_content, input_provider=self.input_provider
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
                await self.notifications.send_agent_text(f"❌ Agent error: {e}")
                return "cancelled"
            else:
                return "end_turn"

    async def handle_event(self, event: RichAgentStreamEvent):  # noqa: PLR0915
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

            case PartDeltaEvent(delta=ToolCallPartDelta() as delta):
                if part := delta.as_part():
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
                current_agent = self.agent
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
        try:
            commands = self.get_acp_commands()
            await self.notifications.update_commands(commands)
        except Exception:
            self.log.exception("Failed to send available commands update")

    async def _register_mcp_prompts_as_commands(self) -> None:
        """Register MCP prompts as slash commands."""
        try:
            # Get all prompts from the agent's ToolManager
            if all_prompts := await self.agent.tools.list_prompts():
                for prompt in all_prompts:
                    command = self.create_mcp_command(prompt)
                    self.command_store.register_command(command)
                self._notify_command_update()
                self.log.info(
                    "Registered MCP prompts as slash commands",
                    prompt_count=len(all_prompts),
                )
                # Send updated command list to client
                await self.send_available_commands_update()

        except Exception:
            self.log.exception("Failed to register MCP prompts as commands")

    async def _register_prompt_hub_commands(self) -> None:
        """Register prompt hub prompts as slash commands."""
        try:
            prompt_manager = self.agent_pool.manifest.prompt_manager
            all_prompts = await prompt_manager.list_prompts()
            command_count = 0
            for provider_name, prompt_names in all_prompts.items():
                if not prompt_names:  # Skip empty providers
                    continue

                for prompt_name in prompt_names:
                    command = self.create_prompt_hub_command(
                        provider_name, prompt_name, prompt_manager
                    )
                    self.command_store.register_command(command)
                    command_count += 1

            if command_count > 0:
                self._notify_command_update()
                self.log.info(
                    "Registered prompt hub prompts as slash commands",
                    command_count=command_count,
                )
                # Send updated command list to client
                await self.send_available_commands_update()

        except Exception:
            self.log.exception("Failed to register prompt hub prompts as commands")

    def _notify_command_update(self) -> None:
        """Notify all registered callbacks about command updates."""
        for callback in self._update_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Command update callback failed")

    def get_acp_commands(self) -> list[AvailableCommand]:
        """Convert all slashed commands to ACP format.

        Args:
            context: Optional agent context to filter commands

        Returns:
            List of ACP AvailableCommand objects
        """
        return [
            AvailableCommand.create(
                name=cmd.name,
                description=cmd.description,
                input_hint=cmd.usage,
            )
            for cmd in self.command_store.list_commands()
        ]

    async def execute_slash_command(self, command_text: str) -> None:
        """Execute any slash command with unified handling.

        Args:
            command_text: Full command text (including slash)
            session: ACP session context
        """
        if match := SLASH_PATTERN.match(command_text.strip()):
            command_name = match.group(1)
            args = match.group(2) or ""
        else:
            logger.warning("Invalid slash command", command=command_text)
            return

        self.agent.context.data = self
        cmd_ctx = self.command_store.create_context(
            data=self.agent.context,
            output_writer=self.notifications.send_agent_text,
        )

        command_str = f"{command_name} {args}".strip()
        try:
            await self.command_store.execute_command(command_str, cmd_ctx)
        except Exception as e:
            logger.exception("Command execution failed")
            await self.notifications.send_agent_text(f"❌ Command error: {e}")

    def register_update_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for command updates.

        Args:
            callback: Function to call when commands are updated
        """
        self._update_callbacks.append(callback)

    def create_mcp_command(self, prompt: MCPClientPrompt) -> Command:
        """Convert MCP prompt to slashed Command.

        Args:
            prompt: MCP prompt to wrap
            session: ACP session for execution context

        Returns:
            Slashed Command that executes the prompt
        """

        async def execute_prompt(
            ctx: CommandContext,
            args: list[str],
            kwargs: dict[str, str],
        ) -> None:
            """Execute the MCP prompt with parsed arguments."""
            # Map parsed args to prompt parameters

            result = {}
            # Map positional args to prompt parameter names
            for i, arg_value in enumerate(args):
                if i < len(prompt.arguments):
                    param_name = prompt.arguments[i]["name"]
                    result[param_name] = arg_value
            result.update(kwargs)
            try:
                # Get prompt components
                components = await prompt.get_components(result or None)
                self.add_staged_parts(components)

                # Send confirmation
                staged_count = self.get_staged_parts_count()
                await ctx.print(
                    f"✅ Prompt '{prompt.name}' staged ({staged_count} total parts)"
                )

            except Exception as e:
                logger.exception("MCP prompt execution failed", prompt=prompt.name)
                await ctx.print(f"❌ Prompt error: {e}")

        usage_hint = (
            " ".join(f"<{arg['name']}>" for arg in prompt.arguments)
            if prompt.arguments
            else None
        )
        return Command(
            execute_func=execute_prompt,
            name=prompt.name,
            description=prompt.description or f"MCP prompt: {prompt.name}",
            category="mcp",
            usage=usage_hint,
        )

    def create_prompt_hub_command(
        self, provider: str, name: str, manager: PromptManager
    ) -> Command:
        """Convert prompt hub prompt to slash command.

        Args:
            provider: Provider name (e.g., 'langfuse', 'builtin')
            name: Prompt name
            manager: PromptManager instance

        Returns:
            Command that executes the prompt hub prompt
        """

        async def execute_prompt(
            ctx: CommandContext,
            args: list[str],
            kwargs: dict[str, str],
        ) -> None:
            """Execute the prompt hub prompt with parsed arguments."""
            try:
                # Build reference string
                reference = f"{provider}:{name}" if provider != "builtin" else name

                # Add variables as query parameters if provided
                if kwargs:
                    params = "&".join(f"{k}={v}" for k, v in kwargs.items())
                    reference = f"{reference}?{params}"

                # Get the rendered prompt
                result = await manager.get(reference)

                # Convert string result to message parts and stage them
                from pydantic_ai.messages import UserPromptPart

                self.add_staged_parts([UserPromptPart(content=result)])

                # Send confirmation
                staged_count = self.get_staged_parts_count()
                await ctx.print(
                    f"✅ Prompt {name!r} from {provider} staged ({staged_count} total parts)"  # noqa: E501
                )

            except Exception as e:
                logger.exception(
                    "Prompt hub execution failed", prompt=name, provider=provider
                )
                await ctx.print(f"❌ Prompt error: {e}")

        # Create command name - prefix with provider if not builtin
        command_name = f"{provider}_{name}" if provider != "builtin" else name

        return Command(
            execute_func=execute_prompt,
            name=command_name,
            description=f"Prompt hub: {provider}:{name}",
            category="prompts",
            usage="[key=value ...]",  # Generic since we don't have parameter schemas
        )
