"""ACP (Agent Client Protocol) session management for llmling-agent.

This module provides session lifecycle management, state tracking, and coordination
between agents and ACP clients through the JSON-RPC protocol.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Any

from exxec.acp_provider import ACPExecutionEnvironment
import logfire
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
    UserPromptPart,
)
from slashed import Command, CommandStore

from acp import RequestPermissionRequest
from acp.acp_requests import ACPRequests
from acp.filesystem import ACPFileSystem
from acp.notifications import ACPNotifications
from acp.schema import (
    AvailableCommand,
    ClientCapabilities,
    ContentToolCallContent,
    PlanEntry,
    TerminalToolCallContent,
    ToolCallLocation,
)
from acp.tool_call_state import ToolCallState
from acp.utils import generate_tool_title, infer_tool_kind, to_acp_content_blocks
from llmling_agent import Agent, AgentContext  # noqa: TC001
from llmling_agent.agents import SlashedAgent
from llmling_agent.agents.acp_agent import ACPAgent
from llmling_agent.agents.events import (
    PlanUpdateEvent,
    StreamCompleteEvent,
    ToolCallProgressEvent,
    ToolCallStartEvent,
)
from llmling_agent.log import get_logger
from llmling_agent_commands import get_commands
from llmling_agent_commands.base import NodeCommand
from llmling_agent_server.acp_server.converters import (
    convert_acp_mcp_server_to_config,
    from_acp_content,
)
from llmling_agent_server.acp_server.input_provider import ACPInputProvider


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic_ai import UserContent
    from pydantic_ai.messages import SystemPromptPart
    from slashed import CommandContext

    from acp import Client, RequestPermissionResponse
    from acp.schema import ContentBlock, McpServer, StopReason
    from llmling_agent import AgentPool
    from llmling_agent.agents import AGUIAgent
    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.prompts.manager import PromptManager
    from llmling_agent.prompts.prompts import MCPClientPrompt
    from llmling_agent_server.acp_server.acp_agent import LLMlingACPAgent
    from llmling_agent_server.acp_server.session_manager import ACPSessionManager

logger = get_logger(__name__)
SLASH_PATTERN = re.compile(r"^/([\w-]+)(?:\s+(.*))?$")
ACP_COMMANDS = {"list-sessions", "load-session", "save-session", "delete-session"}
MAX_CMD_OUTPUT_LENGTH = 60

logger = get_logger(__name__)


def _is_slash_command(text: str) -> bool:
    """Check if text starts with a slash command."""
    return bool(SLASH_PATTERN.match(text.strip()))


def split_commands(
    contents: Sequence[UserContent],
) -> tuple[list[str], list[UserContent]]:
    commands: list[str] = []
    non_command_content: list[UserContent] = []
    for item in contents:
        if isinstance(item, str) and _is_slash_command(item):
            commands.append(item.strip())
        else:
            non_command_content.append(item)
    return commands, non_command_content


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

    client_capabilities: ClientCapabilities = field(default_factory=ClientCapabilities)
    """Client capabilities for tool registration"""

    manager: ACPSessionManager | None = None
    """Session manager for managing sessions. Used for session management commands."""

    def __post_init__(self) -> None:
        """Initialize session state and set up providers."""
        from llmling_agent_server.acp_server.commands import get_commands as get_acp_commands

        self.mcp_servers = self.mcp_servers or []
        self.log = logger.bind(session_id=self.session_id)
        self._task_lock = asyncio.Lock()
        self._cancelled = False
        self._current_tool_inputs: dict[str, dict[str, Any]] = {}
        self._tool_call_states: dict[str, ToolCallState] = {}
        self.fs = ACPFileSystem(self.client, session_id=self.session_id)
        cmds = [
            *get_commands(
                enable_set_model=False,
                enable_list_resources=False,
                enable_add_resource=False,
                enable_show_resource=False,
            ),
            *get_acp_commands(),
        ]
        self.command_store = CommandStore(enable_system_commands=True, commands=cmds)
        self.command_store._initialize_sync()
        self._update_callbacks: list[Callable[[], None]] = []

        self._staged_parts: list[SystemPromptPart | UserPromptPart] = []
        self.notifications = ACPNotifications(client=self.client, session_id=self.session_id)
        self.requests = ACPRequests(client=self.client, session_id=self.session_id)
        self.input_provider = ACPInputProvider(self)
        self.acp_env = ACPExecutionEnvironment(fs=self.fs, requests=self.requests, cwd=self.cwd)
        for agent in self.agent_pool.all_agents.values():
            if isinstance(agent, Agent | ACPAgent):
                agent.env = self.acp_env
            if isinstance(agent, Agent):
                # TODO: need to inject this info for ACP agents, too.
                agent.sys_prompts.prompts.append(self.get_cwd_context)  # pyright: ignore[reportArgumentType]
            if isinstance(agent, ACPAgent):

                async def permission_callback(
                    params: RequestPermissionRequest,
                ) -> RequestPermissionResponse:
                    # Reconstruct request with our session_id (not nested agent's session_id)
                    self.log.debug(
                        "Forwarding permission request",
                        original_session_id=params.session_id,
                        our_session_id=self.session_id,
                        tool_call_id=params.tool_call.tool_call_id,
                        tool_call_title=params.tool_call.title,
                        options=[o.option_id for o in (params.options or [])],
                    )
                    forwarded_request = RequestPermissionRequest(
                        session_id=self.session_id,  # Use llmling-agent ↔ Zed session_id
                        options=params.options,
                        tool_call=params.tool_call,
                    )
                    try:
                        response = await self.requests.client.request_permission(forwarded_request)
                        self.log.debug(
                            "Permission response received",
                            outcome_type=type(response.outcome).__name__,
                            outcome=response.outcome.outcome,
                            option_id=getattr(response.outcome, "option_id", None),
                        )
                    except Exception as exc:
                        self.log.exception("Permission forwarding failed", error=str(exc))
                        raise
                    else:
                        return response

                agent.acp_permission_callback = permission_callback
        self.log.info("Created ACP session", current_agent=self.current_agent_name)

    def _get_or_create_tool_state(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> ToolCallState:
        """Get existing tool call state or create a new one.

        Args:
            tool_call_id: Unique identifier for the tool call
            tool_name: Name of the tool being called
            tool_input: Input parameters for the tool

        Returns:
            ToolCallState instance (existing or newly created)
        """
        if tool_call_id not in self._tool_call_states:
            state = ToolCallState(
                notifications=self.notifications,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                title=generate_tool_title(tool_name, tool_input),
                kind=infer_tool_kind(tool_name),
                raw_input=tool_input,
            )
            self._tool_call_states[tool_call_id] = state
        return self._tool_call_states[tool_call_id]

    def _cleanup_tool_state(self, tool_call_id: str) -> None:
        """Remove tool call state after completion."""
        self._tool_call_states.pop(tool_call_id, None)
        self._current_tool_inputs.pop(tool_call_id, None)

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

    async def init_client_skills(self) -> None:
        """Discover and load skills from client-side .claude/skills directory.

        Adds the client's .claude/skills directory to the pool's skills manager,
        making those skills available to all agents via the SkillsTools toolset.

        We pass the filesystem directly to avoid fsspec trying to create a new
        ACPFileSystem instance without the required client/session_id parameters.
        """
        try:
            await self.agent_pool.skills.add_skills_directory(".claude/skills", fs=self.fs)
            skills = self.agent_pool.skills.list_skills()
            self.log.info("Collected client-side skills", skill_count=len(skills))
        except Exception as e:
            self.log.exception("Failed to discover client-side skills", error=e)

    @property
    def agent(self) -> Agent[ACPSession, str] | ACPAgent | AGUIAgent:
        """Get the currently active agent."""
        if self.current_agent_name in self.agent_pool.acp_agents:
            return self.agent_pool.acp_agents[self.current_agent_name]
        if self.current_agent_name in self.agent_pool.agui_agents:
            return self.agent_pool.agui_agents[self.current_agent_name]
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
        if agent_name not in self.agent_pool.all_agents:
            available = list(self.agent_pool.all_agents.keys())
            raise ValueError(f"Agent {agent_name!r} not found. Available: {available}")

        old_agent_name = self.current_agent_name
        self.current_agent_name = agent_name
        self.log.info("Switched agents", from_agent=old_agent_name, to_agent=agent_name)

        # Persist the agent switch via session manager
        if self.manager:
            await self.manager.update_session_agent(self.session_id, agent_name)

        await self.send_available_commands_update()

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
        """Add prompt parts to staging area."""
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
        self._cancelled = False
        contents = from_acp_content(content_blocks)
        self.log.debug("Converted content", content=contents)
        if not contents:
            self.log.warning("Empty prompt received")
            return "refusal"
        commands, non_command_content = split_commands(contents)
        async with self._task_lock:
            if commands:  # Process commands if found
                for command in commands:
                    self.log.info("Processing slash command", command=command)
                    await self.execute_slash_command(command)

                # If only commands, end turn
                if not non_command_content:
                    return "end_turn"

            self.log.debug("Processing prompt", content_items=len(non_command_content))
            event_count = 0
            self._current_tool_inputs.clear()  # Reset tool inputs for new stream

            try:  # Use the session's persistent input provider
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
                # Send error notification asynchronously to avoid blocking response
                self.acp_agent.tasks.create_task(
                    self._send_error_notification(f"❌ Agent error: {e}"),
                    name=f"agent_error_notification_{self.session_id}",
                )
                return "end_turn"
            else:
                return "end_turn"

    async def _send_error_notification(self, message: str) -> None:
        """Send error notification, with exception handling."""
        try:
            await self.notifications.send_agent_text(message)
        except Exception:
            self.log.exception("Failed to send error notification")

    async def handle_event(self, event: RichAgentStreamEvent[Any]) -> None:  # noqa: PLR0915
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

            case PartStartEvent(part=part):
                self.log.debug("Received unhandled PartStartEvent", part=part)

            # Tool call streaming delta - create/update state and start notification
            case PartDeltaEvent(delta=ToolCallPartDelta() as delta):
                if delta_part := delta.as_part():
                    tool_call_id = delta_part.tool_call_id
                    try:
                        tool_input = delta_part.args_as_dict()
                    except ValueError:
                        # Args still streaming, not valid JSON yet - skip this delta
                        pass
                    else:
                        self._current_tool_inputs[tool_call_id] = tool_input
                        # Create state and send initial notification
                        state = self._get_or_create_tool_state(
                            tool_call_id=tool_call_id,
                            tool_name=delta_part.tool_name,
                            tool_input=tool_input,
                        )
                        await state.start()

            # Tool call started - create/update state and start notification
            case FunctionToolCallEvent(part=part):
                tool_call_id = part.tool_call_id
                try:
                    tool_input = part.args_as_dict()
                except ValueError as e:
                    # Args might be malformed - use empty dict and log
                    self.log.warning(
                        "Failed to parse tool args", tool_name=part.tool_name, error=str(e)
                    )
                    tool_input = {}
                self._current_tool_inputs[tool_call_id] = tool_input
                # Create state and send initial notification
                state = self._get_or_create_tool_state(
                    tool_call_id=tool_call_id,
                    tool_name=part.tool_name,
                    tool_input=tool_input,
                )
                await state.start()

            # Tool completed successfully - update state and finalize
            case FunctionToolResultEvent(
                result=ToolReturnPart(content=content, tool_name=tool_name) as result,
                tool_call_id=tool_call_id,
            ):
                if isinstance(content, AsyncGenerator):
                    full_content = ""
                    async for chunk in content:
                        full_content += str(chunk)
                        # Stream progress through state
                        if tool_state := self._tool_call_states.get(tool_call_id):
                            await tool_state.update(status="in_progress", raw_output=chunk)

                    # Replace the AsyncGenerator with the full content to prevent errors
                    result.content = full_content
                    final_output = full_content
                else:
                    final_output = result.content

                # Complete tool call through state (preserves accumulated content/locations)
                if complete_state := self._tool_call_states.get(tool_call_id):
                    # Only add return value as content if no content was emitted during execution
                    if complete_state.content:
                        # Content already provided via progress events - just set raw_output
                        await complete_state.complete(raw_output=final_output)
                    else:
                        # No content yet - convert return value for display
                        converted_blocks = to_acp_content_blocks(final_output)
                        content_items = [
                            ContentToolCallContent(content=block) for block in converted_blocks
                        ]
                        await complete_state.complete(
                            raw_output=final_output,
                            content=content_items,
                        )
                self._cleanup_tool_state(tool_call_id)

            # Tool failed with retry - update state with error
            case FunctionToolResultEvent(
                result=RetryPromptPart(tool_name=tool_name) as result,
                tool_call_id=tool_call_id,
            ):
                error_message = result.model_response()
                if fail_state := self._tool_call_states.get(tool_call_id):
                    await fail_state.fail(error=error_message)
                self._cleanup_tool_state(tool_call_id)

            # Tool emits its own start event - update state with better title/content
            case ToolCallStartEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                title=title,
                kind=kind,
                locations=loc_items,
                raw_input=raw_input,
            ):
                self.log.debug(
                    "Tool call start event", tool_name=tool_name, tool_call_id=tool_call_id
                )
                # Get or create state (may already exist from FunctionToolCallEvent)
                state = self._get_or_create_tool_state(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_input=raw_input or {},
                )
                # Convert LocationContentItem objects to ACP format
                acp_locations = [
                    ToolCallLocation(path=loc.path, line=loc.line) for loc in loc_items
                ]
                # Update state with tool-provided details (better title, content, locations)
                await state.update(title=title, kind=kind, locations=acp_locations or None)

            # Tool progress event - update state with title and content
            case ToolCallProgressEvent(
                tool_call_id=tool_call_id,
                title=title,
                status=status,
                items=items,
                replace_content=replace_content,
            ) if tool_call_id and tool_call_id in self._tool_call_states:
                progress_state = self._tool_call_states[tool_call_id]
                self.log.debug("Progress event", tool_call_id=tool_call_id, title=title)

                # Convert items to ACP content
                from llmling_agent.agents.events import (
                    DiffContentItem,
                    FileContentItem,
                    LocationContentItem,
                    TerminalContentItem,
                    TextContentItem,
                )
                from llmling_agent_server.acp_server.syntax_detection import (
                    format_zed_code_block,
                )

                acp_content: list[Any] = []
                location_paths: list[str] = []

                for item in items:
                    match item:
                        case TerminalContentItem(terminal_id=tid):
                            acp_content.append(TerminalToolCallContent(terminal_id=tid))
                        case TextContentItem(text=text):
                            acp_content.append(ContentToolCallContent.text(text=text))
                        case FileContentItem(
                            content=file_content,
                            path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                        ):
                            # Format as Zed-compatible code block with clickable path
                            formatted = format_zed_code_block(
                                file_content, file_path, start_line, end_line
                            )
                            acp_content.append(ContentToolCallContent.text(text=formatted))
                            # Also add path to locations for "follow along" feature
                            location_paths.append(file_path)
                        case DiffContentItem(path=diff_path, old_text=old, new_text=new):
                            # Send diff via direct notification
                            await self.notifications.file_edit_progress(
                                tool_call_id=tool_call_id,
                                path=diff_path,
                                old_text=old or "",
                                new_text=new,
                                status=status,
                                changed_lines=[],
                            )
                        case LocationContentItem(path=loc_path):
                            location_paths.append(loc_path)

                await progress_state.update(
                    title=title,
                    status="in_progress",
                    content=acp_content if acp_content else None,
                    locations=location_paths if location_paths else None,
                    replace=replace_content,
                )

            case FinalResultEvent():
                self.log.debug("Final result received")

            case StreamCompleteEvent():
                pass

            case PlanUpdateEvent(entries=entries, tool_call_id=tool_call_id):
                acp_entries = [
                    PlanEntry(content=e.content, priority=e.priority, status=e.status)
                    for e in entries
                ]
                await self.notifications.update_plan(acp_entries)

            case _:
                self.log.debug("Unhandled event", event_type=type(event).__name__)

    async def close(self) -> None:
        """Close the session and cleanup resources."""
        self._current_tool_inputs.clear()
        self._tool_call_states.clear()

        try:
            # Remove cwd context callable from all agents
            for agent in self.agent_pool.agents.values():
                if self.get_cwd_context in agent.sys_prompts.prompts:
                    agent.sys_prompts.prompts.remove(self.get_cwd_context)  # pyright: ignore[reportArgumentType]

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
        if not isinstance(self.agent, Agent):
            return
        try:  # Get all prompts from the agent's ToolManager
            if all_prompts := await self.agent.tools.list_prompts():
                for prompt in all_prompts:
                    command = self.create_mcp_command(prompt)
                    self.command_store.register_command(command)
                self._notify_command_update()
                self.log.info("Registered MCP prompts as commands", prompt_count=len(all_prompts))
                await self.send_available_commands_update()  # Send updated command list to client
        except Exception:
            self.log.exception("Failed to register MCP prompts as commands")

    async def _register_prompt_hub_commands(self) -> None:
        """Register prompt hub prompts as slash commands."""
        manager = self.agent_pool.manifest.prompt_manager
        cmd_count = 0
        try:
            all_prompts = await manager.list_prompts()
            for provider_name, prompt_names in all_prompts.items():
                if not prompt_names:  # Skip empty providers
                    continue
                for prompt_name in prompt_names:
                    command = self.create_prompt_hub_command(provider_name, prompt_name, manager)
                    self.command_store.register_command(command)
                    cmd_count += 1

            if cmd_count > 0:
                self._notify_command_update()
                self.log.info("Registered hub prompts as slash commands", cmd_count=cmd_count)
                await self.send_available_commands_update()  # Send updated command list to client
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

        Filters commands based on current agent's node type compatibility.

        Returns:
            List of ACP AvailableCommand objects compatible with current node
        """
        all_commands = self.command_store.list_commands()
        current_node = self.agent
        # Filter commands by node compatibility
        compatible_commands = []
        for cmd in all_commands:
            cmd_cls = cmd if isinstance(cmd, type) else type(cmd)
            # Check if command supports current node type
            if issubclass(cmd_cls, NodeCommand) and not cmd_cls.supports_node(current_node):  # type: ignore[union-attr]
                continue
            compatible_commands.append(cmd)

        return [
            AvailableCommand.create(name=i.name, description=i.description, input_hint=i.usage)
            for i in compatible_commands
        ]

    @logfire.instrument(r"Execute Slash Command {command_text}")
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

        # Check if command supports current node type
        if cmd := self.command_store.get_command(command_name):
            cmd_cls = cmd if isinstance(cmd, type) else type(cmd)
            if issubclass(cmd_cls, NodeCommand) and not cmd_cls.supports_node(self.agent):  # type: ignore[union-attr]
                error_msg = f"❌ Command `/{command_name}` is not available for this node type"
                await self.notifications.send_agent_text(error_msg)
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
            # Send error notification asynchronously to avoid blocking
            self.acp_agent.tasks.create_task(
                self._send_error_notification(f"❌ Command error: {e}"),
                name=f"command_error_notification_{self.session_id}",
            )

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
            ctx: CommandContext[AgentContext],
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
                await ctx.print(f"✅ Prompt {prompt.name!r} staged ({staged_count} total parts)")

            except Exception as e:
                logger.exception("MCP prompt execution failed", prompt=prompt.name)
                await ctx.print(f"❌ Prompt error: {e}")

        usage_hint = (
            " ".join(f"<{arg['name']}>" for arg in prompt.arguments) if prompt.arguments else None
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
            ctx: CommandContext[Any],
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
                self.add_staged_parts([UserPromptPart(content=result)])
                # Send confirmation
                staged_count = self.get_staged_parts_count()
                await ctx.print(
                    f"✅ Prompt {name!r} from {provider} staged ({staged_count} total parts)"
                )

            except Exception as e:
                logger.exception("Prompt hub execution failed", prompt=name, provider=provider)
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
