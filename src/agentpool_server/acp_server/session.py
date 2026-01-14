"""ACP (Agent Client Protocol) session management for agentpool.

This module provides session lifecycle management, state tracking, and coordination
between agents and ACP clients through the JSON-RPC protocol.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Any, Literal

import anyio
from exxec.acp_provider import ACPExecutionEnvironment
import logfire
from pydantic_ai import UsageLimitExceeded, UserPromptPart
from slashed import Command, CommandStore
from tokonomics.model_discovery.model_info import ModelInfo

from acp import RequestPermissionRequest
from acp.acp_requests import ACPRequests
from acp.filesystem import ACPFileSystem
from acp.notifications import ACPNotifications
from acp.schema import AvailableCommand, ClientCapabilities, SessionNotification
from agentpool import Agent, AgentContext  # noqa: TC001
from agentpool.agents import SlashedAgent
from agentpool.agents.acp_agent import ACPAgent
from agentpool.agents.modes import ModeInfo
from agentpool.log import get_logger
from agentpool_commands import get_commands
from agentpool_commands.base import NodeCommand
from agentpool_server.acp_server.converters import (
    convert_acp_mcp_server_to_config,
    from_acp_content,
)
from agentpool_server.acp_server.event_converter import ACPEventConverter
from agentpool_server.acp_server.input_provider import ACPInputProvider


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic_ai import SystemPromptPart, UserContent
    from slashed import CommandContext

    from acp import Client, RequestPermissionResponse
    from acp.schema import (
        AvailableCommandsUpdate,
        ConfigOptionUpdate,
        ContentBlock,
        Implementation,
        McpServer,
        StopReason,
    )
    from agentpool import AgentPool
    from agentpool.agents import AGUIAgent
    from agentpool.agents.claude_code_agent import ClaudeCodeAgent
    from agentpool.prompts.manager import PromptManager
    from agentpool.prompts.prompts import MCPClientPrompt
    from agentpool_server.acp_server.acp_agent import AgentPoolACPAgent
    from agentpool_server.acp_server.session_manager import ACPSessionManager

logger = get_logger(__name__)
SLASH_PATTERN = re.compile(r"^/([\w-]+)(?:\s+(.*))?$")

# Zed-specific instructions for code references
ZED_CLIENT_PROMPT = """\
## Code References

When referencing code locations in responses, use markdown links with `file://` URLs:

- **File**: `[filename](file:///absolute/path/to/file.py)`
- **Line range**: `[filename#L10-25](file:///absolute/path/to/file.py#L10:25)`
- **Single line**: `[filename#L10](file:///absolute/path/to/file.py#L10:10)`
- **Directory**: `[dirname/](file:///absolute/path/to/dir/)`

Line range format is `#L<start>:<end>` (1-based, inclusive).

Use these clickable references instead of inline code blocks when pointing to specific \
code locations. For showing actual code content, still use fenced code blocks.

## Zed-specific URLs

In addition to `file://` URLs, these `zed://` URLs work in the agent context:

- **File reference**: `[text](zed:///agent/file?path=/absolute/path/to/file.py)`
- **Selection**: `[text](zed:///agent/selection?path=/absolute/path/to/file.py#L10:25)`
- **Symbol**: `[text](zed:///agent/symbol/function_name?path=/absolute/path/to/file.py#L10:25)`
- **Directory**: `[text](zed:///agent/directory?path=/absolute/path/to/dir)`

Query params must be URL-encoded (spaces → `%20`). Paths must be absolute.
"""


def _is_slash_command(text: str) -> bool:
    """Check if text starts with a slash command."""
    return bool(SLASH_PATTERN.match(text.strip()))


def split_commands(
    contents: Sequence[UserContent],
    command_store: CommandStore,
) -> tuple[list[str], list[UserContent]]:
    """Split content into local slash commands and pass-through content.

    Only commands that exist in the local command_store are extracted.
    Remote commands (from nested ACP agents) stay in non_command_content
    so they flow through to the agent and reach the nested server.
    """
    commands: list[str] = []
    non_command_content: list[UserContent] = []
    for item in contents:
        # Check if this is a LOCAL command we handle
        if (
            isinstance(item, str)
            and _is_slash_command(item)
            and (match := SLASH_PATTERN.match(item.strip()))
            and command_store.get_command(match.group(1))
        ):
            commands.append(item.strip())
        else:
            # Not a local command - pass through (may be remote command or regular text)
            non_command_content.append(item)
    return commands, non_command_content


@dataclass
class StagedContent:
    """Buffer for prompt parts to be injected into the next agent call.

    This allows commands (like /fetch-repo, /git-diff) to stage content that will
    be automatically included in the next prompt sent to the agent.
    """

    _parts: list[SystemPromptPart | UserPromptPart] = field(default_factory=list)

    def add(self, parts: list[SystemPromptPart | UserPromptPart]) -> None:
        """Add prompt parts to the staging area."""
        self._parts.extend(parts)

    def add_text(self, content: str) -> None:
        """Add text content to the staging area as a UserPromptPart."""
        self._parts.append(UserPromptPart(content=content))

    def consume(self) -> list[SystemPromptPart | UserPromptPart]:
        """Return all staged parts and clear the buffer."""
        parts = self._parts.copy()
        self._parts.clear()
        return parts

    def consume_as_text(self) -> str | None:
        """Return all staged content as a single string and clear the buffer.

        Returns:
            Combined text content, or None if nothing staged.
        """
        if not self._parts:
            return None
        texts = [part.content for part in self._parts if isinstance(part.content, str)]
        self._parts.clear()
        return "\n\n".join(texts) if texts else None

    def __len__(self) -> int:
        """Return count of staged parts."""
        return len(self._parts)


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

    acp_agent: AgentPoolACPAgent
    """ACP agent instance for capability tools"""

    mcp_servers: Sequence[McpServer] | None = None
    """Optional MCP server configurations"""

    client_capabilities: ClientCapabilities = field(default_factory=ClientCapabilities)
    """Client capabilities for tool registration"""

    client_info: Implementation | None = None
    """Client implementation info (name, version, title)"""

    manager: ACPSessionManager | None = None
    """Session manager for managing sessions. Used for session management commands."""

    subagent_display_mode: Literal["inline", "tool_box"] = "tool_box"
    """How to display subagent output:
    - 'inline': Subagent output flows into main message stream
    - 'tool_box': Subagent output contained in the tool call's progress box (default)
    """

    def __post_init__(self) -> None:
        """Initialize session state and set up providers."""
        from agentpool_server.acp_server.commands import get_commands as get_acp_commands

        self.mcp_servers = self.mcp_servers or []
        self.log = logger.bind(session_id=self.session_id)
        self._task_lock = asyncio.Lock()
        self._cancelled = False
        self._current_converter: ACPEventConverter | None = None
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
        self.command_store = CommandStore(commands=cmds)
        self.command_store._initialize_sync()
        self._update_callbacks: list[Callable[[], None]] = []
        self._remote_commands: list[AvailableCommand] = []  # Commands from nested ACP agents
        self.staged_content = StagedContent()
        # Inject Zed-specific instructions if client is Zed
        if self.client_info and self.client_info.name and "zed" in self.client_info.name.lower():
            self.staged_content.add_text(ZED_CLIENT_PROMPT)
        self.notifications = ACPNotifications(client=self.client, session_id=self.session_id)
        self.requests = ACPRequests(client=self.client, session_id=self.session_id)
        self.input_provider = ACPInputProvider(self)
        self.acp_env = ACPExecutionEnvironment(fs=self.fs, requests=self.requests, cwd=self.cwd)
        for agent in self.agent_pool.all_agents.values():
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
                        session_id=self.session_id,  # Use agentpool ↔ Zed session_id
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

            # Subscribe to state change signal for all agents
            agent.state_updated.connect(self._on_state_updated)
        self.log.info("Created ACP session", current_agent=self.current_agent_name)

    async def _on_state_updated(
        self, state: ModeInfo | ModelInfo | AvailableCommandsUpdate | ConfigOptionUpdate
    ) -> None:
        """Handle state update signal from agent - forward to ACP client."""
        from acp.schema import (
            AvailableCommandsUpdate as ACPAvailableCommandsUpdate,
            ConfigOptionUpdate as ACPConfigOptionUpdate,
            CurrentModelUpdate,
            CurrentModeUpdate,
            SessionNotification,
        )

        update: CurrentModeUpdate | CurrentModelUpdate | ACPConfigOptionUpdate
        match state:
            case ModeInfo(id=mode_id):
                update = CurrentModeUpdate(current_mode_id=mode_id)
                self.log.debug("Forwarding mode change to client", mode_id=mode_id)
            case ModelInfo(id=model_id):
                update = CurrentModelUpdate(current_model_id=model_id)
                self.log.debug("Forwarding model change to client", model_id=model_id)
            case ACPAvailableCommandsUpdate(available_commands=cmds):
                # Store remote commands and send merged list
                self._remote_commands = list(cmds)
                await self.send_available_commands_update()
                self.log.debug("Merged and sent commands update to client")
                return
            case ACPConfigOptionUpdate():
                update = state
                self.log.debug("Forwarding config option update to client")

        notification = SessionNotification(session_id=self.session_id, update=update)
        await self.client.session_update(notification)  # pyright: ignore[reportArgumentType]

    async def initialize(self) -> None:
        """Initialize async resources. Must be called after construction."""
        await self.acp_env.__aenter__()

    async def initialize_mcp_servers(self) -> None:
        """Initialize MCP servers if any are configured."""
        if not self.mcp_servers:
            return
        self.log.info("Initializing MCP servers", server_count=len(self.mcp_servers))
        cfgs = [convert_acp_mcp_server_to_config(s) for s in self.mcp_servers]
        # Add each MCP server to the current agent's MCP manager dynamically
        for cfg in cfgs:
            try:
                await self.agent.mcp.setup_server(cfg)
                self.log.info(
                    "Added MCP server to agent", server_name=cfg.name, agent=self.current_agent_name
                )
            except Exception:
                self.log.exception("Failed to setup MCP server", server_name=cfg.name)
                # Don't fail session creation, just log the error
        # Register MCP prompts as commands after all servers are added
        try:
            await self._register_mcp_prompts_as_commands()
        except Exception:
            self.log.exception("Failed to register MCP prompts as commands")

    async def init_project_context(self) -> None:
        """Load AGENTS.md/CLAUDE.md file and stage as initial context.

        The project context is staged as user message content rather than system prompts,
        which ensures it's available for all agent types and avoids timing issues with
        agent initialization.
        """
        if info := await self.requests.read_agent_rules(self.cwd):
            # Stage as user message to be prepended to first prompt
            self.staged_content.add_text(f"## Project Information\n\n{info}")

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
    def agent(
        self,
    ) -> (
        Agent[ACPSession, str]
        | ACPAgent[ACPSession]
        | AGUIAgent[ACPSession]
        | ClaudeCodeAgent[ACPSession]
    ):
        """Get the currently active agent."""
        if self.current_agent_name in self.agent_pool.agents:
            return self.agent_pool.get_agent(self.current_agent_name, deps_type=ACPSession)
        return self.agent_pool.all_agents[self.current_agent_name]  # type: ignore[return-value]

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

    async def cancel(self) -> None:
        """Cancel the current prompt turn.

        This actively interrupts the running agent by calling its interrupt() method,
        which handles protocol-specific cancellation (e.g., sending CancelNotification
        for ACP agents, calling SDK interrupt for ClaudeCodeAgent, etc.).

        Note:
            Tool call cleanup is handled in process_prompt() to avoid race conditions
            with the converter state being modified from multiple async contexts.
        """
        self._cancelled = True
        self.log.info("Session cancelled, interrupting agent")

        # Actively interrupt the agent's stream
        try:
            await self.agent.interrupt()
        except Exception:
            self.log.exception("Failed to interrupt agent")

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
        self._cancelled = False
        contents = from_acp_content(content_blocks)
        self.log.debug("Converted content", content=contents)
        if not contents:
            self.log.warning("Empty prompt received")
            return "refusal"
        commands, non_command_content = split_commands(contents, self.command_store)
        async with self._task_lock:
            if commands:  # Process commands if found
                for command in commands:
                    self.log.info("Processing slash command", command=command)
                    await self.execute_slash_command(command)

                # If only commands, end turn
                if not non_command_content:
                    return "end_turn"

            # Consume any staged content and prepend to the prompt
            staged = self.staged_content.consume_as_text()
            all_content = [staged, *non_command_content] if staged else list(non_command_content)
            self.log.debug(
                "Processing prompt",
                content_items=len(non_command_content),
                has_staged=staged is not None,
            )
            event_count = 0
            # Create a new event converter for this prompt
            converter = ACPEventConverter(subagent_display_mode=self.subagent_display_mode)
            self._current_converter = converter  # Track for cancellation

            try:  # Use the session's persistent input provider
                async for event in self.agent.run_stream(
                    *all_content,
                    input_provider=self.input_provider,
                    deps=self,
                    conversation_id=self.session_id,  # Tie agent conversation to ACP session
                ):
                    if self._cancelled:
                        self.log.info("Cancelled during event loop, cleaning up tool calls")
                        # Send cancellation notifications for any pending tool calls
                        # This happens in the same async context as the converter
                        async for cancel_update in converter.cancel_pending_tools():
                            notification = SessionNotification(
                                session_id=self.session_id, update=cancel_update
                            )
                            await self.client.session_update(notification)  # pyright: ignore[reportArgumentType]
                        # CRITICAL: Allow time for client to process tool completion notifications
                        # before sending PromptResponse. Without this delay, the client may receive
                        # and process the PromptResponse before the tool notifications, causing UI
                        # state desync where subsequent prompts appear stuck/unresponsive.
                        # This is needed because even though send() awaits the write, the client
                        # may process messages asynchronously or out of order.
                        await anyio.sleep(0.05)
                        self._current_converter = None
                        return "cancelled"

                    event_count += 1
                    async for update in converter.convert(event):
                        notification = SessionNotification(
                            session_id=self.session_id, update=update
                        )
                        await self.client.session_update(notification)  # pyright: ignore[reportArgumentType]
                    # Yield control to allow notifications to be sent immediately
                    await anyio.sleep(0.01)
                self.log.info("Streaming finished", events_processed=event_count)

            except asyncio.CancelledError:
                # Task was cancelled (e.g., via interrupt()) - return proper stop reason
                # This is critical: CancelledError doesn't inherit from Exception,
                # so we must catch it explicitly to send the PromptResponse
                self.log.info("Stream cancelled via CancelledError, cleaning up tool calls")
                # Send cancellation notifications for any pending tool calls
                async for cancel_update in converter.cancel_pending_tools():
                    notification = SessionNotification(
                        session_id=self.session_id, update=cancel_update
                    )
                    await self.client.session_update(notification)  # pyright: ignore[reportArgumentType]
                # CRITICAL: Allow time for client to process tool completion notifications
                # before sending PromptResponse. See comment in cancellation branch above.
                await anyio.sleep(0.05)
                self._current_converter = None
                return "cancelled"
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
                self._current_converter = None  # Clear converter reference
                self.log.exception("Error during streaming")
                # Send error notification asynchronously to avoid blocking response
                self.acp_agent.tasks.create_task(
                    self._send_error_notification(f"❌ Agent error: {e}"),
                    name=f"agent_error_notification_{self.session_id}",
                )
                return "end_turn"
            else:
                # Title generation is now handled automatically by log_conversation
                self._current_converter = None  # Clear converter reference
                return "end_turn"

    async def _send_error_notification(self, message: str) -> None:
        """Send error notification, with exception handling."""
        if self._cancelled:
            return
        try:
            await self.notifications.send_agent_text(message)
        except Exception:
            self.log.exception("Failed to send error notification")

    async def close(self) -> None:
        """Close the session and cleanup resources."""
        try:
            await self.acp_env.__aexit__(None, None, None)
        except Exception:
            self.log.exception("Error closing acp_env")

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
        """Send current available commands to client.

        Merges local commands from command_store with any remote commands
        from nested ACP agents.
        """
        try:
            commands = self.get_acp_commands()  # Local commands
            commands.extend(self._remote_commands)  # Merge remote commands
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
        current_node = self.agent
        # Filter commands by node compatibility
        compatible_commands = []
        for cmd in self.command_store.list_commands():
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

        # Create context with session data
        agent_context = self.agent.get_context(data=self)
        cmd_ctx = self.command_store.create_context(
            data=agent_context,
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
        """Register callback for command updates."""
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
            # Map positional args to prompt parameter names
            result = {
                prompt.arguments[i]["name"]: arg_value
                for i, arg_value in enumerate(args)
                if i < len(prompt.arguments)
            } | kwargs
            try:  # Get prompt components
                components = await prompt.get_components(result or None)
                self.staged_content.add(components)
                # Send confirmation
                staged_count = len(self.staged_content)
                await ctx.print(f"✅ Prompt {prompt.name!r} staged ({staged_count} total parts)")

            except Exception as e:
                logger.exception("MCP prompt execution failed", prompt=prompt.name)
                await ctx.print(f"❌ Prompt error: {e}")

        usage = " ".join(f"<{i['name']}>" for i in args) if (args := prompt.arguments) else None
        return Command.from_raw(
            execute_prompt,
            name=prompt.name,
            description=prompt.description or f"MCP prompt: {prompt.name}",
            category="mcp",
            usage=usage,
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
                self.staged_content.add([UserPromptPart(content=result)])
                # Send confirmation
                staged_count = len(self.staged_content)
                await ctx.print(
                    f"✅ Prompt {name!r} from {provider} staged ({staged_count} total parts)"
                )

            except Exception as e:
                logger.exception("Prompt hub execution failed", prompt=name, provider=provider)
                await ctx.print(f"❌ Prompt error: {e}")

        # Create command name - prefix with provider if not builtin
        command_name = f"{provider}_{name}" if provider != "builtin" else name

        return Command.from_raw(
            execute_prompt,
            name=command_name,
            description=f"Prompt hub: {provider}:{name}",
            category="prompts",
            usage="[key=value ...]",  # Generic since we don't have parameter schemas
        )
