"""Codex agent - wraps Codex app-server via JSON-RPC protocol."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self
from uuid import uuid4

import anyenv
from pydantic import TypeAdapter
from pydantic_ai import TextPartDelta
from pydantic_ai.usage import RequestUsage, RunUsage

from agentpool.agents.base_agent import BaseAgent
from agentpool.agents.codex_agent.codex_converters import (
    convert_codex_stream,
    mcp_config_to_codex,
    to_model_info,
    to_session_data,
    turns_to_chat_messages,
    user_content_to_codex,
)
from agentpool.agents.events import PartDeltaEvent, RunStartedEvent, StreamCompleteEvent
from agentpool.agents.exceptions import (
    AgentNotInitializedError,
    UnknownCategoryError,
    UnknownModeError,
)
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType

    from exxec import ExecutionEnvironment
    from pydantic_ai import UserContent
    from tokonomics.model_discovery.model_info import ModelInfo

    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory
    from agentpool.common_types import AnyEventHandlerType, MCPServerStatus, StrPath
    from agentpool.delegation import AgentPool
    from agentpool.hooks import AgentHooks
    from agentpool.messaging import MessageHistory
    from agentpool.models.codex_agents import CodexAgentConfig
    from agentpool.resource_providers import ResourceProvider
    from agentpool.sessions.models import SessionData
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig
    from codex_adapter import ApprovalPolicy, CodexClient, ReasoningEffort, SandboxMode
    from codex_adapter.events import CodexEvent


logger = get_logger(__name__)


class CodexAgent[TDeps = None, OutputDataT = str](BaseAgent[TDeps, OutputDataT]):
    """MessageNode that wraps a Codex app-server instance."""

    AGENT_TYPE: ClassVar = "codex"

    def __init__(
        self,
        *,
        deps_type: type[TDeps] | None = None,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        base_instructions: str | None = None,
        developer_instructions: str | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        env: ExecutionEnvironment | StrPath | None = None,
        input_provider: InputProvider | None = None,
        env_vars: dict[str, str] | None = None,
        output_type: type[OutputDataT] = str,  # type: ignore[assignment]
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        hooks: AgentHooks | None = None,
        session_id: str | None = None,
        toolsets: list[ResourceProvider] | None = None,
        approval_policy: ApprovalPolicy | None = None,
        sandbox: SandboxMode | None = None,
    ) -> None:
        """Initialize Codex agent.

        Args:
            name: Agent name
            deps_type: Type of dependencies for the agent
            description: Agent description
            display_name: Human-readable display name
            model: Model to use (e.g., "claude-3-5-sonnet-20241022")
            reasoning_effort: Reasoning effort level ("low", "medium", "high")
            base_instructions: Base system instructions for the session
            developer_instructions: Developer-provided instructions
            agent_pool: Agent pool for coordination
            enable_logging: Whether to enable database logging
            mcp_servers: MCP server configurations
            env: Execution environment
            input_provider: Provider for user input
            env_vars: Environment variables for the agent
            output_type: Output type for structured responses (default: str)
            event_handlers: Event handlers for this agent
            hooks: Agent hooks for pre/post tool execution
            session_id: Session/thread ID to resume on connect (avoids reconnect overhead)
            toolsets: Resource providers for tools to expose via MCP bridge
            approval_policy: Approval policy for tool execution
            sandbox: Sandbox mode for execution
        """
        from agentpool.mcp_server.tool_bridge import ToolManagerBridge
        from agentpool_config.mcp_server import BaseMCPServerConfig

        super().__init__(
            name=name or "codex",
            deps_type=deps_type,
            description=description,
            display_name=display_name,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            env=env,
            input_provider=input_provider,
            output_type=output_type,
            event_handlers=event_handlers,
            hooks=hooks,
        )

        # Codex settings
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._base_instructions = base_instructions
        self._developer_instructions = developer_instructions
        self._approval_policy: ApprovalPolicy = approval_policy or "never"
        self._sandbox = sandbox
        self._toolsets = toolsets or []
        self._env_vars = env_vars or {}
        # Client state
        self._client: CodexClient | None = None
        self._sdk_session_id: str | None = session_id

        # Process MCP servers
        if mcp_servers:
            processed: list[MCPServerConfig] = []
            for server in mcp_servers:
                if isinstance(server, str):
                    processed.append(BaseMCPServerConfig.from_string(server))
                else:
                    processed.append(server)
            self._external_mcp_servers = processed
        else:
            self._external_mcp_servers = []

        # Extra MCP servers in Codex format (e.g., tool bridge)
        self._extra_mcp_servers: list[tuple[str, Any]] = []
        # Track current settings (for when they change mid-session)
        self._current_model: str | None = model
        self._current_effort: ReasoningEffort | None = reasoning_effort
        self._current_sandbox: SandboxMode | None = sandbox
        self._current_turn_id: str | None = None
        # Populated by capture_metadata during streaming, read after stream completes
        self._token_usage_data: dict[str, int] | None = None
        # Pass injection_manager for mid-run injection support
        self._tool_bridge = ToolManagerBridge(node=self, injection_manager=self._injection_manager)

    @classmethod
    def from_config(
        cls,
        config: CodexAgentConfig,
        *,
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        deps_type: type[TDeps] | None = None,
    ) -> Self:
        """Create agent from configuration.

        All config values are extracted here and passed to the constructor.
        """
        from agentpool.models.manifest import AgentsManifest
        from agentpool.utils.result_utils import to_type

        # Resolve output type from config
        manifest = agent_pool.manifest if agent_pool else AgentsManifest()
        agent_output_type = config.output_type or str
        if isinstance(agent_output_type, str) and agent_output_type != "str":
            resolved_output_type = to_type(agent_output_type, manifest.responses)
        else:
            resolved_output_type = to_type(agent_output_type)

        # Merge config-level handlers with provided handlers
        config_handlers = config.get_event_handlers()
        merged_handlers: list[AnyEventHandlerType] = [*config_handlers, *(event_handlers or [])]
        # Extract toolsets from config
        return cls(
            # Identity
            name=config.name,
            deps_type=deps_type,
            description=config.description,
            display_name=config.display_name,
            # Codex settings
            model=config.model,
            env=config.get_execution_environment(),
            reasoning_effort=config.reasoning_effort,
            base_instructions=config.base_instructions,
            developer_instructions=config.developer_instructions,
            approval_policy=config.approval_policy,
            sandbox=config.sandbox,
            # MCP and toolsets
            mcp_servers=config.get_mcp_servers(),
            toolsets=config.get_tool_providers(),
            # Runtime
            event_handlers=merged_handlers or None,
            input_provider=input_provider,
            agent_pool=agent_pool,
            output_type=resolved_output_type,  # type: ignore[arg-type]
            hooks=config.hooks.get_agent_hooks() if config.hooks else None,
        )

    async def _setup_toolsets(self) -> None:
        """Setup toolsets and start the tool bridge."""
        if not self._toolsets:
            return

        # Add toolset providers to tool manager
        for provider in self._toolsets:
            self.tools.add_provider(provider)
        # Start bridge to expose tools via MCP
        await self._tool_bridge.start()
        # Add bridge's MCP server config to extra servers
        # get_codex_mcp_server_config returns (name, HttpMcpServer)
        bridge_config = self._tool_bridge.get_codex_mcp_server_config()
        self._extra_mcp_servers.append(bridge_config)

    async def __aenter__(self) -> Self:
        """Start Codex client and create or resume thread."""
        from codex_adapter import CodexClient

        await super().__aenter__()
        await self._setup_toolsets()
        # Collect MCP servers: extra (bridge) + configured servers
        # Build dict mapping server name -> McpServerConfig (Codex type)
        mcp_servers_dict = {}
        # Add extra MCP servers (e.g., tool bridge) - already in Codex format
        mcp_servers_dict.update(dict(self._extra_mcp_servers))
        # Add configured/external MCP servers (convert native -> Codex format)
        if self._external_mcp_servers:
            servers = [mcp_config_to_codex(c) for c in self._external_mcp_servers]
            mcp_servers_dict.update(dict(servers))
        # Create and connect client with MCP servers
        self._client = CodexClient(mcp_servers=mcp_servers_dict)
        await self._client.__aenter__()
        cwd = str(self.env.cwd or Path.cwd())

        # Resume existing session or start new thread
        if self._sdk_session_id:
            # Resume the specified thread
            response = await self._client.thread_resume(self._sdk_session_id)
            thread = response.thread
            self._sdk_session_id = thread.id
            self.log.info("Codex thread resumed", sdk_session_id=self._sdk_session_id, cwd=cwd)
            # Restore conversation history from resumed thread
            if thread.turns:
                chat_messages = turns_to_chat_messages(thread.turns)
                self.conversation.chat_messages.clear()
                self.conversation.chat_messages.extend(chat_messages)
                self.log.info("Restored conversation history", turn_count=len(thread.turns))
        else:
            # Start a new thread
            response = await self._client.thread_start(
                cwd=cwd,
                model=self._model,
                base_instructions=self._base_instructions,
                developer_instructions=self._developer_instructions,
                sandbox=self._current_sandbox,
                approval_policy=self._approval_policy,
            )
            self._sdk_session_id = response.thread.id
            self.log.info("Codex thread started", sdk_session_id=self._sdk_session_id, cwd=cwd)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up Codex client."""
        await self._cleanup()
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def get_mcp_server_info(self) -> dict[str, MCPServerStatus]:
        """Get MCP server status from connected Codex client.

        Queries live status including tools, resources, and auth when
        the client is connected. Falls back to config-based reporting.
        """
        from agentpool.common_types import MCPServerStatus

        result: dict[str, MCPServerStatus] = {}
        if self._client:
            try:
                response = await self._client.mcp_server_status_list()
            except Exception:  # noqa: BLE001
                pass
            else:
                for server in response.data:
                    result[server.name] = MCPServerStatus(
                        name=server.name,
                        status="connected" if server.tools else "disconnected",
                        server_name=server.name,
                    )
                return result
        # Fallback: report from config
        for name in self._extra_mcp_servers:
            if isinstance(name, tuple):
                result[name[0]] = MCPServerStatus(name=name[0], status="connected")
        return result

    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Stop tool bridge if it was started
        if self._tool_bridge._mcp is not None:
            await self._tool_bridge.stop()
        self._extra_mcp_servers.clear()
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                self.log.exception("Error closing Codex client")
            self._client = None
        self._sdk_session_id = None

    async def _stream_events(  # noqa: PLR0915
        self,
        prompts: list[UserContent],
        *,
        user_msg: ChatMessage[Any],
        message_history: MessageHistory,
        effective_parent_id: str | None,
        message_id: str | None = None,
        session_id: str | None = None,
        parent_id: str | None = None,
        input_provider: InputProvider | None = None,
        deps: TDeps | None = None,
        wait_for_connections: bool | None = None,
        store_history: bool = True,
    ) -> AsyncIterator[RichAgentStreamEvent[OutputDataT]]:
        """Stream events from Codex turn execution."""
        from agentpool.agents.events import PlanUpdateEvent
        from agentpool.messaging.messages import TokenCost
        from codex_adapter import TurnStartedData
        from codex_adapter.models import ThreadTokenUsageUpdatedData

        if not self._client or not self._sdk_session_id:
            raise AgentNotInitializedError

        input_items = user_content_to_codex(prompts)
        # Generate IDs if not provided
        run_id = str(uuid4())
        final_message_id = message_id or str(uuid4())
        final_session_id = session_id or self.session_id
        # Ensure session_id is set (should always be from base class)
        if final_session_id is None:
            raise ValueError("session_id must be set")
        yield RunStartedEvent(session_id=final_session_id, run_id=run_id)
        # Stream turn events with bridge context set
        accumulated_text: list[str] = []
        self._token_usage_data = None
        # Pass output type directly - adapter handles conversion to JSON schema
        output_schema = None if self._output_type is str else self._output_type

        async def capture_metadata(
            raw_events: AsyncIterator[CodexEvent],
        ) -> AsyncIterator[CodexEvent]:
            """Wrapper to capture token usage and turn_id before event conversion."""
            async for event in raw_events:
                # Capture turn_id for interrupt support
                if event.event_type == "turn/started" and isinstance(event.data, TurnStartedData):
                    self._current_turn_id = event.data.turn.id
                # Capture token usage
                elif event.event_type == "thread/tokenUsage/updated" and isinstance(
                    event.data, ThreadTokenUsageUpdatedData
                ):
                    usage = event.data.token_usage.last
                    self._token_usage_data = {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cache_read_tokens": usage.cached_input_tokens,
                        "reasoning_tokens": usage.reasoning_output_tokens,
                    }
                yield event

        try:
            async with self._tool_bridge.set_run_context(deps, input_provider, prompt=prompts):
                raw_stream = self._client.turn_stream(
                    self._sdk_session_id,
                    input_items,
                    model=self._current_model,
                    effort=self._current_effort,
                    approval_policy=self._approval_policy,
                    sandbox_policy=self._current_sandbox,
                    output_schema=output_schema,
                )
                # Wrap to capture metadata (turn_id, token usage), then convert
                async for native_event in convert_codex_stream(capture_metadata(raw_stream)):
                    yield native_event

                    # Handle plan updates - sync to pool.todos
                    if (
                        isinstance(native_event, PlanUpdateEvent)
                        and self.agent_pool
                        and self.agent_pool.todos
                    ):
                        # Replace all entries in pool.todos with Codex plan
                        self.agent_pool.todos.replace_all([
                            (e.content, e.priority, e.status) for e in native_event.entries
                        ])

                    # Accumulate text for final message
                    if isinstance(native_event, PartDeltaEvent) and isinstance(
                        native_event.delta, TextPartDelta
                    ):
                        accumulated_text.append(native_event.delta.content_delta)

        except Exception as e:
            self.log.exception("Error during Codex turn", error=str(e))
            raise
        finally:
            # Clear turn_id when turn completes or errors
            self._current_turn_id = None

        # Emit completion event
        final_text = "".join(accumulated_text)
        cost_info: TokenCost | None = None
        request_usage = RequestUsage()

        if self._token_usage_data:
            run_usage = RunUsage(
                input_tokens=self._token_usage_data.get("input_tokens", 0),
                output_tokens=self._token_usage_data.get("output_tokens", 0),
                cache_read_tokens=self._token_usage_data.get("cache_read_tokens", 0),
                cache_write_tokens=0,  # Codex doesn't provide cache write tokens
            )
            # TODO: Calculate actual cost - for now set to 0
            cost_info = TokenCost(token_usage=run_usage, total_cost=Decimal(0))
            request_usage = RequestUsage(
                input_tokens=self._token_usage_data.get("input_tokens", 0),
                output_tokens=self._token_usage_data.get("output_tokens", 0),
                cache_read_tokens=self._token_usage_data.get("cache_read_tokens", 0),
                cache_write_tokens=0,
            )

        # Parse structured output if output_type is not str
        final_content: OutputDataT
        if self._output_type is not str and self._output_type is not None:
            try:
                parsed = anyenv.load_json(final_text)
                final_content = TypeAdapter(self._output_type).validate_python(parsed)
            except (anyenv.JsonLoadError, ValueError) as e:
                msg = "Failed to parse structured output, returning raw text"
                self.log.warning(msg, error=str(e), output_type=self._output_type)
                final_content = final_text  # type: ignore[assignment]
        else:
            final_content = final_text  # type: ignore[assignment]

        complete_msg: ChatMessage[OutputDataT] = ChatMessage(
            content=final_content,
            role="assistant",
            message_id=final_message_id,
            session_id=final_session_id,
            parent_id=parent_id,
            cost_info=cost_info,
            usage=request_usage,
            model_name=self.model_name,
        )

        yield StreamCompleteEvent[OutputDataT](message=complete_msg)

    @property
    def model_name(self) -> str:
        """Get current model name."""
        return self._model or "unknown"

    def to_structured[NewOutputDataT](
        self,
        output_type: type[NewOutputDataT],
    ) -> CodexAgent[TDeps, NewOutputDataT]:
        """Configure agent for structured output.

        Codex supports structured output via output_schema parameter in turn_stream.
        This method sets the output type which will be converted to JSON schema
        and passed to Codex on each turn.

        Args:
            output_type: Pydantic model type for structured responses

        Returns:
            Self (mutates in place)
        """
        from agentpool.utils.result_utils import to_type

        self.log.debug("Setting result type", output_type=output_type)
        self._output_type = to_type(output_type)  # type: ignore[assignment]
        return self  # type: ignore[return-value]

    async def set_model(self, model: str) -> None:
        """Set the model for this agent."""
        await self._set_mode(model, "model")

    async def set_approval_policy(self, policy: ApprovalPolicy) -> None:
        """Set the approval policy for tool execution.

        Args:
            policy: Approval policy - "never", "on-request", "on-failure", or "untrusted"
        """
        self._approval_policy = policy
        self.log.info("Approval policy updated", policy=policy)

    async def _interrupt(self) -> None:
        """Call Codex turn_interrupt if there's an active turn."""
        if self._client and self._sdk_session_id and self._current_turn_id:
            try:
                await self._client.turn_interrupt(self._sdk_session_id, self._current_turn_id)
                self.log.info(
                    "Codex turn interrupted",
                    sdk_session_id=self._sdk_session_id,
                    turn_id=self._current_turn_id,
                )
            except Exception:
                self.log.exception("Failed to interrupt Codex turn")

    async def get_available_models(self) -> list[ModelInfo] | None:
        """Get available models from Codex server.

        Returns:
            List of tokonomics ModelInfo for available models, or None if not connected
        """
        if not self._client:
            self.log.warning("Cannot get models: client not connected")
            return None

        try:
            models = [to_model_info(i) for i in await self._client.model_list()]
        except Exception:
            self.log.exception("Failed to fetch models from Codex")
            return None
        else:
            return models

    async def get_modes(self) -> list[ModeCategory]:
        """Get available mode categories for Codex agent (approval poliy, effort, model)."""
        from agentpool.agents.codex_agent.static_info import (
            EFFORT_MODES,
            POLICY_MODES,
            SANDBOX_MODES,
        )
        from agentpool.agents.modes import ModeCategory, ModeInfo

        categories = [
            ModeCategory(
                id="mode",
                name="Tool Approval",
                available_modes=POLICY_MODES,
                current_mode_id=self._approval_policy,
                category="mode",
            ),
            ModeCategory(
                id="thought_level",
                name="Reasoning Effort",
                available_modes=EFFORT_MODES,
                current_mode_id=self._current_effort or "medium",
                category="thought_level",
            ),
            ModeCategory(
                id="sandbox",
                name="Sandbox Mode",
                available_modes=SANDBOX_MODES,
                current_mode_id=self._current_sandbox or "workspace-write",
                category="other",
            ),
        ]
        if models := await self.get_available_models():
            model_modes = [
                ModeInfo(
                    id=m.id,
                    name=m.name or m.id,
                    description=m.description or "",
                    category_id="model",
                )
                for m in models
            ]
            categories.append(
                ModeCategory(
                    id="model",
                    name="Model",
                    available_modes=model_modes,
                    current_mode_id=self._current_model or self._model or "",
                    category="model",
                )
            )
        return categories

    async def _set_mode(self, mode_id: str, category_id: str) -> None:
        """Handle approval_policy, reasoning_effort, and model mode switching."""
        from agentpool.agents.modes import ConfigOptionChanged

        if category_id == "mode":
            valid_policies = ["never", "on-request", "on-failure", "untrusted"]
            if mode_id not in valid_policies:
                raise UnknownModeError(mode_id, valid_policies)
            self._approval_policy = mode_id  # type: ignore[assignment]
        elif category_id == "thought_level":
            valid_efforts = ["low", "medium", "high", "xhigh"]
            if mode_id not in valid_efforts:
                raise UnknownModeError(mode_id, valid_efforts)
            self._current_effort = mode_id  # type: ignore[assignment]
        elif category_id == "model":
            self._current_model = mode_id
        elif category_id == "sandbox":
            valid_sandboxes = [
                "read-only",
                "workspace-write",
                "danger-full-access",
                "external-sandbox",
            ]
            if mode_id not in valid_sandboxes:
                raise UnknownModeError(mode_id, valid_sandboxes)
            self._current_sandbox = mode_id  # type: ignore[assignment]
        else:
            raise UnknownCategoryError(category_id)
        self.log.info("Config option changed", category=category_id, value=mode_id)
        change = ConfigOptionChanged(config_id=category_id, value_id=mode_id)
        await self.state_updated.emit(change)

    async def list_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[SessionData]:
        """List threads ("sessions") from Codex server."""
        if not self._client:
            return []
        try:
            response = await self._client.thread_list(limit=limit)
        except Exception:
            self.log.exception("Failed to list Codex threads")
            return []
        else:
            cwd = self.env.cwd or str(Path.cwd())
            result = [to_session_data(i, agent_name=self.name, cwd=cwd) for i in response.data]
            # Apply cwd filter (Codex doesn't support cwd filter in request)
            if cwd is not None:
                result = [s for s in result if s.cwd == cwd]
            return result

    async def load_session(self, session_id: str) -> SessionData | None:
        """Load and resume a thread from Codex server.

        Resumes the specified thread on the Codex server, making it the active thread
        for this agent. The conversation history is managed by the Codex server.

        Args:
            session_id: Thread ID to resume

        Returns:
            SessionData if thread was resumed successfully, None otherwise
        """
        if not self._client:
            self.log.error("Cannot load session: Codex client not initialized")
            return None

        try:
            response = await self._client.thread_resume(session_id)
        except Exception:
            self.log.exception("Failed to resume Codex thread", session_id=session_id)
            return None
        # Update current thread ID
        thread = response.thread
        self._sdk_session_id = thread.id
        self.log.info("Thread resumed from Codex server", sdk_session_id=thread.id)
        # Convert turns to ChatMessages and populate conversation
        if thread.turns:
            chat_messages = turns_to_chat_messages(thread.turns)
            self.conversation.chat_messages.clear()
            self.conversation.chat_messages.extend(chat_messages)
            self.log.info(
                "Restored conversation history",
                session_id=session_id,
                turn_count=len(thread.turns),
                message_count=len(chat_messages),
            )
        cwd = self.env.cwd or str(Path.cwd())
        return to_session_data(thread, agent_name=self.name, cwd=cwd)
