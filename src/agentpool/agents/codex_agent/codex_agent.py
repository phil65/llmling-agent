"""Codex agent - wraps Codex app-server via JSON-RPC protocol."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self
from uuid import uuid4

from pydantic import TypeAdapter
from pydantic_ai import TextPartDelta
from pydantic_ai.usage import RequestUsage, RunUsage

from agentpool.agents.base_agent import BaseAgent
from agentpool.agents.events import PartDeltaEvent, RunStartedEvent, StreamCompleteEvent
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType

    from anyenv import MultiEventHandler
    from exxec import ExecutionEnvironment
    from pydantic_ai import UserContent
    from tokonomics.model_discovery.model_info import ModelInfo

    from agentpool.agents import AgentContext
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory
    from agentpool.common_types import BuiltinEventHandlerType, IndividualEventHandler
    from agentpool.delegation import AgentPool
    from agentpool.hooks import AgentHooks
    from agentpool.messaging import MessageHistory
    from agentpool.models.codex_agents import CodexAgentConfig
    from agentpool.resource_providers import ResourceProvider
    from agentpool.sessions import SessionData
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig
    from agentpool_config.nodes import ToolConfirmationMode
    from codex_adapter import ApprovalPolicy, CodexClient, ReasoningEffort, SandboxMode
    from codex_adapter.events import CodexEvent


logger = get_logger(__name__)


class CodexAgent[TDeps = None, OutputDataT = str](BaseAgent[TDeps, OutputDataT]):
    """MessageNode that wraps a Codex app-server instance."""

    AGENT_TYPE: ClassVar = "codex"

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        cwd: str | Path | None = None,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        base_instructions: str | None = None,
        developer_instructions: str | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        env: ExecutionEnvironment | None = None,
        input_provider: InputProvider | None = None,
        output_type: type[OutputDataT] = str,  # type: ignore[assignment]
        tool_confirmation_mode: ToolConfirmationMode = "always",
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        hooks: AgentHooks | None = None,
        session_id: str | None = None,
        toolsets: list[ResourceProvider] | None = None,
        approval_policy: ApprovalPolicy | None = None,
        sandbox: SandboxMode | None = None,
    ) -> None:
        """Initialize Codex agent.

        Args:
            name: Agent name
            description: Agent description
            display_name: Human-readable display name
            cwd: Working directory for Codex
            model: Model to use (e.g., "claude-3-5-sonnet-20241022")
            reasoning_effort: Reasoning effort level ("low", "medium", "high")
            base_instructions: Base system instructions for the session
            developer_instructions: Developer-provided instructions
            agent_pool: Agent pool for coordination
            enable_logging: Whether to enable database logging
            mcp_servers: MCP server configurations
            env: Execution environment
            input_provider: Provider for user input
            output_type: Output type for structured responses (default: str)
            tool_confirmation_mode: Tool confirmation behavior
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
            description=description,
            display_name=display_name,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            env=env,
            input_provider=input_provider,
            output_type=output_type,
            tool_confirmation_mode=tool_confirmation_mode,
            event_handlers=event_handlers,
            hooks=hooks,
        )

        # Codex settings
        self._cwd = cwd
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._base_instructions = base_instructions
        self._developer_instructions = developer_instructions
        self._approval_policy: ApprovalPolicy = approval_policy or "never"
        self._sandbox = sandbox
        self._toolsets = toolsets or []

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

        self._tool_bridge = ToolManagerBridge(node=self, server_name=f"agentpool-{self.name}-tools")

    @classmethod
    def from_config(
        cls,
        config: CodexAgentConfig,
        *,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
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
        merged_handlers: list[IndividualEventHandler | BuiltinEventHandlerType] = [
            *config_handlers,
            *(event_handlers or []),
        ]

        # Extract toolsets from config
        toolsets = config.get_tool_providers() if config.tools else []

        return cls(
            # Identity
            name=config.name,
            description=config.description,
            display_name=config.display_name,
            # Codex settings
            cwd=config.cwd,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            base_instructions=config.base_instructions,
            developer_instructions=config.developer_instructions,
            approval_policy=config.approval_policy,
            sandbox=config.sandbox,
            # MCP and toolsets
            mcp_servers=config.get_mcp_servers(),
            toolsets=toolsets,
            # Runtime
            event_handlers=merged_handlers or None,
            input_provider=input_provider,
            agent_pool=agent_pool,
            output_type=resolved_output_type,  # type: ignore[arg-type]
            hooks=config.hooks.get_agent_hooks() if config.hooks else None,
        )

    def get_context(
        self,
        data: Any = None,
        input_provider: InputProvider | None = None,
    ) -> AgentContext:
        """Get agent context.

        Args:
            data: Optional context data
            input_provider: Optional input provider override

        Returns:
            Agent context with configuration
        """
        from agentpool.agents.context import AgentContext

        return AgentContext(
            node=self,
            pool=self.agent_pool,
            input_provider=input_provider or self._input_provider,
            model_name=self.model_name,
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
        from agentpool.agents.codex_agent.codex_converters import (
            mcp_config_to_codex,
            turns_to_chat_messages,
        )
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
        cwd = str(self._cwd or Path.cwd())

        # Resume existing session or start new thread
        if self._sdk_session_id:
            # Resume the specified thread
            response = await self._client.thread_resume(self._sdk_session_id)
            thread = response.thread
            self._sdk_session_id = thread.id
            self.log.info("Codex thread resumed", thread_id=self._sdk_session_id, cwd=cwd)
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
            )
            self._sdk_session_id = response.thread.id
            self.log.info("Codex thread started", thread_id=self._sdk_session_id, cwd=cwd)
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
        conversation_id: str | None = None,
        parent_id: str | None = None,
        input_provider: InputProvider | None = None,
        deps: TDeps | None = None,
        event_handlers: MultiEventHandler[IndividualEventHandler],
        wait_for_connections: bool | None = None,
        store_history: bool = True,
    ) -> AsyncIterator[RichAgentStreamEvent[OutputDataT]]:
        """Stream events from Codex turn execution."""
        from agentpool.agents.codex_agent.codex_converters import (
            convert_codex_stream,
            user_content_to_codex,
        )
        from agentpool.agents.events import PlanUpdateEvent
        from agentpool.messaging.messages import TokenCost
        from codex_adapter.models import ThreadTokenUsageUpdatedData

        if not self._client or not self._sdk_session_id:
            raise RuntimeError("Codex client not initialized")

        input_items = user_content_to_codex(prompts)
        # Generate IDs if not provided
        run_id = str(uuid4())
        final_message_id = message_id or str(uuid4())
        final_conversation_id = conversation_id or self.conversation_id
        # Ensure conversation_id is set (should always be from base class)
        if final_conversation_id is None:
            raise ValueError("conversation_id must be set")
        run_started = RunStartedEvent(thread_id=final_conversation_id, run_id=run_id)
        await event_handlers(None, run_started)
        yield run_started
        # Stream turn events with bridge context set
        accumulated_text: list[str] = []
        token_usage_data: dict[str, int] | None = None
        # Pass output type directly - adapter handles conversion to JSON schema
        output_schema = None if self._output_type is str else self._output_type

        async def capture_metadata(
            raw_events: AsyncIterator[CodexEvent],
        ) -> AsyncIterator[CodexEvent]:
            """Wrapper to capture token usage and turn_id before event conversion."""
            nonlocal token_usage_data
            from codex_adapter import TurnStartedData

            async for event in raw_events:
                # Capture turn_id for interrupt support
                if event.event_type == "turn/started" and isinstance(event.data, TurnStartedData):
                    self._current_turn_id = event.data.turn.id
                # Capture token usage
                elif event.event_type == "thread/tokenUsage/updated" and isinstance(
                    event.data, ThreadTokenUsageUpdatedData
                ):
                    usage = event.data.token_usage.last
                    token_usage_data = {
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
                    await event_handlers(None, native_event)
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

        if token_usage_data:
            run_usage = RunUsage(
                input_tokens=token_usage_data.get("input_tokens", 0),
                output_tokens=token_usage_data.get("output_tokens", 0),
                cache_read_tokens=token_usage_data.get("cache_read_tokens", 0),
                cache_write_tokens=0,  # Codex doesn't provide cache write tokens
            )
            # TODO: Calculate actual cost - for now set to 0
            cost_info = TokenCost(token_usage=run_usage, total_cost=Decimal(0))
            request_usage = RequestUsage(
                input_tokens=token_usage_data.get("input_tokens", 0),
                output_tokens=token_usage_data.get("output_tokens", 0),
                cache_read_tokens=token_usage_data.get("cache_read_tokens", 0),
                cache_write_tokens=0,
            )

        # Parse structured output if output_type is not str
        final_content: OutputDataT
        if self._output_type is not str and self._output_type is not None:
            try:
                parsed = json.loads(final_text)
                adapter = TypeAdapter(self._output_type)
                final_content = adapter.validate_python(parsed)
            except (json.JSONDecodeError, ValueError) as e:
                msg = "Failed to parse structured output, returning raw text"
                self.log.warning(msg, error=str(e), output_type=self._output_type)
                final_content = final_text  # type: ignore[assignment]
        else:
            final_content = final_text  # type: ignore[assignment]

        complete_msg: ChatMessage[OutputDataT] = ChatMessage(
            content=final_content,
            role="assistant",
            message_id=final_message_id,
            conversation_id=final_conversation_id,
            parent_id=parent_id,
            cost_info=cost_info,
            usage=request_usage,
            model_name=self.model_name,
        )

        complete_event: StreamCompleteEvent[OutputDataT] = StreamCompleteEvent(message=complete_msg)
        await event_handlers(None, complete_event)
        yield complete_event

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

    async def set_tool_confirmation_mode(self, mode: ToolConfirmationMode) -> None:
        """Set tool confirmation mode."""
        self.tool_confirmation_mode = mode
        self.log.info("Tool confirmation mode updated", mode=mode)

    async def _interrupt(self) -> None:
        """Call Codex turn_interrupt if there's an active turn."""
        if self._client and self._sdk_session_id and self._current_turn_id:
            try:
                await self._client.turn_interrupt(self._sdk_session_id, self._current_turn_id)
                self.log.info(
                    "Codex turn interrupted",
                    thread_id=self._sdk_session_id,
                    turn_id=self._current_turn_id,
                )
            except Exception:
                self.log.exception("Failed to interrupt Codex turn")

    async def get_available_models(self) -> list[ModelInfo] | None:
        """Get available models from Codex server.

        Returns:
            List of tokonomics ModelInfo for available models, or None if not connected
        """
        from tokonomics.model_discovery.model_info import ModelInfo as TokModelInfo

        if not self._client:
            self.log.warning("Cannot get models: client not connected")
            return None

        try:
            models = []
            for model_data in await self._client.model_list():
                # Infer provider from model name
                model_id = model_data.model or model_data.id
                # Use display_name and description from API if available
                name = model_data.display_name or model_data.id
                desc = model_data.description or f"Model: {model_id}"
                info = TokModelInfo(
                    id=model_id,
                    name=name,
                    provider="openai",
                    description=desc,
                    id_override=model_id,
                )
                models.append(info)

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

        categories: list[ModeCategory] = []
        categories.append(
            ModeCategory(
                id="mode",
                name="Tool Approval",
                available_modes=POLICY_MODES,
                current_mode_id=self._approval_policy,
                category="mode",
            )
        )
        # Reasoning effort modes
        categories.append(
            ModeCategory(
                id="thought_level",
                name="Reasoning Effort",
                available_modes=EFFORT_MODES,
                current_mode_id=self._current_effort or "medium",
                category="thought_level",
            )
        )
        # Sandbox modes
        categories.append(
            ModeCategory(
                id="sandbox",
                name="Sandbox Mode",
                available_modes=SANDBOX_MODES,
                current_mode_id=self._current_sandbox or "workspace-write",
                category="other",
            )
        )
        # Model selection
        models = await self.get_available_models()
        if models:
            current_model = self._current_model or self._model or ""
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
                    current_mode_id=current_model,
                    category="model",
                )
            )
        return categories

    async def _set_mode(self, mode_id: str, category_id: str) -> None:
        """Handle approval_policy, reasoning_effort, and model mode switching."""
        from agentpool.agents.modes import ConfigOptionChanged

        if category_id == "mode":
            if mode_id not in ["never", "on-request", "on-failure", "untrusted"]:
                raise ValueError(f"Invalid approval policy: {mode_id}")
            self._approval_policy = mode_id  # type: ignore[assignment]
        elif category_id == "thought_level":
            if mode_id not in ["low", "medium", "high", "xhigh"]:
                raise ValueError(f"Invalid reasoning effort: {mode_id}")
            self._current_effort = mode_id  # type: ignore[assignment]
        elif category_id == "model":
            self._current_model = mode_id
        elif category_id == "sandbox":
            valid = ["read-only", "workspace-write", "danger-full-access", "external-sandbox"]
            if mode_id not in valid:
                raise ValueError(f"Invalid sandbox mode: {mode_id}. Valid: {valid}")
            self._current_sandbox = mode_id  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown category: {category_id}")
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
        from agentpool.sessions.models import SessionData

        if not self._client:
            return []
        try:
            response = await self._client.thread_list(limit=limit)
        except Exception:
            self.log.exception("Failed to list Codex threads")
            return []
        else:
            result: list[SessionData] = []
            for thread_data in response.data:
                created_at = datetime.fromtimestamp(thread_data.created_at, tz=UTC)
                session_data = SessionData(
                    session_id=thread_data.id,
                    agent_name=self.name,
                    cwd=thread_data.cwd or str(self._cwd or Path.cwd()),
                    created_at=created_at,
                    last_active=created_at,  # Codex doesn't track separate last_active
                    metadata={"title": thread_data.preview} if thread_data.preview else {},
                )

                result.append(session_data)

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
        from agentpool.sessions.models import SessionData

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
        self.log.info("Thread resumed from Codex server", thread_id=thread.id)

        # Convert turns to ChatMessages and populate conversation
        if thread.turns:
            from agentpool.agents.codex_agent.codex_converters import turns_to_chat_messages

            chat_messages = turns_to_chat_messages(thread.turns)
            self.conversation.chat_messages.clear()
            self.conversation.chat_messages.extend(chat_messages)
            self.log.info(
                "Restored conversation history",
                session_id=session_id,
                turn_count=len(thread.turns),
                message_count=len(chat_messages),
            )

        # Build SessionData from the resumed thread
        created_at = datetime.fromtimestamp(thread.created_at, tz=UTC)
        cwd = thread.cwd or str(self._cwd or Path.cwd())

        return SessionData(
            session_id=thread.id,
            agent_name=self.name,
            cwd=cwd,
            created_at=created_at,
            last_active=created_at,  # Codex doesn't track separate last_active
            metadata={"title": thread.preview} if thread.preview else {},
        )
