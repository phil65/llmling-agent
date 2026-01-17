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
    from agentpool.sessions import SessionData
    from agentpool.sessions.protocol import SessionInfo
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig
    from agentpool_config.nodes import ToolConfirmationMode
    from codex_adapter import ApprovalPolicy, CodexClient, ReasoningEffort, SandboxMode
    from codex_adapter.events import CodexEvent


logger = get_logger(__name__)


class CodexAgent[TDeps = None, OutputDataT = str](BaseAgent[TDeps, OutputDataT]):
    """MessageNode that wraps a Codex app-server instance.

    This allows integrating Codex into the agentpool ecosystem, enabling
    composition with native agents via connections, teams, etc.

    The agent manages:
    - CodexClient lifecycle (connect on enter, disconnect on exit)
    - Thread/turn management
    - Event conversion from Codex to agentpool events
    - Streaming responses

    Example:
        ```python
        # From config
        config = CodexAgentConfig(cwd="/project")
        agent = CodexAgent(config=config, agent_pool=pool)

        # From kwargs
        agent = CodexAgent(name="codex", cwd="/project")
        ```
    """

    AGENT_TYPE: ClassVar = "codex"

    def __init__(
        self,
        *,
        config: CodexAgentConfig | None = None,
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
    ) -> None:
        """Initialize Codex agent.

        Args:
            config: Codex agent configuration
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
        """
        from agentpool.mcp_server.tool_bridge import ToolManagerBridge
        from agentpool.models.codex_agents import CodexAgentConfig
        from agentpool_config.mcp_server import BaseMCPServerConfig

        # Build config from kwargs if not provided
        if config is None:
            config = CodexAgentConfig(
                cwd=cwd,
                model=model,
                reasoning_effort=reasoning_effort,
                base_instructions=base_instructions,
                developer_instructions=developer_instructions,
            )
        # Use provided name or config name, fallback to "codex"
        agent_name = name or config.name or "codex"
        super().__init__(
            name=agent_name,
            description=description or config.description,
            display_name=display_name or config.display_name,
            # Don't pass mcp_servers to BaseAgent - we manage them ourselves
            # CodexClient will handle MCP server lifecycle
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            env=env,
            input_provider=input_provider,
            output_type=output_type,
            tool_confirmation_mode=tool_confirmation_mode,
            event_handlers=event_handlers,
            hooks=hooks,
        )

        self.config = config
        self._client: CodexClient | None = None
        self._thread_id: str | None = None
        self._approval_policy: ApprovalPolicy = config.approval_policy or "never"
        # Store MCP servers separately - will be passed to CodexClient
        # External MCP servers from config (already processed to MCPServerConfig objects)
        # If mcp_servers param provided, we need to process it similarly
        if mcp_servers:
            # Convert any strings to MCPServerConfig objects
            processed: list[MCPServerConfig] = []
            for server in mcp_servers:
                if isinstance(server, str):
                    processed.append(BaseMCPServerConfig.from_string(server))
                else:
                    processed.append(server)
            self._external_mcp_servers = processed
        else:
            self._external_mcp_servers = config.get_mcp_servers()
        # Extra MCP servers in Codex format (e.g., tool bridge)
        self._extra_mcp_servers: list[tuple[str, Any]] = []
        # Track current settings (for when they change mid-session)
        self._current_model: str | None = config.model
        self._current_effort: ReasoningEffort | None = config.reasoning_effort
        self._current_sandbox: SandboxMode | None = config.sandbox
        self._current_turn_id: str | None = None  # Track current turn for interrupt
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

        Args:
            config: Agent configuration
            event_handlers: Event handlers (merged with config handlers)
            input_provider: Provider for user input
            agent_pool: Agent pool for coordination

        Returns:
            Configured agent instance
        """
        from agentpool.models.manifest import AgentsManifest
        from agentpool.utils.result_utils import to_type

        # Resolve output type from config
        manifest = agent_pool.manifest if agent_pool else AgentsManifest()
        agent_output_type = config.output_type or str
        if isinstance(agent_output_type, str) and agent_output_type != "str":
            # Try to resolve from manifest responses
            resolved_output_type = to_type(agent_output_type, manifest.responses)
        else:
            resolved_output_type = to_type(agent_output_type)

        # Merge config-level handlers with provided handlers
        config_handlers = config.get_event_handlers()
        merged_handlers: list[IndividualEventHandler | BuiltinEventHandlerType] = [
            *config_handlers,
            *(event_handlers or []),
        ]
        return cls(
            config=config,
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
        from agentpool.models.manifest import AgentsManifest

        defn = self.agent_pool.manifest if self.agent_pool else AgentsManifest()
        return AgentContext(
            node=self,
            pool=self.agent_pool,
            config=self.config,
            definition=defn,
            input_provider=input_provider or self._input_provider,
            model_name=self.model_name,
        )

    async def _setup_toolsets(self) -> None:
        """Setup toolsets and start the tool bridge."""
        if not self.config.tools:
            return

        # Create providers from tool configs and add to tool manager
        for provider in self.config.get_tool_providers():
            self.tools.add_provider(provider)
        # Start bridge to expose tools via MCP
        await self._tool_bridge.start()
        # Add bridge's MCP server config to extra servers
        # get_codex_mcp_server_config returns (name, HttpMcpServer)
        bridge_config = self._tool_bridge.get_codex_mcp_server_config()
        self._extra_mcp_servers.append(bridge_config)

    async def __aenter__(self) -> Self:
        """Start Codex client and create thread."""
        from agentpool.agents.codex_agent.codex_converters import mcp_configs_to_codex
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
            mcp_servers_dict.update(dict(mcp_configs_to_codex(self._external_mcp_servers)))
        # Create and connect client with MCP servers
        self._client = CodexClient(mcp_servers=mcp_servers_dict)
        await self._client.__aenter__()
        # Start a thread
        cwd = str(self.config.cwd or Path.cwd())
        thread = await self._client.thread_start(
            cwd=cwd,
            model=self.config.model,
            base_instructions=self.config.base_instructions,
            developer_instructions=self.config.developer_instructions,
            sandbox=self._current_sandbox,
        )
        self._thread_id = thread.id
        self.log.info(
            "Codex thread started", thread_id=self._thread_id, cwd=cwd, model=self.config.model
        )
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
        self._thread_id = None

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
        """Stream events from Codex turn execution.

        Args:
            prompts: User prompts to send
            user_msg: Original user message
            message_history: Conversation history
            effective_parent_id: Effective parent message ID
            message_id: Optional message ID
            conversation_id: Optional conversation ID
            parent_id: Optional parent message ID
            input_provider: Provider for user input
            deps: Dependencies
            event_handlers: Event handlers
            wait_for_connections: Whether to wait for message routing
            store_history: Whether to store in conversation history

        Yields:
            Stream events from Codex execution
        """
        from agentpool.agents.codex_agent.codex_converters import convert_codex_stream
        from agentpool.agents.events import PlanUpdateEvent
        from agentpool.messaging.messages import TokenCost
        from codex_adapter.models import ThreadTokenUsageUpdatedData

        if not self._client or not self._thread_id:
            msg = "Codex client not initialized"
            raise RuntimeError(msg)

        # Convert prompts to text
        prompt_text = "\n\n".join(str(p) for p in prompts)
        # Generate IDs if not provided
        run_id = str(uuid4())
        final_message_id = message_id or str(uuid4())
        final_conversation_id = conversation_id or self.conversation_id
        # Ensure conversation_id is set (should always be from base class)
        if final_conversation_id is None:
            msg = "conversation_id must be set"
            raise ValueError(msg)
        # Emit run started event
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
                    self._thread_id,
                    prompt_text,
                    model=self._current_model,
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
        return self.config.model or "unknown"

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
        """Set the model for this agent.

        Args:
            model: Model identifier
        """
        await self._set_mode(model, "model")

    async def set_tool_confirmation_mode(self, mode: ToolConfirmationMode) -> None:
        """Set tool confirmation mode.

        Args:
            mode: Tool confirmation mode
        """
        self.tool_confirmation_mode = mode
        self.log.info("Tool confirmation mode updated", mode=mode)

    async def interrupt(self) -> None:
        """Interrupt current execution."""
        self._cancelled = True
        # Use Codex's turn_interrupt if we have an active turn
        if self._client and self._thread_id and self._current_turn_id:
            try:
                await self._client.turn_interrupt(self._thread_id, self._current_turn_id)
                self.log.info(
                    "Codex turn interrupted",
                    thread_id=self._thread_id,
                    turn_id=self._current_turn_id,
                )
            except Exception:
                self.log.exception("Failed to interrupt Codex turn")
        else:
            self.log.info("Interrupt requested (no active turn)")

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
            models_data = await self._client.model_list()
            models = []
            for model_data in models_data:
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
        """Get available mode categories for Codex agent.

        Codex exposes approval policy, reasoning effort levels, and model selection.

        Returns:
            List of ModeCategory for approval policy, reasoning effort, and models
        """
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
            current_model = self._current_model or self.config.model or ""
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

        # Handle based on category
        if category_id == "mode":
            if mode_id not in ["never", "on-request", "on-failure", "untrusted"]:
                msg = f"Invalid approval policy: {mode_id}"
                raise ValueError(msg)

            # Update instance attribute (doesn't require thread restart)
            # Type assertion: we've already validated mode_id is one of the valid values
            self._approval_policy = mode_id  # type: ignore[assignment]
            change = ConfigOptionChanged(config_id="mode", value_id=mode_id)
            await self.state_updated.emit(change)
            self.log.info("Approval policy changed", policy=mode_id)

        elif category_id == "thought_level":
            if mode_id not in ["low", "medium", "high", "xhigh"]:
                msg = f"Invalid reasoning effort: {mode_id}"
                raise ValueError(msg)

            if not self._client or not self._thread_id:
                # Store for initialization
                # Type assertion: we've already validated mode_id is one of the valid values
                self._current_effort = mode_id  # type: ignore[assignment]
                self.log.info("Reasoning effort set for initialization", effort=mode_id)
                return

            # Archive and restart with new effort
            old_thread_id = self._thread_id
            await self._client.thread_archive(old_thread_id)
            cwd = str(self.config.cwd or Path.cwd())
            thread = await self._client.thread_start(
                cwd=cwd,
                model=self._current_model or self.config.model,
                base_instructions=self.config.base_instructions,
                developer_instructions=self.config.developer_instructions,
                sandbox=self._current_sandbox,
            )
            self._thread_id = thread.id
            # Type assertion: we've already validated mode_id is one of the valid values
            self._current_effort = mode_id  # type: ignore[assignment]
            self.log.info(
                "Reasoning effort changed - new thread started",
                old_thread=old_thread_id,
                new_thread=self._thread_id,
                effort=mode_id,
            )
            # Emit state change signal
            change = ConfigOptionChanged(config_id="thought_level", value_id=mode_id)
            await self.state_updated.emit(change)
        elif category_id == "model":
            # Set model directly
            self._current_model = mode_id
            self.log.info("Model changed", model=mode_id)
            # Emit state change signal
            from agentpool.agents.modes import ConfigOptionChanged

            await self.state_updated.emit(ConfigOptionChanged(config_id="model", value_id=mode_id))

        elif category_id == "sandbox":
            valid = ["read-only", "workspace-write", "danger-full-access", "external-sandbox"]
            if mode_id not in valid:
                msg = f"Invalid sandbox mode: {mode_id}. Valid: {valid}"
                raise ValueError(msg)
            self._current_sandbox = mode_id  # type: ignore[assignment]
            self.log.info("Sandbox mode changed", sandbox=mode_id)
            change = ConfigOptionChanged(config_id="sandbox", value_id=mode_id)
            await self.state_updated.emit(change)

        else:
            msg = f"Unknown category: {category_id}"
            raise ValueError(msg)

    async def list_sessions(self) -> list[SessionInfo]:
        """List threads from Codex server.

        Queries the Codex server for available threads (sessions).

        Returns:
            List of SessionInfo objects converted from Codex ThreadData
        """
        from agentpool.sessions.models import SessionData

        if not self._client:
            return []
        try:
            response = await self._client.thread_list()
            result: list[SessionInfo] = []
            for thread_data in response.data:
                # Convert Codex ThreadData to SessionData
                # created_at is Unix timestamp (seconds)
                created_at = datetime.fromtimestamp(thread_data.created_at, tz=UTC)
                session_data = SessionData(
                    session_id=thread_data.id,
                    agent_name=self.name,
                    conversation_id=thread_data.id,
                    cwd=thread_data.cwd or str(self.config.cwd or Path.cwd()),
                    created_at=created_at,
                    last_active=created_at,  # Codex doesn't track separate last_active
                    metadata={"title": thread_data.preview} if thread_data.preview else {},
                )

                result.append(session_data)  # type: ignore[arg-type]

        except Exception:
            self.log.exception("Failed to list Codex threads")
            return []
        else:
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
            # Resume the thread on Codex server
            thread = await self._client.thread_resume(session_id)
            # Update current thread ID
            self._thread_id = thread.id
            self.log.info("Thread resumed from Codex server", thread_id=thread.id)
            # Build SessionData from the resumed thread
            created_at = datetime.fromtimestamp(thread.created_at, tz=UTC)
            # CodexThread doesn't include cwd, use config default
            cwd = str(self.config.cwd or Path.cwd())
            return SessionData(
                session_id=thread.id,
                agent_name=self.name,
                conversation_id=thread.id,
                cwd=cwd,
                created_at=created_at,
                last_active=created_at,  # Codex doesn't track separate last_active
                metadata={"title": thread.preview} if thread.preview else {},
            )

        except Exception:
            self.log.exception("Failed to resume Codex thread", session_id=session_id)
            return None
