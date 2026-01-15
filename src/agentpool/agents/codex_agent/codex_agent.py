"""Codex agent - wraps Codex app-server via JSON-RPC protocol."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

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
    from agentpool.agents.modes import ModeCategory, ModeInfo
    from agentpool.common_types import (
        BuiltinEventHandlerType,
        IndividualEventHandler,
    )
    from agentpool.delegation import AgentPool
    from agentpool.messaging import MessageHistory
    from agentpool.models.codex_agents import CodexAgentConfig
    from agentpool.sessions import SessionData
    from agentpool.sessions.protocol import SessionInfo
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig
    from agentpool_config.nodes import ToolConfirmationMode
    from codex_adapter import CodexClient


logger = get_logger(__name__)


class CodexAgent[TDeps = None](BaseAgent[TDeps, str]):
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
        reasoning_effort: str | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        env: ExecutionEnvironment | None = None,
        input_provider: InputProvider | None = None,
        tool_confirmation_mode: ToolConfirmationMode = "always",
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
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
            agent_pool: Agent pool for coordination
            enable_logging: Whether to enable database logging
            mcp_servers: MCP server configurations
            env: Execution environment
            input_provider: Provider for user input
            tool_confirmation_mode: Tool confirmation behavior
            event_handlers: Event handlers for this agent
        """
        from agentpool.models.codex_agents import CodexAgentConfig

        # Build config from kwargs if not provided
        if config is None:
            config = CodexAgentConfig(
                cwd=cwd,
                model=model,
                reasoning_effort=reasoning_effort,
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
            tool_confirmation_mode=tool_confirmation_mode,
            event_handlers=event_handlers,
        )

        self.config = config
        self._client: CodexClient | None = None
        self._thread_id: str | None = None
        self._approval_policy: Literal["always", "never", "auto"] = (
            config.approval_policy or "never"
        )
        # Store MCP servers separately - will be passed to CodexClient
        # External MCP servers from config (already processed to MCPServerConfig objects)
        # If mcp_servers param provided, we need to process it similarly
        if mcp_servers:
            # Convert any strings to MCPServerConfig objects
            from agentpool_config.mcp_server import StdioMCPServerConfig

            processed: list[MCPServerConfig] = []
            for server in mcp_servers:
                if isinstance(server, str):
                    parts = server.split()
                    if not parts:
                        msg = "MCP server command string is empty"
                        raise ValueError(msg)
                    processed.append(StdioMCPServerConfig(command=parts[0], args=parts[1:]))
                else:
                    processed.append(server)
            self._external_mcp_servers = processed
        else:
            self._external_mcp_servers = config.get_mcp_servers()
        # Extra MCP servers in Codex format (e.g., tool bridge)
        self._extra_mcp_servers: list[tuple[str, Any]] = []

        # Track current settings (for when they change mid-session)
        self._current_model: str | None = None
        self._current_effort: Literal["low", "medium", "high"] | None = None

        # Create bridge for exposing our tools via MCP
        from agentpool.mcp_server.tool_bridge import ToolManagerBridge

        self._tool_bridge = ToolManagerBridge(
            node=self,
            server_name=f"agentpool-{self.name}-tools",
        )

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
        await super().__aenter__()
        await self._setup_toolsets()

        from agentpool.agents.codex_agent.codex_converters import mcp_configs_to_codex
        from codex_adapter import CodexClient

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
            effort=self.config.reasoning_effort,
        )
        self._thread_id = thread.id

        self.log.info(
            "Codex thread started",
            thread_id=self._thread_id,
            cwd=cwd,
            model=self.config.model,
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

    async def _stream_events(
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
    ) -> AsyncIterator[RichAgentStreamEvent[str]]:
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
        from uuid import uuid4

        from agentpool.agents.codex_agent.codex_converters import codex_to_native_event

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
        run_started = RunStartedEvent(
            thread_id=final_conversation_id,
            run_id=run_id,
        )
        await event_handlers(None, run_started)
        yield run_started
        # Stream turn events with bridge context set
        accumulated_text: list[str] = []
        # Pass output type directly - adapter handles conversion to JSON schema
        output_schema = None if self._output_type is str else self._output_type
        try:
            async with self._tool_bridge.set_run_context(deps, input_provider):
                async for event in self._client.turn_stream(
                    self._thread_id,
                    prompt_text,
                    approval_policy=self._approval_policy,
                    output_schema=output_schema,
                ):
                    # Convert Codex event to native event
                    if native_event := codex_to_native_event(event):
                        await event_handlers(None, native_event)
                        yield native_event

                        # Accumulate text for final message
                        from pydantic_ai import TextPartDelta

                        match native_event:
                            case PartDeltaEvent(delta=TextPartDelta(content_delta=text)):
                                accumulated_text.append(text)

        except Exception as e:
            self.log.exception("Error during Codex turn", error=str(e))
            raise

        # Emit completion event
        final_text = "".join(accumulated_text)
        complete_msg = ChatMessage(
            content=final_text,
            role="assistant",
            message_id=final_message_id,
            conversation_id=final_conversation_id,
            parent_id=parent_id,
        )

        complete_event = StreamCompleteEvent(
            message=complete_msg,
        )
        await event_handlers(None, complete_event)
        yield complete_event

    @property
    def model_name(self) -> str:
        """Get current model name."""
        return self.config.model or "unknown"

    def to_structured[NewOutputDataT](
        self,
        output_type: type[NewOutputDataT],
    ) -> CodexAgent[TDeps]:
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
        return self

    async def set_model(self, model: str) -> None:
        """Set the model for this agent.

        Archives the current thread and starts a new one with the new model,
        preserving conversation history in the archive.

        Args:
            model: Model identifier
        """
        if not self._client or not self._thread_id:
            # Not connected yet, just store for initialization
            self._current_model = model
            self.log.info("Model set for initialization", model=model)
            return

        # Archive current thread and start new one with new model
        old_thread_id = self._thread_id
        await self._client.thread_archive(old_thread_id)

        # Start new thread with new model, preserving other settings
        cwd = str(self.config.cwd or Path.cwd())
        thread = await self._client.thread_start(
            cwd=cwd,
            model=model,
            effort=self.config.reasoning_effort,
        )
        self._thread_id = thread.id
        self._current_model = model

        self.log.info(
            "Model changed - new thread started",
            old_thread=old_thread_id,
            new_thread=self._thread_id,
            model=model,
        )

    async def set_tool_confirmation_mode(self, mode: ToolConfirmationMode) -> None:
        """Set tool confirmation mode.

        Args:
            mode: Tool confirmation mode
        """
        self.tool_confirmation_mode = mode
        self.log.info("Tool confirmation mode updated", mode=mode)

    async def interrupt(self) -> None:
        """Interrupt current execution."""
        # Codex doesn't have explicit interrupt, but we can track cancellation
        self._cancelled = True
        self.log.info("Interrupt requested")

    async def get_available_models(self) -> list[ModelInfo] | None:
        """Get available models from Codex server.

        Returns:
            List of tokonomics ModelInfo for available models, or None if not connected
        """
        if not self._client:
            self.log.warning("Cannot get models: client not connected")
            return None

        try:
            from tokonomics.model_discovery.model_info import ModelInfo as TokModelInfo

            models_data = await self._client.model_list()
            models = []

            for model_data in models_data:
                # Infer provider from model name
                model_id = model_data.model or model_data.id
                provider = "unknown"
                if "claude" in model_id.lower():
                    provider = "anthropic"
                elif "gpt" in model_id.lower() or "o1" in model_id.lower():
                    provider = "openai"
                elif "gemini" in model_id.lower():
                    provider = "google"

                # Create basic ModelInfo (Codex doesn't provide pricing/capability details)
                models.append(
                    TokModelInfo(
                        id=model_id,
                        name=model_data.id,
                        provider=provider,
                        description=f"Model: {model_id}",
                    )
                )

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
        from agentpool.agents.modes import ModeCategory, ModeInfo

        categories: list[ModeCategory] = []

        # Approval policy modes
        current_policy = self._approval_policy
        policy_modes = [
            ModeInfo(
                id="never",
                name="Auto-Execute",
                description="Execute tools without approval (default for programmatic use)",
                category_id="approval_policy",
            ),
            ModeInfo(
                id="auto",
                name="Auto-Approve Safe",
                description="Auto-approve low-risk tools, ask for high-risk",
                category_id="approval_policy",
            ),
            ModeInfo(
                id="always",
                name="Always Confirm",
                description="Request approval before executing any tool",
                category_id="approval_policy",
            ),
        ]
        categories.append(
            ModeCategory(
                id="approval_policy",
                name="Tool Approval",
                available_modes=policy_modes,
                current_mode_id=current_policy,
                category="mode",
            )
        )

        # Reasoning effort modes
        current_effort = self.config.reasoning_effort or "medium"
        effort_modes = [
            ModeInfo(
                id="low",
                name="Low Effort",
                description="Fast, minimal reasoning",
                category_id="reasoning_effort",
            ),
            ModeInfo(
                id="medium",
                name="Medium Effort",
                description="Balanced reasoning",
                category_id="reasoning_effort",
            ),
            ModeInfo(
                id="high",
                name="High Effort",
                description="Deep reasoning",
                category_id="reasoning_effort",
            ),
        ]
        categories.append(
            ModeCategory(
                id="reasoning_effort",
                name="Reasoning Effort",
                available_modes=effort_modes,
                current_mode_id=current_effort,
                category="thought_level",
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

    async def set_mode(self, mode: ModeInfo | str, category_id: str | None = None) -> None:
        """Set a mode within a category.

        Codex supports:
        - approval_policy: "always", "never", "auto"
        - reasoning_effort: "low", "medium", "high"
        - model: Any valid model identifier

        Approval policy can be changed without restarting thread.
        Reasoning effort and model require archiving current thread and starting new one.

        Args:
            mode: Mode to set (ModeInfo or mode ID string)
            category_id: Category ID ("approval_policy", "reasoning_effort", or "model")
        """
        from agentpool.agents.modes import ModeInfo as ModeInfoType

        # Extract mode ID and infer category if needed
        if isinstance(mode, ModeInfoType):
            mode_id = mode.id
            category_id = category_id or mode.category_id
        else:
            mode_id = mode

        # If no category specified, try to infer from mode_id
        if not category_id:
            if mode_id in ["always", "never", "auto"]:
                category_id = "approval_policy"
            elif mode_id in ["low", "medium", "high"]:
                category_id = "reasoning_effort"
            else:
                category_id = "model"

        # Handle based on category
        if category_id == "approval_policy":
            if mode_id not in ["always", "never", "auto"]:
                msg = f"Invalid approval policy: {mode_id}. Must be 'always', 'never', or 'auto'"
                raise ValueError(msg)

            # Update instance attribute (doesn't require thread restart)
            # Type assertion: we've already validated mode_id is one of the valid values
            self._approval_policy = mode_id  # type: ignore[assignment]
            self.log.info("Approval policy changed", policy=mode_id)

        elif category_id == "reasoning_effort":
            if mode_id not in ["low", "medium", "high"]:
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
                effort=mode_id,  # type: ignore[arg-type]
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

        elif category_id == "model":
            await self.set_model(mode_id)

        else:
            msg = f"Unknown category: {category_id}"
            raise ValueError(msg)

    async def list_sessions(self) -> list[SessionInfo]:
        """List threads from Codex server.

        Queries the Codex server for available threads (sessions).

        Returns:
            List of SessionInfo objects converted from Codex ThreadData
        """
        from datetime import datetime

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
        from datetime import datetime

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
