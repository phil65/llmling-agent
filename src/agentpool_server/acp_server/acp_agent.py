"""ACP (Agent Client Protocol) Agent implementation."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any, ClassVar

from acp import Agent as ACPAgent
from acp.schema import (
    ForkSessionResponse,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    ModelInfo as ACPModelInfo,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionConfigOption,
    SessionConfigSelectOption,
    SessionInfo,
    SessionMode,
    SessionModelState,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModelRequest,
    SetSessionModelResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
)
from agentpool.log import get_logger
from agentpool.utils.tasks import TaskManager
from agentpool_server.acp_server.session_manager import ACPSessionManager


if TYPE_CHECKING:
    from pydantic_ai import ModelRequest, ModelResponse

    from acp import Client
    from acp.schema import (
        AuthenticateRequest,
        CancelNotification,
        ClientCapabilities,
        ForkSessionRequest,
        Implementation,
        InitializeRequest,
        ListSessionsRequest,
        LoadSessionRequest,
        NewSessionRequest,
        PromptRequest,
        ResumeSessionRequest,
        SetSessionConfigOptionRequest,
        SetSessionModelRequest,
        SetSessionModeRequest,
    )
    from agentpool import AgentPool
    from agentpool_server.acp_server.server import ACPServer
    from agentpool_server.acp_server.session import ACPSession

logger = get_logger(__name__)


async def get_session_model_state(
    agent: Any, current_model: str | None = None
) -> SessionModelState | None:
    """Get SessionModelState from an agent using its get_available_models() method.

    Converts tokonomics ModelInfo to ACP ModelInfo format.

    Args:
        agent: Any agent with get_available_models() method
        current_model: Currently active model ID (defaults to first available)

    Returns:
        SessionModelState with all available models, None if no models available
    """
    from agentpool.agents.base_agent import BaseAgent

    if not isinstance(agent, BaseAgent):
        return None

    try:
        toko_models = await agent.get_available_models()
    except Exception:
        logger.exception("Failed to get available models from agent")
        return None

    if not toko_models:
        return None

    # Convert tokonomics ModelInfo to ACP ModelInfo
    acp_models: list[ACPModelInfo] = []
    for toko in toko_models:
        # Use id_override if set (e.g., "opus" for Claude Code), otherwise use id
        model_id = toko.id_override if toko.id_override else toko.id
        info = ACPModelInfo(model_id=model_id, name=toko.name, description=toko.description or "")
        acp_models.append(info)
    if not acp_models:
        return None

    # Ensure current model is in the list
    all_ids = [model.model_id for model in acp_models]
    if current_model and current_model not in all_ids:
        # Add current model to the list so the UI shows it
        desc = "Currently configured model"
        model_info = ACPModelInfo(model_id=current_model, name=current_model, description=desc)
        acp_models.insert(0, model_info)
        current_model_id = current_model
    else:
        current_model_id = current_model if current_model in all_ids else all_ids[0]

    return SessionModelState(available_models=acp_models, current_model_id=current_model_id)


async def get_session_mode_state(agent: Any) -> SessionModeState | None:
    """Get SessionModeState from an agent using its get_modes() method.

    Converts agentpool ModeCategory to ACP SessionModeState format.
    Uses the first category that looks like permissions (not model).

    Args:
        agent: Any agent with get_modes() method

    Returns:
        SessionModeState from agent's modes, None if no modes available
    """
    from agentpool.agents.base_agent import BaseAgent

    if not isinstance(agent, BaseAgent):
        return None

    try:
        mode_categories = await agent.get_modes()
    except Exception:
        logger.exception("Failed to get modes from agent")
        return None

    if not mode_categories:
        return None

    # Find the permissions category (not model)
    category = next((c for c in mode_categories if c.id != "model"), None)
    if not category:
        return None

    # Convert ModeInfo to ACP SessionMode
    acp_modes = [
        SessionMode(
            id=mode.id,
            name=mode.name,
            description=mode.description,
        )
        for mode in category.available_modes
    ]

    return SessionModeState(
        available_modes=acp_modes,
        current_mode_id=category.current_mode_id,
    )


async def get_session_config_options(agent: Any) -> list[SessionConfigOption]:
    """Get SessionConfigOptions from an agent using its get_modes() method.

    Converts all agentpool ModeCategories to ACP SessionConfigOption format.

    Args:
        agent: Any agent with get_modes() method

    Returns:
        List of SessionConfigOption from agent's mode categories
    """
    from agentpool.agents.base_agent import BaseAgent

    if not isinstance(agent, BaseAgent):
        return []

    try:
        mode_categories = await agent.get_modes()
    except Exception:
        logger.exception("Failed to get modes from agent")
        return []

    if not mode_categories:
        return []

    # Convert each ModeCategory to a SessionConfigOption
    config_options: list[SessionConfigOption] = []
    for category in mode_categories:
        options = [
            SessionConfigSelectOption(
                value=mode.id,
                name=mode.name,
                description=mode.description,
            )
            for mode in category.available_modes
        ]
        config_options.append(
            SessionConfigOption(
                id=category.id,
                name=category.name,
                description=None,
                category=category.category,
                current_value=category.current_mode_id,
                options=options,
            )
        )

    return config_options


@dataclass
class AgentPoolACPAgent(ACPAgent):
    """Implementation of ACP Agent protocol interface for AgentPool.

    This class implements the external library's Agent protocol interface,
    bridging AgentPool with the standard ACP JSON-RPC protocol.
    """

    PROTOCOL_VERSION: ClassVar = 1

    client: Client
    """ACP connection for client communication."""

    agent_pool: AgentPool[Any]
    """AgentPool containing available agents."""

    _: KW_ONLY

    file_access: bool = True
    """Whether agent can access filesystem."""

    terminal_access: bool = True
    """Whether agent can use terminal."""

    debug_commands: bool = False
    """Whether to enable debug slash commands for testing."""

    default_agent: str | None = None
    """Optional specific agent name to use as default."""

    load_skills: bool = True
    """Whether to load client-side skills from .claude/skills directory."""

    server: ACPServer | None = field(default=None)
    """Reference to the ACPServer for pool hot-switching."""

    def __post_init__(self) -> None:
        """Initialize derived attributes and setup after field assignment."""
        self.client_capabilities: ClientCapabilities | None = None
        self.client_info: Implementation | None = None
        self.session_manager = ACPSessionManager(pool=self.agent_pool)
        self.tasks = TaskManager()
        self._initialized = False

        agent_count = len(self.agent_pool.agents)
        logger.info("Created ACP agent implementation", agent_count=agent_count)
        if self.debug_commands:
            logger.info("Debug slash commands enabled for ACP testing")

        # Note: Tool registration happens after initialize() when we know client caps

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        """Initialize the agent and negotiate capabilities."""
        version = min(params.protocol_version, self.PROTOCOL_VERSION)
        self.client_capabilities = params.client_capabilities
        self.client_info = params.client_info
        logger.info("Client info", request=params.model_dump_json())
        self._initialized = True
        return InitializeResponse.create(
            protocol_version=version,
            name="agentpool",
            title="AgentPool",
            version=_version("agentpool"),
            load_session=True,
            list_sessions=True,
            http_mcp_servers=True,
            sse_mcp_servers=True,
            audio_prompts=True,
            embedded_context_prompts=True,
            image_prompts=True,
        )

    async def new_session(self, params: NewSessionRequest) -> NewSessionResponse:
        """Create a new session."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        try:
            names = list(self.agent_pool.all_agents.keys())
            if not names:
                logger.error("No agents available for session creation")
                raise RuntimeError("No agents available")  # noqa: TRY301

            # Use specified default agent or fall back to first agent
            if self.default_agent and self.default_agent in names:
                default_name = self.default_agent
            else:
                default_name = names[0]

            logger.info("Creating new session", agents=names, default_agent=default_name)
            session_id = await self.session_manager.create_session(
                default_agent_name=default_name,
                cwd=params.cwd,
                client=self.client,
                acp_agent=self,
                mcp_servers=params.mcp_servers,
                client_capabilities=self.client_capabilities,
                client_info=self.client_info,
            )

            # Get mode and model information from the agent
            from agentpool.agents.acp_agent import ACPAgent as ACPAgentClient

            state: SessionModeState | None = None
            models: SessionModelState | None = None
            config_options: list[SessionConfigOption] = []

            if session := self.session_manager.get_session(session_id):
                if isinstance(session.agent, ACPAgentClient):
                    # Nested ACP agent - pass through its state directly
                    if session.agent._state:
                        models = session.agent._state.models
                        state = session.agent._state.modes
                    # Also get config_options from nested agent
                    config_options = await get_session_config_options(session.agent)
                else:
                    # Use unified helpers for all other agents
                    models = await get_session_model_state(session.agent, session.agent.model_name)
                    state = await get_session_mode_state(session.agent)
                    config_options = await get_session_config_options(session.agent)
        except Exception:
            logger.exception("Failed to create new session")
            raise
        else:
            # Schedule available commands update after session response is returned
            if session := self.session_manager.get_session(session_id):
                # Schedule task to run after response is sent
                self.tasks.create_task(session.send_available_commands_update())
                self.tasks.create_task(session.init_project_context())
                self.tasks.create_task(session._register_prompt_hub_commands())
                if self.load_skills:
                    coro_4 = session.init_client_skills()
                    self.tasks.create_task(coro_4, name=f"init_client_skills_{session_id}")
            logger.info("Created session", session_id=session_id)

            return NewSessionResponse(
                session_id=session_id,
                modes=state,
                models=models,
                config_options=config_options if config_options else None,
            )

    async def load_session(self, params: LoadSessionRequest) -> LoadSessionResponse:
        """Load an existing session from storage.

        This tries to:
        1. Check if session is already active
        2. If not, try to resume from persistent storage
        3. Initialize MCP servers and other session resources
        4. Replay conversation history via ACP notifications
        5. Return session state (modes, models)
        """
        from agentpool.agents.acp_agent import ACPAgent as ACPAgentClient

        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        try:
            # First check if session is already active
            session = self.session_manager.get_session(params.session_id)
            if not session:
                # Try to resume from storage
                msg = "Attempting to resume session from storage"
                logger.info(msg, session_id=params.session_id)
                session = await self.session_manager.resume_session(
                    session_id=params.session_id,
                    client=self.client,
                    acp_agent=self,
                    client_capabilities=self.client_capabilities,
                    client_info=self.client_info,
                )

            if not session:
                logger.warning("Session not found in storage", session_id=params.session_id)
                return LoadSessionResponse()

            # Update session with new request parameters if provided
            if params.cwd and params.cwd != session.cwd:
                session.cwd = params.cwd
                logger.info("Updated session cwd", session_id=params.session_id, cwd=params.cwd)

            # Initialize MCP servers if provided in load request
            if params.mcp_servers:
                session.mcp_servers = params.mcp_servers
                await session.initialize_mcp_servers()

            mode_state: SessionModeState | None = None
            models: SessionModelState | None = None
            config_opts: list[SessionConfigOption] = []

            if isinstance(session.agent, ACPAgentClient):
                # Nested ACP agent - pass through its state directly
                if session.agent._state:
                    mode_state = session.agent._state.modes
                    models = session.agent._state.models
                config_opts = await get_session_config_options(session.agent)
            elif session.agent:
                # Use unified helpers for all other agents
                mode_state = await get_session_mode_state(session.agent)
                models = await get_session_model_state(session.agent, session.agent.model_name)
                config_opts = await get_session_config_options(session.agent)
            else:
                models = None
            # Schedule post-load initialization tasks
            self.tasks.create_task(session.send_available_commands_update())
            self.tasks.create_task(session.init_project_context())
            # Replay conversation history via ACP notifications
            self.tasks.create_task(self._replay_conversation_history(session))
            logger.info("Session loaded successfully", agent=session.current_agent_name)
            return LoadSessionResponse(models=models, modes=mode_state, config_options=config_opts)

        except Exception:
            logger.exception("Failed to load session", session_id=params.session_id)
            return LoadSessionResponse()

    async def _replay_conversation_history(self, session: ACPSession) -> None:
        """Replay conversation history for a loaded session via ACP notifications.

        Per ACP spec, when loading a session the agent MUST replay the entire
        conversation to the client via session/update notifications.

        Args:
            session: The ACP session to replay history for
        """
        from agentpool_config.session import SessionQuery

        # Get session data to find conversation_id
        session_data = await self.session_manager.session_manager.store.load(session.session_id)
        if not session_data:
            logger.warning("No session data found for replay", session_id=session.session_id)
            return

        # Get storage provider
        storage = self.agent_pool.storage
        if not storage:
            logger.debug("No storage provider, skipping conversation replay")
            return

        # Query messages by conversation_id
        query = SessionQuery(name=session_data.conversation_id)
        try:
            chat_messages = await storage.filter_messages(query)
        except NotImplementedError:
            logger.debug("Storage provider doesn't support history loading")
            return

        if not chat_messages:
            logger.debug("No messages to replay", session_id=session.session_id)
            return

        logger.info("Replaying conversation history", message_count=len(chat_messages))
        # Extract ModelRequest/ModelResponse from ChatMessage.messages field
        model_messages: list[ModelRequest | ModelResponse] = []
        for chat_msg in chat_messages:
            if chat_msg.messages:
                model_messages.extend(chat_msg.messages)

        if not model_messages:
            logger.debug("No model messages to replay", session_id=session.session_id)
            return

        # Use ACPNotifications.replay() which handles all content types properly
        try:
            await session.notifications.replay(model_messages)
            logger.info("Conversation replay complete", replayed_count=len(model_messages))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to replay conversation history", error=str(e))

    async def list_sessions(self, params: ListSessionsRequest) -> ListSessionsResponse:
        """List available sessions.

        Returns sessions from both active memory and persistent storage.
        Supports pagination via cursor and optional cwd filtering.
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        try:
            # Get session IDs from storage (includes both active and persisted)
            session_ids = await self.session_manager.list_sessions(active_only=False)
            # Build SessionInfo objects
            sessions: list[SessionInfo] = []
            for session_id in session_ids:
                # Try active session first
                active_session = self.session_manager.get_session(session_id)
                if active_session:
                    # Filter by cwd if specified
                    if params.cwd and active_session.cwd != params.cwd:
                        continue
                    title = f"Session with {active_session.current_agent_name}"
                    sessions.append(
                        SessionInfo(session_id=session_id, cwd=active_session.cwd, title=title)
                    )
                else:
                    # Load from storage to get details
                    data = await self.session_manager.session_manager.store.load(session_id)
                    if data:
                        # Filter by cwd if specified
                        if params.cwd and data.cwd != params.cwd:
                            continue
                        info = SessionInfo(
                            session_id=session_id,
                            cwd=data.cwd or "",
                            title=f"Session with {data.agent_name}",
                            updated_at=data.last_active.isoformat(),
                        )
                        sessions.append(info)

            logger.info("Listed sessions", count=len(sessions))
            return ListSessionsResponse(sessions=sessions)

        except Exception:
            logger.exception("Failed to list sessions")
            return ListSessionsResponse(sessions=[])

    async def fork_session(self, params: ForkSessionRequest) -> ForkSessionResponse:
        """Fork an existing session.

        Creates a new session with the same state as the original.
        UNSTABLE: This feature is not part of the spec yet.
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        logger.info("Forking session", session_id=params.session_id)
        # For now, just create a new session - full fork implementation would copy state
        default_agent = next(iter(self.agent_pool.manifest.agents.keys()))
        session_id = await self.session_manager.create_session(
            default_agent_name=default_agent,
            cwd=params.cwd,
            client=self.client,
            acp_agent=self,
            mcp_servers=params.mcp_servers,
            session_id=None,  # Let it generate a new ID
            client_capabilities=self.client_capabilities,
            client_info=self.client_info,
        )
        return ForkSessionResponse(session_id=session_id)

    async def resume_session(self, params: ResumeSessionRequest) -> ResumeSessionResponse:
        """Resume an existing session.

        Like load_session but doesn't return previous messages.
        UNSTABLE: This feature is not part of the spec yet.
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        logger.info("Resuming session", session_id=params.session_id)
        # Similar to load_session but without replaying history
        try:
            session = await self.session_manager.resume_session(
                session_id=params.session_id,
                client=self.client,
                acp_agent=self,
                client_capabilities=self.client_capabilities,
                client_info=self.client_info,
            )
            if not session:
                logger.warning("Session not found", session_id=params.session_id)
            return ResumeSessionResponse()
        except Exception:
            logger.exception("Failed to resume session", session_id=params.session_id)
            return ResumeSessionResponse()

    async def authenticate(self, params: AuthenticateRequest) -> None:
        """Authenticate with the agent."""
        logger.info("Authentication requested", method_id=params.method_id)

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        """Process a prompt request."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        logger.info("Processing prompt", session_id=params.session_id)
        session = self.session_manager.get_session(params.session_id)

        # Auto-recreate session if not found (e.g., after pool swap)
        if not session:
            logger.info("Session not found, recreating", session_id=params.session_id)
            try:
                # Get default agent name
                names = list(self.agent_pool.all_agents.keys())
                if not names:
                    logger.error("No agents available for session recreation")
                    return PromptResponse(stop_reason="end_turn")

                default_name = (
                    self.default_agent
                    if self.default_agent and self.default_agent in names
                    else names[0]
                )

                # Try to get cwd from stored session data
                cwd = "."
                try:
                    stored = await self.session_manager.session_manager.store.load(
                        params.session_id
                    )
                    if stored and stored.cwd:
                        cwd = stored.cwd
                except Exception:  # noqa: BLE001
                    pass  # Use default cwd

                # Recreate session with same ID
                await self.session_manager.create_session(
                    default_agent_name=default_name,
                    cwd=cwd,
                    client=self.client,
                    acp_agent=self,
                    session_id=params.session_id,
                    client_capabilities=self.client_capabilities,
                    client_info=self.client_info,
                )
                session = self.session_manager.get_session(params.session_id)
                if session:
                    # Initialize session extras
                    self.tasks.create_task(session.send_available_commands_update())
                    self.tasks.create_task(session.init_project_context())
            except Exception:
                logger.exception("Failed to recreate session", session_id=params.session_id)
                return PromptResponse(stop_reason="end_turn")

        try:
            if not session:
                raise ValueError(f"Session {params.session_id} not found")  # noqa: TRY301
            stop_reason = await session.process_prompt(params.prompt)
            # Return the actual stop reason from the session
        except Exception as e:
            logger.exception("Failed to process prompt", session_id=params.session_id)
            msg = f"Error processing prompt: {e}"
            if session:
                # Send error notification asynchronously to avoid blocking response
                name = f"error_notification_{params.session_id}"
                self.tasks.create_task(session._send_error_notification(msg), name=name)

            return PromptResponse(stop_reason="end_turn")
        else:
            response = PromptResponse(stop_reason=stop_reason)
            logger.info("Returning PromptResponse", stop_reason=stop_reason)
            return response

    async def cancel(self, params: CancelNotification) -> None:
        """Cancel operations for a session."""
        logger.info("Cancelling session", session_id=params.session_id)
        try:
            # Get session and cancel it
            if session := self.session_manager.get_session(params.session_id):
                await session.cancel()
                logger.info("Cancelled operations", session_id=params.session_id)
            else:
                msg = "Session not found for cancellation"
                logger.warning(msg, session_id=params.session_id)

        except Exception:
            logger.exception("Failed to cancel session", session_id=params.session_id)

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"example": "response"}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None

    async def set_session_mode(
        self, params: SetSessionModeRequest
    ) -> SetSessionModeResponse | None:
        """Set the session mode (change tool confirmation level).

        Calls set_mode directly on the agent with the mode_id, allowing the agent
        to handle mode-specific logic (e.g., acceptEdits auto-allowing edit tools).
        """
        from agentpool.agents.acp_agent import ACPAgent as ACPAgentClient

        try:
            session = self.session_manager.get_session(params.session_id)
            if not session:
                logger.warning("Session not found for mode switch", session_id=params.session_id)
                return None

            # Call set_mode directly - agent handles mode-specific logic
            await session.agent.set_mode(params.mode_id)

            # Update stored mode state for ACPAgent
            if (
                isinstance(session.agent, ACPAgentClient)
                and session.agent._state
                and session.agent._state.modes
            ):
                session.agent._state.modes.current_mode_id = params.mode_id

            logger.info(
                "Set mode",
                mode_id=params.mode_id,
                session_id=params.session_id,
                agent_type=type(session.agent).__name__,
            )
            return SetSessionModeResponse()

        except Exception:
            logger.exception("Failed to set session mode", session_id=params.session_id)
            return None

    async def set_session_model(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse | None:
        """Set the session model.

        Changes the model for the active agent in the session.
        Validates that the requested model is available via agent.get_available_models().
        """
        from agentpool.agents.acp_agent import ACPAgent as ACPAgentClient

        try:
            session = self.session_manager.get_session(params.session_id)
            if not session:
                msg = "Session not found for model switch"
                logger.warning(msg, session_id=params.session_id)
                return None

            # Handle ACPAgent specially - pass through to nested agent's model state
            if isinstance(session.agent, ACPAgentClient):
                if session.agent._state and session.agent._state.models:
                    available_ids = [
                        m.model_id for m in session.agent._state.models.available_models
                    ]
                    if params.model_id not in available_ids:
                        logger.warning(
                            "Model not in ACPAgent's available models",
                            model_id=params.model_id,
                            available=available_ids,
                        )
                        return None
                # TODO: Use ACP protocol once set_session_model stabilizes
                logger.warning(
                    "Model change for ACPAgent not yet supported (ACP protocol UNSTABLE)",
                    model_id=params.model_id,
                )
                return None

            # Get available models from agent and validate
            toko_models = await session.agent.get_available_models()
            if toko_models:
                # Build list of valid model IDs (using id_override if set)
                available_ids = [m.id_override if m.id_override else m.id for m in toko_models]
                if params.model_id not in available_ids:
                    logger.warning(
                        "Model not in available models",
                        model_id=params.model_id,
                        available=available_ids,
                    )
                    return None

            # Set the model on the agent (all agents now have async set_model)
            await session.agent.set_model(params.model_id)
            logger.info("Set model", model_id=params.model_id, session_id=params.session_id)
            return SetSessionModelResponse()
        except Exception:
            logger.exception("Failed to set session model", session_id=params.session_id)
            return None

    async def set_session_config_option(
        self, params: SetSessionConfigOptionRequest
    ) -> SetSessionConfigOptionResponse | None:
        """Set a session config option.

        Forwards the config option change to the agent's set_mode method
        and returns the updated config options.
        """
        try:
            session = self.session_manager.get_session(params.session_id)
            if not session or not session.agent:
                msg = "Session not found for config option change"
                logger.warning(msg, session_id=params.session_id)
                return None

            logger.info(
                "Set config option",
                config_id=params.config_id,
                value=params.value,
                session_id=params.session_id,
            )

            # Forward to agent's set_mode method
            # config_id maps to category_id, value maps to mode_id
            await session.agent.set_mode(params.value, category_id=params.config_id)

            # Return updated config options
            config_options = await get_session_config_options(session.agent)
            return SetSessionConfigOptionResponse(config_options=config_options)
        except Exception:
            logger.exception("Failed to set session config option", session_id=params.session_id)
            return None

    async def swap_pool(self, config_path: str, agent: str | None = None) -> list[str]:
        """Swap the agent pool with a new one from configuration.

        This coordinates the full pool swap:
        1. Closes all active sessions
        2. Delegates to server.swap_pool() for pool lifecycle
        3. Updates internal references

        Args:
            config_path: Path to the new agent configuration file
            agent: Optional specific agent to use as default

        Returns:
            List of agent names in the new pool

        Raises:
            RuntimeError: If server reference is not set
            ValueError: If config is invalid or agent not found
        """
        if not self.server:
            msg = "Server reference not set - cannot swap pool"
            raise RuntimeError(msg)

        logger.info("Swapping pool", config_path=config_path, agent=agent)
        # 1. Close all active sessions
        closed_count = await self.session_manager.close_all_sessions()
        logger.info("Closed sessions before pool swap", count=closed_count)
        # 2. Delegate to server for pool lifecycle management
        agent_names = await self.server.swap_pool(config_path, agent)
        # 3. Update internal references
        self.agent_pool = self.server.pool
        self.session_manager._pool = self.server.pool
        self.default_agent = agent
        logger.info("Pool swap complete", agent_names=agent_names)
        return agent_names
