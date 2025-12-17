"""ACP (Agent Client Protocol) Agent implementation."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any, ClassVar

from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo

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
    SessionInfo,
    SessionModelState,
    SessionModeState,
    SetSessionModelRequest,
    SetSessionModelResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
)
from llmling_agent import Agent
from llmling_agent.log import get_logger
from llmling_agent.utils.tasks import TaskManager
from llmling_agent_server.acp_server.converters import (
    # agent_to_mode,  # TODO: Re-enable when supporting agent switching via modes
    get_confirmation_modes,
    mode_id_to_confirmation_mode,
)
from llmling_agent_server.acp_server.session_manager import ACPSessionManager


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import ModelRequest, ModelResponse

    from acp import AgentSideConnection, Client
    from acp.schema import (
        AuthenticateRequest,
        CancelNotification,
        ClientCapabilities,
        ForkSessionRequest,
        InitializeRequest,
        ListSessionsRequest,
        LoadSessionRequest,
        NewSessionRequest,
        PromptRequest,
        ResumeSessionRequest,
        SetSessionModelRequest,
        SetSessionModeRequest,
    )
    from llmling_agent import AgentPool
    from llmling_agent_server.acp_server.session import ACPSession

logger = get_logger(__name__)


def to_tokonomics_model_info(model_info: ACPModelInfo) -> TokoModelInfo:
    if len(model_info.model_id.split(":")) > 1:
        provider, model_id = model_info.model_id.split(":")
    else:
        provider = ""
        model_id = model_info.model_id
    return TokoModelInfo(
        id=model_id,
        provider=provider,
        name=model_info.name,
        description=model_info.description,
    )


def create_session_model_state(
    available_models: Sequence[TokoModelInfo], current_model: str | None = None
) -> SessionModelState | None:
    """Create a SessionModelState from available models.

    Args:
        available_models: List of all models the agent can switch between
        current_model: The currently active model (defaults to first available)

    Returns:
        SessionModelState with all available models, None if no models provided
    """
    if not available_models:
        return None
    # Create ModelInfo objects for each available model
    models = [
        ACPModelInfo(
            model_id=model.pydantic_ai_id,
            name=f"{model.provider}: {model.name}" if model.provider else model.name,
            description=model.format(),
        )
        for model in available_models
    ]
    # Use first model as current if not specified
    all_ids = [model.pydantic_ai_id for model in available_models]
    current_model_id = current_model if current_model in all_ids else all_ids[0]
    return SessionModelState(available_models=models, current_model_id=current_model_id)


@dataclass
class LLMlingACPAgent(ACPAgent):
    """Implementation of ACP Agent protocol interface for llmling agents.

    This class implements the external library's Agent protocol interface,
    bridging llmling agents with the standard ACP JSON-RPC protocol.
    """

    PROTOCOL_VERSION: ClassVar = 1

    connection: AgentSideConnection
    """ACP connection for client communication."""

    agent_pool: AgentPool[Any]
    """AgentPool containing available agents."""

    _: KW_ONLY

    available_models: Sequence[TokoModelInfo] = field(default_factory=list)
    """List of available tokonomics TokoModelInfo objects."""

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

    def __post_init__(self) -> None:
        """Initialize derived attributes and setup after field assignment."""
        self.client: Client = self.connection
        self.client_capabilities: ClientCapabilities | None = None
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
        logger.info("Initializing ACP agent implementation")
        version = min(params.protocol_version, self.PROTOCOL_VERSION)
        self.client_capabilities = params.client_capabilities
        logger.info("Client capabilities", capabilities=self.client_capabilities)
        self._initialized = True
        response = InitializeResponse.create(
            protocol_version=version,
            name="llmling-agent",
            title="LLMLing-Agent",
            version=_version("llmling-agent"),
            load_session=True,
            list_sessions=True,
            http_mcp_servers=True,
            sse_mcp_servers=True,
            audio_prompts=True,
            embedded_context_prompts=True,
            image_prompts=True,
        )
        logger.info("ACP agent initialized successfully", response=response)
        return response

    async def new_session(self, params: NewSessionRequest) -> NewSessionResponse:  # noqa: PLR0915
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
            )

            # Get mode information - pass through from ACPAgent or use our confirmation modes
            from llmling_agent.agents.acp_agent import ACPAgent as ACPAgentClient

            if session := self.session_manager.get_session(session_id):
                if isinstance(session.agent, ACPAgentClient):
                    # Pass through nested agent's modes (e.g., Claude Code's modes)
                    if session.agent._state and session.agent._state.modes:
                        state = session.agent._state.modes
                        modes = state.available_modes
                    else:
                        # Fallback to our confirmation modes if nested agent has none
                        modes = get_confirmation_modes()
                        state = SessionModeState(current_mode_id="default", available_modes=modes)
                else:
                    # Native Agent - use our tool confirmation modes
                    modes = get_confirmation_modes()
                    state = SessionModeState(current_mode_id="default", available_modes=modes)
            else:
                modes = get_confirmation_modes()
                state = SessionModeState(current_mode_id="default", available_modes=modes)
            # TODO: Re-enable agent switching via modes when needed:
            # modes = [agent_to_mode(agent) for agent in self.agent_pool.all_agents.values()]
            # state = SessionModeState(current_mode_id=default_name, available_modes=modes)

            # Get model information from the default agent
            if session := self.session_manager.get_session(session_id):
                if isinstance(session.agent, ACPAgentClient):
                    current_model = (
                        session.agent._state.current_model_id if session.agent._state else None
                    )
                    all_models = (
                        session.agent._state.models.available_models
                        if session.agent._state and session.agent._state.models
                        else []
                    )
                    converted = [to_tokonomics_model_info(i) for i in all_models]
                    models = create_session_model_state(converted, current_model)
                else:
                    current_model = session.agent.model_name
                    models = create_session_model_state(self.available_models, current_model)
            else:
                models = None
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
            logger.info("Created session", session_id=session_id, agent_count=len(modes))
            return NewSessionResponse(session_id=session_id, modes=state, models=models)

    async def load_session(self, params: LoadSessionRequest) -> LoadSessionResponse:
        """Load an existing session from storage.

        This tries to:
        1. Check if session is already active
        2. If not, try to resume from persistent storage
        3. Initialize MCP servers and other session resources
        4. Replay conversation history via ACP notifications
        5. Return session state (modes, models)
        """
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

            # Build response with current session state
            # Get mode information - pass through from ACPAgent or use our confirmation modes
            from llmling_agent.agents.acp_agent import ACPAgent as ACPAgentClient

            if isinstance(session.agent, ACPAgentClient):
                # Pass through nested agent's modes (e.g., Claude Code's modes)
                if session.agent._state and session.agent._state.modes:
                    mode_state = session.agent._state.modes
                else:
                    # Fallback to our confirmation modes if nested agent has none
                    modes = get_confirmation_modes()
                    mode_state = SessionModeState(current_mode_id="default", available_modes=modes)
            else:
                # Native Agent - use our tool confirmation modes
                modes = get_confirmation_modes()
                mode_state = SessionModeState(current_mode_id="default", available_modes=modes)
            # TODO: Re-enable agent switching via modes when needed:
            # modes = [agent_to_mode(agent) for agent in self.agent_pool.all_agents.values()]
            # mode_state = SessionModeState(
            #     current_mode_id=session.current_agent_name,
            #     available_modes=modes,
            # )

            current_model = session.agent.model_name if session.agent else None
            models = create_session_model_state(self.available_models, current_model)
            # Schedule post-load initialization tasks
            self.tasks.create_task(session.send_available_commands_update())
            self.tasks.create_task(session.init_project_context())
            # Replay conversation history via ACP notifications
            self.tasks.create_task(self._replay_conversation_history(session))
            logger.info("Session loaded successfully", agent=session.current_agent_name)
            return LoadSessionResponse(models=models, modes=mode_state)

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
        from llmling_agent_config.session import SessionQuery

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
                session.cancel()
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

        Maps ACP mode IDs to ToolConfirmationMode and calls set_tool_confirmation_mode
        on the session's agent. Each agent type handles the mode change appropriately:
        - Agent/AGUIAgent: Updates local confirmation mode
        - ACPAgent: Updates local mode AND forwards to remote ACP server

        Mode mappings:
        - "default": per_tool (confirm tools marked as requiring it)
        - "acceptEdits": never (auto-approve all tool calls)
        """
        from llmling_agent.agents.acp_agent import ACPAgent as ACPAgentClient

        try:
            session = self.session_manager.get_session(params.session_id)
            if not session:
                logger.warning("Session not found for mode switch", session_id=params.session_id)
                return None

            # Map mode_id to confirmation mode
            confirmation_mode = mode_id_to_confirmation_mode(params.mode_id)
            if not confirmation_mode:
                logger.error("Invalid mode_id", mode_id=params.mode_id)
                return None

            # All agent types support set_tool_confirmation_mode
            # ACPAgent handles forwarding to remote server internally
            await session.agent.set_tool_confirmation_mode(confirmation_mode)

            # Update stored mode state for ACPAgent
            if (
                isinstance(session.agent, ACPAgentClient)
                and session.agent._state
                and session.agent._state.modes
            ):
                session.agent._state.modes.current_mode_id = params.mode_id

            logger.info(
                "Set tool confirmation mode",
                mode_id=params.mode_id,
                confirmation_mode=confirmation_mode,
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
        Validates that the requested model is in the available models list:
        - For native Agent: validates against server's available_models
        - For ACPAgent: validates against the nested agent's model list
        """
        from llmling_agent.agents.acp_agent import ACPAgent as ACPAgentClient

        try:
            session = self.session_manager.get_session(params.session_id)
            if not session:
                msg = "Session not found for model switch"
                logger.warning(msg, session_id=params.session_id)
                return None

            # Validate model based on agent type
            if isinstance(session.agent, ACPAgentClient):
                # For ACPAgent, validate against nested agent's model list
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
                # For now, we can't actually change the model on ACPAgent
                logger.warning(
                    "Model change for ACPAgent not yet supported (ACP protocol UNSTABLE)",
                    model_id=params.model_id,
                )
                return None

            if isinstance(session.agent, Agent):
                # For native Agent, validate against server's available models
                available_ids = [m.pydantic_ai_id for m in self.available_models]
                if params.model_id not in available_ids:
                    logger.warning(
                        "Model not in available models",
                        model_id=params.model_id,
                        available=available_ids,
                    )
                    return None
                session.agent.set_model(params.model_id)

            logger.info("Set model", model_id=params.model_id, session_id=params.session_id)
            return SetSessionModelResponse()
        except Exception:
            logger.exception("Failed to set session model", session_id=params.session_id)
            return None
