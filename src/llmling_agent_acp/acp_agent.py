"""ACP (Agent Client Protocol) Agent implementation."""

from __future__ import annotations

from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any

from slashed import CommandStore

from acp import Agent as ACPAgent
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AuthenticateRequest,
    AuthenticateResponse,
    CustomRequest,
    CustomResponse,
    Implementation,
    InitializeRequest,
    InitializeResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    McpCapabilities,
    ModelInfo as ACPModelInfo,
    NewSessionRequest,
    NewSessionResponse,
    PromptCapabilities,
    PromptRequest,
    PromptResponse,
    SessionModelState,
    SessionModeState,
    SessionNotification,
    SetSessionModelRequest,
    SetSessionModelResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
    TextContentBlock,
)
from llmling_agent.log import get_logger
from llmling_agent.utils.tasks import TaskManager
from llmling_agent_acp.command_bridge import ACPCommandBridge
from llmling_agent_acp.commands.acp_commands import get_acp_commands
from llmling_agent_acp.converters import agent_to_mode
from llmling_agent_acp.session_manager import ACPSessionManager
from llmling_agent_commands import get_commands


if TYPE_CHECKING:
    from collections.abc import Sequence

    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo

    from acp import AgentSideConnection, Client
    from acp.schema import (
        AgentResponse,
        CancelNotification,
        ClientCapabilities,
        ClientRequest,
    )
    from llmling_agent import AgentPool
    from llmling_agent_acp.session import ACPSession
    from llmling_agent_providers.base import UsageLimits

logger = get_logger(__name__)


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
            name=f"{model.provider}: {model.name}",
            description=model.format(),
        )
        for model in available_models
    ]
    # Use first model as current if not specified
    all_ids = [model.pydantic_ai_id for model in available_models]
    current_model_id = current_model if current_model in all_ids else all_ids[0]
    return SessionModelState(available_models=models, current_model_id=current_model_id)


class LLMlingACPAgent(ACPAgent):
    """Implementation of ACP Agent protocol interface for llmling agents.

    This class implements the external library's Agent protocol interface,
    bridging llmling agents with the standard ACP JSON-RPC protocol.
    """

    PROTOCOL_VERSION = 1

    def __init__(
        self,
        connection: AgentSideConnection,
        agent_pool: AgentPool[Any],
        *,
        available_models: list[TokoModelInfo] | None = None,
        session_support: bool = True,
        file_access: bool = True,
        terminal_access: bool = True,
        usage_limits: UsageLimits | None = None,
        debug_commands: bool = False,
        default_agent: str | None = None,
    ) -> None:
        """Initialize ACP agent implementation.

        Args:
            connection: ACP connection for client communication
            agent_pool: AgentPool containing available agents
            available_models: List of available tokonomics TokoModelInfo objects
            session_support: Whether agent supports session loading
            file_access: Whether agent can access filesystem
            terminal_access: Whether agent can use terminal
            usage_limits: Optional usage limits for model requests and tokens
            debug_commands: Whether to enable debug slash commands for testing
            default_agent: Optional specific agent name to use as default
        """
        self.connection = connection
        self.agent_pool = agent_pool
        self.available_models: Sequence[TokoModelInfo] = available_models or []
        self.session_support = session_support
        self.file_access = file_access
        self.terminal_access = terminal_access
        self.client: Client = connection
        self.usage_limits = usage_limits
        self.debug_commands = debug_commands
        self.default_agent = default_agent
        self.client_capabilities: ClientCapabilities | None = None
        command_store = CommandStore(enable_system_commands=True)
        command_store._initialize_sync()

        commands_to_register = [*get_commands(), *get_acp_commands()]
        if debug_commands:
            from llmling_agent_acp.commands.debug_commands import get_debug_commands

            commands_to_register.extend(get_debug_commands())

        for command in commands_to_register:
            command_store.register_command(command)
        self.command_bridge = ACPCommandBridge(command_store)
        self.session_manager = ACPSessionManager(command_bridge=self.command_bridge)
        self.tasks = TaskManager()

        self._initialized = False
        agent_count = len(self.agent_pool.agents)
        logger.info("Created ACP agent implementation", agent_count=agent_count)
        if debug_commands:
            logger.info("Debug slash commands enabled for ACP testing")

        # Note: Tool registration happens after initialize() when we know client caps

    async def handle_request(self, request: ClientRequest) -> AgentResponse:  # noqa: PLR0911
        """Unified request handler with type-safe dispatch."""
        match request:
            case InitializeRequest() as req:
                return await self._handle_initialize(req)
            case NewSessionRequest() as req:
                return await self._handle_new_session(req)
            case LoadSessionRequest() as req:
                return await self._handle_load_session(req)
            case AuthenticateRequest() as req:
                return await self._handle_authenticate(req)
            case PromptRequest() as req:
                return await self._handle_prompt(req)
            case SetSessionModeRequest() as req:
                return await self._handle_set_session_mode(req)
            case SetSessionModelRequest() as req:
                return await self._handle_set_session_model(req)
            case CustomRequest() as req:
                return await self._handle_custom_request(req)
            case _:
                msg = f"Unsupported request type: {type(request)}"
                raise TypeError(msg)

    async def _handle_initialize(self, params: InitializeRequest) -> InitializeResponse:
        """Initialize the agent and negotiate capabilities."""
        logger.info("Initializing ACP agent implementation")
        version = min(params.protocol_version, self.PROTOCOL_VERSION)
        self.client_capabilities = params.client_capabilities
        logger.info("Client capabilities", capabilities=self.client_capabilities)
        prompt_caps = PromptCapabilities(audio=True, embedded_context=True, image=True)
        mcp_caps = McpCapabilities(http=True, sse=True)
        caps = AgentCapabilities(
            load_session=self.session_support,
            prompt_capabilities=prompt_caps,
            mcp_capabilities=mcp_caps,
        )
        self._initialized = True
        impl = Implementation(
            name="llmling-agent",
            title="LLMLing-Agent",
            version=_version("llmling-agent"),
        )
        response = InitializeResponse(
            protocol_version=version,
            agent_capabilities=caps,
            agent_info=impl,
        )
        logger.info("ACP agent initialized successfully", response=response)
        return response

    async def _handle_new_session(self, params: NewSessionRequest) -> NewSessionResponse:
        """Create a new session."""
        if not self._initialized:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        try:
            agent_names = list(self.agent_pool.agents.keys())
            if not agent_names:
                logger.error("No agents available for session creation")
                msg = "No agents available"
                raise RuntimeError(msg)  # noqa: TRY301

            # Use specified default agent or fall back to first agent
            if self.default_agent and self.default_agent in agent_names:
                default_name = self.default_agent
            else:
                default_name = agent_names[0]

            logger.info(
                "Creating new session",
                available_agents=agent_names,
                default_agent=default_name,
            )
            session_id = await self.session_manager.create_session(
                agent_pool=self.agent_pool,
                default_agent_name=default_name,
                cwd=params.cwd,
                client=self.client,
                mcp_servers=params.mcp_servers,
                usage_limits=self.usage_limits,
                acp_agent=self,
                client_capabilities=self.client_capabilities,
            )

            modes = [agent_to_mode(agent) for agent in self.agent_pool.agents.values()]
            state = SessionModeState(current_mode_id=default_name, available_modes=modes)
            # Get model information from the default agent
            if session := self.session_manager.get_session(session_id):
                current_model = session.agent.model_name
                models = create_session_model_state(self.available_models, current_model)
            else:
                models = None
        except Exception:
            logger.exception("Failed to create new session")
            raise
        else:
            # Schedule available commands update after session response is returned
            session = self.session_manager.get_session(session_id)
            if session:
                # Schedule task to run after response is sent
                coro = session.send_available_commands_update()
                coro_2 = session.init_project_context()
                self.tasks.create_task(coro, name=f"send_commands_update_{session_id}")
                self.tasks.create_task(coro_2, name=f"init_project_context_{session_id}")
            logger.info("Created session", session_id=session_id, agent_count=len(modes))
            return NewSessionResponse(session_id=session_id, modes=state, models=models)

    async def _handle_load_session(
        self, params: LoadSessionRequest
    ) -> LoadSessionResponse:
        """Load an existing session."""
        if not self._initialized:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        try:
            session = self.session_manager.get_session(params.session_id)
            if not session:
                logger.warning("Session not found", session_id=params.session_id)
                return LoadSessionResponse()

            current_model = session.agent.model_name if session.agent else None
            models = create_session_model_state(self.available_models, current_model)

            return LoadSessionResponse(models=models)
        except Exception:
            logger.exception("Failed to load session", session_id=params.session_id)
            return LoadSessionResponse()

    async def _handle_authenticate(
        self, params: AuthenticateRequest
    ) -> AuthenticateResponse:
        """Authenticate with the agent."""
        logger.info("Authentication requested", method_id=params.method_id)
        return AuthenticateResponse()

    async def _handle_prompt(self, params: PromptRequest) -> PromptResponse:
        """Process a prompt request."""
        if not self._initialized:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        logger.info("Processing prompt", session_id=params.session_id)
        session = self.session_manager.get_session(params.session_id)
        try:
            if not session:
                msg = f"Session {params.session_id} not found"
                raise ValueError(msg)  # noqa: TRY301
            stop_reason = await session.process_prompt(params.prompt)
            # Return the actual stop reason from the session
            response = PromptResponse(stop_reason=stop_reason)
            logger.info("Returning PromptResponse", stop_reason=stop_reason)
        except Exception as e:
            logger.exception("Failed to process prompt", session_id=params.session_id)
            msg = f"Error processing prompt: {e}"
            chunk = AgentMessageChunk(content=TextContentBlock(text=msg))
            notification = SessionNotification(session_id=params.session_id, update=chunk)
            try:
                await self.connection.session_update(notification)
            except Exception:
                logger.exception("Failed to send error update")

            return PromptResponse(stop_reason="refusal")
        else:
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

    async def _handle_custom_request(self, request: CustomRequest) -> CustomResponse:
        """Handle custom extension requests."""
        return CustomResponse(data={"example": "response", "method": request.method})

    async def _handle_set_session_mode(
        self, params: SetSessionModeRequest
    ) -> SetSessionModeResponse:
        """Set the session mode (switch active agent).

        The mode ID corresponds to the agent name in the pool.
        """

        def _validate_session(sess: ACPSession | None) -> None:
            if not sess:
                msg = "Session not found for mode switch"
                logger.warning(msg, session_id=params.session_id)
                raise RuntimeError(msg)

        def _validate_agent() -> None:
            if not self.agent_pool or params.mode_id not in self.agent_pool.agents:
                msg = f"Agent not found in pool: {params.mode_id}"
                logger.error(msg, mode_id=params.mode_id)
                raise RuntimeError(msg)

        try:
            session = self.session_manager.get_session(params.session_id)
            _validate_session(session)
            _validate_agent()
            assert session is not None  # For mypy
            await session.switch_active_agent(params.mode_id)
            return SetSessionModeResponse()

        except Exception:
            logger.exception("Failed to set session mode", session_id=params.session_id)
            raise

    async def _handle_set_session_model(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse:
        """Set the session model.

        Changes the model for the active agent in the session.
        """

        def _validate_session(sess: ACPSession | None) -> None:
            if not sess:
                msg = "Session not found for model switch"
                logger.warning(msg, session_id=params.session_id)
                raise RuntimeError(msg)

        try:
            session = self.session_manager.get_session(params.session_id)
            _validate_session(session)
            assert session is not None  # For mypy
            session.agent.set_model(params.model_id)
            logger.info(
                "Set model",
                model_id=params.model_id,
                session_id=params.session_id,
            )
            return SetSessionModelResponse()
        except Exception:
            logger.exception("Failed to set session model", session_id=params.session_id)
            raise
