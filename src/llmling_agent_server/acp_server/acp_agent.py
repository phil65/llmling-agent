"""ACP (Agent Client Protocol) Agent implementation."""

from __future__ import annotations

from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any

from acp import Agent as ACPAgent
from acp.schema import (
    InitializeResponse,
    LoadSessionResponse,
    ModelInfo as ACPModelInfo,
    NewSessionResponse,
    PromptResponse,
    SessionModelState,
    SessionModeState,
    SetSessionModelRequest,
    SetSessionModelResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
)
from llmling_agent.log import get_logger
from llmling_agent.utils.tasks import TaskManager
from llmling_agent_server.acp_server.converters import agent_to_mode
from llmling_agent_server.acp_server.session_manager import ACPSessionManager


if TYPE_CHECKING:
    from collections.abc import Sequence

    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo

    from acp import AgentSideConnection, Client
    from acp.schema import (
        AuthenticateRequest,
        CancelNotification,
        ClientCapabilities,
        InitializeRequest,
        LoadSessionRequest,
        NewSessionRequest,
        PromptRequest,
        SetSessionModelRequest,
        SetSessionModeRequest,
    )
    from llmling_agent import AgentPool

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
        file_access: bool = True,
        terminal_access: bool = True,
        debug_commands: bool = False,
        default_agent: str | None = None,
    ) -> None:
        """Initialize ACP agent implementation.

        Args:
            connection: ACP connection for client communication
            agent_pool: AgentPool containing available agents
            available_models: List of available tokonomics TokoModelInfo objects
            file_access: Whether agent can access filesystem
            terminal_access: Whether agent can use terminal
            debug_commands: Whether to enable debug slash commands for testing
            default_agent: Optional specific agent name to use as default
        """
        self.connection = connection
        self.agent_pool = agent_pool
        self.available_models: Sequence[TokoModelInfo] = available_models or []
        self.file_access = file_access
        self.terminal_access = terminal_access
        self.client: Client = connection
        self.debug_commands = debug_commands
        self.default_agent = default_agent
        self.client_capabilities: ClientCapabilities | None = None

        self.session_manager = ACPSessionManager()
        self.tasks = TaskManager()

        self._initialized = False
        agent_count = len(self.agent_pool.agents)
        logger.info("Created ACP agent implementation", agent_count=agent_count)
        if debug_commands:
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
            http_mcp_servers=True,
            sse_mcp_servers=True,
            audio_prompts=True,
            embedded_context_prompts=True,
            image_prompts=True,
        )
        logger.info("ACP agent initialized successfully", response=response)
        return response

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
                agent_pool=self.agent_pool,
                default_agent_name=default_name,
                cwd=params.cwd,
                client=self.client,
                mcp_servers=params.mcp_servers,
                acp_agent=self,
                client_capabilities=self.client_capabilities,
            )

            modes = [agent_to_mode(agent) for agent in self.agent_pool.all_agents.values()]
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
            if session := self.session_manager.get_session(session_id):
                # Schedule task to run after response is sent
                coro = session.send_available_commands_update()
                coro_2 = session.init_project_context()
                coro_3 = session._register_prompt_hub_commands()
                coro_4 = session.init_client_skills()
                self.tasks.create_task(coro, name=f"send_commands_update_{session_id}")
                self.tasks.create_task(coro_2, name=f"init_project_context_{session_id}")
                self.tasks.create_task(coro_3, name=f"init_prompthub_cmds_{session_id}")
                self.tasks.create_task(coro_4, name=f"init_client_skills_{session_id}")
            logger.info("Created session", session_id=session_id, agent_count=len(modes))
            return NewSessionResponse(session_id=session_id, modes=state, models=models)

    async def load_session(self, params: LoadSessionRequest) -> LoadSessionResponse:
        """Load an existing session."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

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
            try:
                assert session
                await session.notifications.send_agent_text(msg)
            except Exception:
                logger.exception("Failed to send error update")

            return PromptResponse(stop_reason="refusal")
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
        """Set the session mode (switch active agent).

        The mode ID corresponds to the agent name in the pool.
        """
        try:
            session = self.session_manager.get_session(params.session_id)
            if not session:
                msg = "Session not found for mode switch"
                logger.warning(msg, session_id=params.session_id)
                return None

            # Validate agent exists in pool
            if not self.agent_pool or params.mode_id not in self.agent_pool.agents:
                logger.error("Agent not found in pool", mode_id=params.mode_id)
                return None
            await session.switch_active_agent(params.mode_id)
            return SetSessionModeResponse()

        except Exception:
            logger.exception("Failed to set session mode", session_id=params.session_id)
            return None

    async def set_session_model(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse | None:
        """Set the session model.

        Changes the model for the active agent in the session.
        """
        try:
            session = self.session_manager.get_session(params.session_id)
            if not session:
                msg = "Session not found for model switch"
                logger.warning(msg, session_id=params.session_id)
                return None
            session.agent.set_model(params.model_id)
            logger.info(
                "Set model",
                model_id=params.model_id,
                session_id=params.session_id,
            )
            return SetSessionModelResponse()
        except Exception:
            logger.exception("Failed to set session model", session_id=params.session_id)
            return None
