"""ACP (Agent Client Protocol) server implementation for llmling-agent.

This module provides the main server class for exposing llmling agents via
the Agent Client Protocol.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING, Any, Self

import logfire

from acp import AgentSideConnection
from acp.stdio import stdio_streams
from llmling_agent import AgentPool
from llmling_agent.log import get_logger
from llmling_agent.models.manifest import AgentsManifest
from llmling_agent_acp.acp_agent import LLMlingACPAgent


if TYPE_CHECKING:
    from tokonomics.model_discovery import ProviderType
    from tokonomics.model_discovery.model_info import ModelInfo
    from upath.types import JoinablePathLike

    from llmling_agent import Agent
    from llmling_agent_providers.base import UsageLimits

logger = get_logger(__name__)


@dataclass
class ACPServer:
    """ACP (Agent Client Protocol) server for llmling-agent using external library.

    Provides a bridge between llmling-agent's Agent system and the standard ACP
    JSON-RPC protocol using the external acp library for robust communication.

    The actual client communication happens via the AgentSideConnection created
    when run() is called, which communicates with the external process over stdio.
    """

    agent_pool: AgentPool[Any]
    """AgentPool containing available agents"""

    usage_limits: UsageLimits | None = None
    """Optional usage limits for model requests and tokens"""

    session_support: bool = True
    """Whether to support session-based operations"""

    file_access: bool = True
    """Whether to support file access operations"""

    terminal_access: bool = True
    """Whether to support terminal access operations"""

    providers: list[ProviderType] | None = None
    """List of providers to use for model discovery (None = openrouter)"""

    debug_messages: bool = False
    """Whether to enable debug message logging"""

    debug_file: str | None = None
    """File path for debug message logging"""

    debug_commands: bool = False
    """Whether to enable debug slash commands for testing"""

    agent: str | None = None
    """Optional specific agent name to use (defaults to first agent)"""

    def __post_init__(self) -> None:
        """Initialize server configuration."""
        # Set default providers if None
        if self.providers is None:
            self.providers = ["openrouter"]

        self._running = False
        self._available_models: list[ModelInfo] = []
        self._models_initialized = False

    @classmethod
    async def from_config(
        cls,
        config_path: JoinablePathLike,
        *,
        usage_limits: UsageLimits | None = None,
        session_support: bool = True,
        file_access: bool = True,
        terminal_access: bool = True,
        providers: list[ProviderType] | None = None,
        debug_messages: bool = False,
        debug_file: str | None = None,
        debug_commands: bool = False,
        agent: str | None = None,
    ) -> Self:
        """Create ACP server from existing llmling-agent configuration.

        Args:
            config_path: Path to llmling-agent YAML config file
            usage_limits: Optional usage limits for model requests and tokens
            session_support: Enable session loading support
            file_access: Enable file system access
            terminal_access: Enable terminal access
            providers: List of provider types to use for model discovery
            debug_messages: Enable saving JSON messages to file
            debug_file: Path to debug file
            debug_commands: Enable debug slash commands for testing
            agent: Optional specific agent name to use (defaults to first agent)

        Returns:
            Configured ACP server instance with agent pool from config
        """
        manifest = AgentsManifest.from_file(config_path)
        pool = AgentPool(manifest=manifest)
        server = cls(
            agent_pool=pool,
            usage_limits=usage_limits,
            session_support=session_support,
            file_access=file_access,
            terminal_access=terminal_access,
            providers=providers,
            debug_messages=debug_messages,
            debug_file=debug_file,
            debug_commands=debug_commands,
            agent=agent,
        )
        agent_names = list(server.agent_pool.agents.keys())

        # Validate specified agent exists if provided
        if agent and agent not in agent_names:
            msg = (
                f"Specified agent {agent!r} not found in config. "
                "Available agents: {agent_names}"
            )
            raise ValueError(msg)

        logger.info("Created ACP server with agent pool containing: %s", agent_names)
        if agent:
            logger.info("Will use agent '%s' for ACP sessions", agent)
        return server

    def get_agent(self, name: str) -> Agent[Any]:
        """Get agent by name from the pool."""
        return self.agent_pool.get_agent(name)

    async def run(self) -> None:
        """Run the ACP server."""
        if self._running:
            msg = "Server is already running"
            raise RuntimeError(msg)
        self._running = True

        try:
            await self._initialize_models()  # Initialize models on first run
            agent_names = list(self.agent_pool.agents.keys())
            msg = "Starting ACP server with %d agents on stdio: %s"
            logger.info(msg, len(agent_names), agent_names)

            create_acp_agent = functools.partial(
                LLMlingACPAgent,
                agent_pool=self.agent_pool,
                available_models=self._available_models,
                session_support=self.session_support,
                file_access=self.file_access,
                terminal_access=self.terminal_access,
                usage_limits=self.usage_limits,
                debug_commands=self.debug_commands,
                default_agent=self.agent,
            )

            reader, writer = await stdio_streams()
            file = self.debug_file if self.debug_messages else None
            conn = AgentSideConnection(create_acp_agent, writer, reader, debug_file=file)
            logger.info(
                "ACP server started: file_access=%s, terminal=%s, session_support=%s",
                self.file_access,
                self.terminal_access,
                self.session_support,
            )
            try:  # Keep the connection alive
                while self._running:
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("ACP server shutdown requested")
            except Exception:
                logger.exception("Connection receive task failed")
            finally:
                await conn.close()
        except Exception:
            logger.exception("Error running ACP server")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the ACP server and cleanup resources."""
        if not self._running:
            msg = "Server is not running"
            raise RuntimeError(msg)

        self._running = False
        logger.info("Shutting down ACP server")

        try:
            await self.agent_pool.__aexit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to cleanup agent pool: %s", e)

        logger.info("ACP server shutdown complete")

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.agent_pool.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Async context manager exit."""
        await self.shutdown()

    @logfire.instrument("ACP: Initializing models.")
    async def _initialize_models(self) -> None:
        """Initialize available models using tokonomics model discovery."""
        from tokonomics.model_discovery import get_all_models

        if self._models_initialized:
            return
        try:
            logger.info("Discovering available models...")
            self._available_models = await get_all_models(providers=self.providers)
            self._models_initialized = True
            logger.info("Discovered %d models", len(self._available_models))
        except Exception:
            logger.exception("Failed to discover models")
            self._available_models = []
        finally:
            self._models_initialized = True
