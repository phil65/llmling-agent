"""ACP (Agent Client Protocol) server implementation for agentpool.

This module provides the main server class for exposing AgentPool via
the Agent Client Protocol.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
import functools
from typing import TYPE_CHECKING, Any, Self

import logfire

from acp import AgentSideConnection
from acp.stdio import stdio_streams
from agentpool import AgentPool
from agentpool.log import get_logger
from agentpool.models.manifest import AgentsManifest
from agentpool_server import BaseServer
from agentpool_server.acp_server.acp_agent import AgentPoolACPAgent


if TYPE_CHECKING:
    from collections.abc import Sequence

    from tokonomics.model_discovery import ProviderType
    from tokonomics.model_discovery.model_info import ModelInfo
    from upathtools import JoinablePathLike

    from acp.schema import ModelInfo as ACPModelInfo


logger = get_logger(__name__)


def _convert_to_acp_model_info(
    toko_models: Sequence[ModelInfo],
) -> list[ACPModelInfo]:
    """Convert tokonomics ModelInfo list to ACP ModelInfo list.

    Args:
        toko_models: List of tokonomics ModelInfo objects

    Returns:
        List of ACP ModelInfo objects with pydantic_ai_id as model_id
    """
    from acp.schema import ModelInfo as ACPModelInfo

    return [
        ACPModelInfo(
            model_id=model.pydantic_ai_id,
            name=f"{model.provider}: {model.name}" if model.provider else model.name,
            description=model.format(),
        )
        for model in toko_models
    ]


class ACPServer(BaseServer):
    """ACP (Agent Client Protocol) server for agentpool using external library.

    Provides a bridge between agentpool's Agent system and the standard ACP
    JSON-RPC protocol using the external acp library for robust communication.

    The actual client communication happens via the AgentSideConnection created
    when start() is called, which communicates with the external process over stdio.
    """

    def __init__(
        self,
        pool: AgentPool[Any],
        *,
        name: str | None = None,
        file_access: bool = True,
        terminal_access: bool = True,
        providers: list[ProviderType] | None = None,
        debug_messages: bool = False,
        debug_file: str | None = None,
        debug_commands: bool = False,
        agent: str | None = None,
        load_skills: bool = True,
        config_path: str | None = None,
    ) -> None:
        """Initialize ACP server with configuration.

        Args:
            pool: AgentPool containing available agents
            name: Optional Server name (auto-generated if None)
            file_access: Whether to support file access operations
            terminal_access: Whether to support terminal access operations
            providers: List of providers to use for model discovery (None = openrouter)
            debug_messages: Whether to enable debug message logging
            debug_file: File path for debug message logging
            debug_commands: Whether to enable debug slash commands for testing
            agent: Optional specific agent name to use (defaults to first agent)
            load_skills: Whether to load client-side skills from .claude/skills
            config_path: Path to the configuration file (for tracking/hot-switching)
        """
        super().__init__(pool, name=name, raise_exceptions=True)
        self.file_access = file_access
        self.terminal_access = terminal_access
        self.providers = providers or ["openai", "anthropic", "gemini"]
        self.debug_messages = debug_messages
        self.debug_file = debug_file
        self.debug_commands = debug_commands
        self.agent = agent
        self.load_skills = load_skills
        self.config_path = config_path

        self._available_models: list[ACPModelInfo] = []
        self._models_initialized = False

    @classmethod
    def from_config(
        cls,
        config_path: JoinablePathLike,
        *,
        file_access: bool = True,
        terminal_access: bool = True,
        providers: list[ProviderType] | None = None,
        debug_messages: bool = False,
        debug_file: str | None = None,
        debug_commands: bool = False,
        agent: str | None = None,
        load_skills: bool = True,
    ) -> Self:
        """Create ACP server from existing agentpool configuration.

        Args:
            config_path: Path to agentpool YAML config file
            file_access: Enable file system access
            terminal_access: Enable terminal access
            providers: List of provider types to use for model discovery
            debug_messages: Enable saving JSON messages to file
            debug_file: Path to debug file
            debug_commands: Enable debug slash commands for testing
            agent: Optional specific agent name to use (defaults to first agent)
            load_skills: Whether to load client-side skills from .claude/skills

        Returns:
            Configured ACP server instance with agent pool from config
        """
        manifest = AgentsManifest.from_file(config_path)
        pool = AgentPool(manifest=manifest)
        server = cls(
            pool,
            file_access=file_access,
            terminal_access=terminal_access,
            providers=providers,
            debug_messages=debug_messages,
            debug_file=debug_file or "acp-debug.jsonl" if debug_messages else None,
            debug_commands=debug_commands,
            agent=agent,
            load_skills=load_skills,
            config_path=str(config_path),
        )
        agent_names = list(server.pool.agents.keys())

        # Validate specified agent exists if provided
        if agent and agent not in agent_names:
            msg = f"Specified agent {agent!r} not found in config. Available agents: {agent_names}"
            raise ValueError(msg)

        server.log.info("Created ACP server with agent pool", agent_names=agent_names)
        if agent:
            server.log.info("ACP session agent", agent=agent)
        return server

    async def _start_async(self) -> None:
        """Start the ACP server (blocking async - runs until stopped)."""
        agent_names = list(self.pool.agents.keys())
        self.log.info("Starting ACP server on stdio", agent_names=agent_names)
        await self._initialize_models()  # Initialize models on first run
        create_acp_agent = functools.partial(
            AgentPoolACPAgent,
            agent_pool=self.pool,
            available_models=self._available_models,
            file_access=self.file_access,
            terminal_access=self.terminal_access,
            debug_commands=self.debug_commands,
            default_agent=self.agent,
            load_skills=self.load_skills,
            server=self,
        )
        reader, writer = await stdio_streams()
        file = self.debug_file if self.debug_messages else None
        conn = AgentSideConnection(create_acp_agent, writer, reader, debug_file=file)
        self.log.info("ACP server started", file=self.file_access, terminal=self.terminal_access)
        try:  # Keep the connection alive until shutdown
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            self.log.info("ACP server shutdown requested")
            raise
        except KeyboardInterrupt:
            self.log.info("ACP server shutdown requested")
        except Exception:
            self.log.exception("Connection receive task failed")
        finally:
            await conn.close()

    async def swap_pool(
        self,
        config_path: str,
        agent: str | None = None,
    ) -> list[str]:
        """Swap the current pool with a new one from config.

        This method handles the full lifecycle of swapping pools:
        1. Validates the new configuration
        2. Creates and initializes the new pool
        3. Cleans up the old pool
        4. Updates internal references

        Args:
            config_path: Path to the new agent configuration file
            agent: Optional specific agent to use as default

        Returns:
            List of agent names in the new pool

        Raises:
            ValueError: If config is invalid or specified agent not found
            FileNotFoundError: If config file doesn't exist
        """
        # 1. Parse and validate new config before touching current pool
        self.log.info("Loading new pool configuration", config_path=config_path)
        new_manifest = AgentsManifest.from_file(config_path)
        new_pool = AgentPool(manifest=new_manifest)

        # 2. Validate agent exists in new pool if specified
        agent_names = list(new_pool.all_agents.keys())
        if not agent_names:
            msg = "New configuration contains no agents"
            raise ValueError(msg)

        if agent and agent not in agent_names:
            msg = f"Agent {agent!r} not found in new config. Available: {agent_names}"
            raise ValueError(msg)

        # 3. Enter new pool context first (so we can roll back if it fails)
        try:
            await new_pool.__aenter__()
        except Exception as e:
            self.log.exception("Failed to initialize new pool")
            msg = f"Failed to initialize new pool: {e}"
            raise ValueError(msg) from e

        # 4. Exit old pool context
        old_pool = self.pool
        try:
            await old_pool.__aexit__(None, None, None)
        except Exception:
            self.log.exception("Error closing old pool (continuing with swap)")

        # 5. Update references
        self.pool = new_pool
        self.agent = agent
        self.config_path = config_path

        self.log.info("Pool swapped successfully", agent_names=agent_names, default_agent=agent)
        return agent_names

    @logfire.instrument("ACP: Initializing models.")
    async def _initialize_models(self) -> None:
        """Initialize available models using tokonomics model discovery.

        Converts tokonomics ModelInfo to ACP ModelInfo format at startup
        so all downstream code works with ACP types consistently.
        """
        from tokonomics.model_discovery import get_all_models

        if self._models_initialized:
            return
        try:
            self.log.info("Discovering available models...")
            delta = timedelta(days=200)
            toko_models = await get_all_models(providers=self.providers, max_age=delta)
            # Convert to ACP format once at startup
            self._available_models = _convert_to_acp_model_info(toko_models)
            self._models_initialized = True
            self.log.info("Discovered models", count=len(self._available_models))
        except Exception:
            self.log.exception("Failed to discover models")
            self._available_models = []
        finally:
            self._models_initialized = True
