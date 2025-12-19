"""ACP (Agent Client Protocol) server implementation for llmling-agent.

This module provides the main server class for exposing llmling agents via
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
from llmling_agent import AgentPool
from llmling_agent.log import get_logger
from llmling_agent.models.manifest import AgentsManifest
from llmling_agent_server import BaseServer
from llmling_agent_server.acp_server.acp_agent import LLMlingACPAgent


if TYPE_CHECKING:
    from tokonomics.model_discovery import ProviderType
    from tokonomics.model_discovery.model_info import ModelInfo
    from upathtools import JoinablePathLike


logger = get_logger(__name__)


class ACPServer(BaseServer):
    """ACP (Agent Client Protocol) server for llmling-agent using external library.

    Provides a bridge between llmling-agent's Agent system and the standard ACP
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

        self._available_models: list[ModelInfo] = []
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
        """Create ACP server from existing llmling-agent configuration.

        Args:
            config_path: Path to llmling-agent YAML config file
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
            LLMlingACPAgent,
            agent_pool=self.pool,
            available_models=self._available_models,
            file_access=self.file_access,
            terminal_access=self.terminal_access,
            debug_commands=self.debug_commands,
            default_agent=self.agent,
            load_skills=self.load_skills,
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

    @logfire.instrument("ACP: Initializing models.")
    async def _initialize_models(self) -> None:
        """Initialize available models using tokonomics model discovery."""
        from tokonomics.model_discovery import get_all_models

        if self._models_initialized:
            return
        try:
            self.log.info("Discovering available models...")
            delta = timedelta(days=200)
            self._available_models = await get_all_models(providers=self.providers, max_age=delta)
            self._models_initialized = True
            self.log.info("Discovered models", count=len(self._available_models))
        except Exception:
            self.log.exception("Failed to discover models")
            self._available_models = []
        finally:
            self._models_initialized = True
