"""ACP Agent - MessageNode wrapping an external ACP subprocess.

This module provides an agent implementation that communicates with external
ACP (Agent Client Protocol) servers via stdio, enabling integration of any
ACP-compatible agent into the agentpool pool.

The ACPAgent class acts as an ACP client, spawning an ACP server subprocess
and communicating with it via JSON-RPC over stdio. This allows:
- Integration of external ACP-compatible agents (like claude-code-acp)
- Composition with native agents via connections, teams, etc.
- Full ACP protocol support including file operations and terminals

Example:
    ```python
    from agentpool.models.acp_agents import ACPAgentConfig

    config = ACPAgentConfig(
        command="claude-code-acp",
        name="claude_coder",
        cwd="/path/to/project",
    )
    agent = ACPAgent.from_config(config)
    async with agent:
        result = await agent.run("Write a hello world program")
        print(result.content)
    ```
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self
import uuid

import anyio
from pydantic_ai import ModelRequest, ModelResponse, TextPart, UserPromptPart

from acp import InitializeRequest
from agentpool.agents.acp_agent.acp_converters import event_to_part
from agentpool.agents.acp_agent.session_state import ACPSessionState
from agentpool.agents.base_agent import BaseAgent
from agentpool.agents.events import (
    RunStartedEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolResultMetadataEvent,
)
from agentpool.agents.events.processors import FileTracker
from agentpool.agents.exceptions import (
    AgentNotInitializedError,
    UnknownCategoryError,
    UnknownModeError,
)
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.utils.streams import merge_queue_into_iterator
from agentpool.utils.subprocess_utils import SubprocessError, run_with_process_monitor
from agentpool.utils.token_breakdown import calculate_usage_from_parts


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
    from types import TracebackType

    from anyio.abc import Process
    from evented_config import EventConfig
    from exxec import ExecutionEnvironment
    from pydantic_ai import ThinkingPart, ToolCallPart, UserContent
    from slashed import BaseCommand
    from tokonomics.model_discovery.model_info import ModelInfo

    from acp.agent.protocol import Agent as ACPAgentProtocol
    from acp.client.connection import ClientSideConnection
    from acp.client.protocol import Client
    from acp.schema import Implementation, RequestPermissionRequest, RequestPermissionResponse
    from acp.schema.capabilities import AgentCapabilities
    from acp.schema.mcp import McpServer
    from agentpool.agents.acp_agent.client_handler import ACPClientHandler
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory
    from agentpool.common_types import AnyEventHandlerType
    from agentpool.delegation import AgentPool
    from agentpool.hooks import AgentHooks
    from agentpool.messaging import MessageHistory
    from agentpool.models.acp_agents import BaseACPAgentConfig
    from agentpool.resource_providers import ResourceProvider
    from agentpool.sessions import SessionData
    from agentpool.ui.base import InputProvider
    from agentpool_config.mcp_server import MCPServerConfig

logger = get_logger(__name__)


def get_updated_at(date_str: str | None) -> datetime:
    from agentpool.utils.time_utils import get_now

    updated_at = get_now()
    if date_str:
        with contextlib.suppress(ValueError, AttributeError):
            updated_at = datetime.fromisoformat(date_str)
    return updated_at


class ACPAgent[TDeps = None](BaseAgent[TDeps, str]):
    """MessageNode that wraps an external ACP agent subprocess.

    This allows integrating any ACP-compatible agent into the agentpool
    pool, enabling composition with native agents via connections, teams, etc.
    """

    AGENT_TYPE: ClassVar = "acp"

    def __init__(
        self,
        *,
        # Required
        command: str,
        args: list[str] | None = None,
        # Identity
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        provider_type: str = "acp",
        # Environment
        cwd: str | None = None,
        env_vars: dict[str, str] | None = None,
        execution_env: ExecutionEnvironment | None = None,
        client_execution_env: ExecutionEnvironment | None = None,
        # ACP initialization
        init_request: InitializeRequest | None = None,
        # Tools
        tool_providers: list[ResourceProvider] | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        # Runtime options
        deps_type: type[TDeps] | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        enable_logging: bool = True,
        event_configs: Sequence[EventConfig] | None = None,
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        auto_approve: bool = False,
        commands: Sequence[BaseCommand] | None = None,
        hooks: AgentHooks | None = None,
        session_id: str | None = None,
    ) -> None:
        from agentpool.mcp_server.tool_bridge import ToolManagerBridge

        super().__init__(
            name=name or command,
            description=description,
            deps_type=deps_type,
            display_name=display_name,
            mcp_servers=mcp_servers,
            agent_pool=agent_pool,
            enable_logging=enable_logging,
            event_configs=event_configs,
            env=execution_env,
            input_provider=input_provider,
            event_handlers=event_handlers,
            commands=commands,
            hooks=hooks,
        )
        # Permission handling
        self.auto_approve = auto_approve
        # Command
        self._command = command
        self._args = args or []
        # Environment
        self._cwd = cwd
        self._env_vars = env_vars or {}
        self._client_env = client_execution_env
        # ACP initialization
        self._init_request = init_request or self._create_default_init_request()
        # Tools
        self._tool_providers = tool_providers or []
        # Provider type for model messages
        self._provider_type = provider_type
        # ACP-specific state
        self.acp_permission_callback: (
            Callable[[RequestPermissionRequest], Awaitable[RequestPermissionResponse]] | None
        ) = None
        self._process: Process | None = None
        self._connection: ClientSideConnection | None = None
        self._client_handler: ACPClientHandler | None = None
        self._agent_info: Implementation | None = None
        self._caps: AgentCapabilities | None = None
        self._sdk_session_id: str | None = session_id
        self._state: ACPSessionState | None = None
        self._extra_mcp_servers: list[McpServer] = []
        self._sessions_cache: list[SessionData] | None = None
        # Create bridge (not started yet) - pass injection_manager for mid-run injection support
        self._tool_bridge = ToolManagerBridge(node=self, injection_manager=self._injection_manager)
        # Track the prompt task for cancellation
        self._prompt_task: asyncio.Task[Any] | None = None

    @staticmethod
    def _create_default_init_request(
        allow_terminal: bool = True,
        allow_file_operations: bool = True,
    ) -> InitializeRequest:
        """Create default ACP initialize request."""
        from importlib.metadata import metadata

        pkg_meta = metadata("agentpool")
        return InitializeRequest.create(
            title=pkg_meta["Name"],
            version=pkg_meta["Version"],
            name="agentpool",
            protocol_version=1,
            terminal=allow_terminal,
            read_text_file=allow_file_operations,
            write_text_file=allow_file_operations,
        )

    @classmethod
    def from_config(
        cls,
        config: BaseACPAgentConfig,
        *,
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        input_provider: InputProvider | None = None,
        deps_type: type[TDeps] | None = None,
        agent_pool: AgentPool[Any] | None = None,
    ) -> Self:
        """Create an ACPAgent from a config object."""
        # Merge config-level handlers with provided handlers
        config_handlers = config.get_event_handlers()
        merged_handlers: list[AnyEventHandlerType] = [
            *config_handlers,
            *(event_handlers or []),
        ]
        return cls(
            command=config.get_command(),
            args=config.get_args(),
            # Identity
            name=config.name,
            description=config.description,
            display_name=config.display_name,
            provider_type=config.type,
            # Environment
            cwd=config.cwd,
            env_vars=config.env_vars,
            execution_env=config.get_execution_environment(),
            client_execution_env=config.get_client_execution_environment(),
            # ACP initialization
            init_request=cls._create_default_init_request(
                allow_terminal=config.allow_terminal,
                allow_file_operations=config.allow_file_operations,
            ),
            # Tools
            tool_providers=config.get_tool_providers(),
            mcp_servers=config.mcp_servers,
            # Runtime options
            event_handlers=merged_handlers or None,
            event_configs=list(config.triggers),
            input_provider=input_provider,
            agent_pool=agent_pool,
            deps_type=deps_type,
            auto_approve=config.auto_approve,
            hooks=config.hooks.get_agent_hooks() if config.hooks else None,
        )

    @property
    def client_env(self) -> ExecutionEnvironment:
        """Execution environment for handling subprocess requests.

        This is used by ACPClientHandler for file/terminal operations requested
        by the subprocess. Falls back to the agent's main env if not explicitly set.
        """
        return self._client_env if self._client_env is not None else self.env

    async def _setup_toolsets(self) -> None:
        """Initialize toolsets and start bridge if needed."""
        if not self._tool_providers:
            return
        # Add all tool providers to tool manager
        for provider in self._tool_providers:
            self.tools.add_provider(provider)
        await self._tool_bridge.start()
        mcp_config = self._tool_bridge.get_mcp_server_config()
        self._extra_mcp_servers.append(mcp_config)

    async def __aenter__(self) -> Self:
        """Start subprocess and initialize ACP connection."""
        await super().__aenter__()
        await self._setup_toolsets()
        process = await self._start_process()
        try:
            await run_with_process_monitor(process, self._initialize, context="ACP initialization")
            # Load existing session or create new one
            if session_to_load := self._sdk_session_id:
                self._sdk_session_id = None
                result = await run_with_process_monitor(
                    process,
                    lambda: self.load_session(session_to_load),
                    context="ACP session load",
                )
                if result is None:
                    self.log.warning(
                        "Failed to load session, creating new one",
                        session_id=session_to_load,
                    )
                    await run_with_process_monitor(
                        process, self._create_session, context="ACP session creation"
                    )
            else:
                await run_with_process_monitor(
                    process, self._create_session, context="ACP session creation"
                )
        except SubprocessError as e:
            raise RuntimeError(str(e)) from e
        await anyio.sleep(0.3)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up subprocess and connection."""
        await self._cleanup()
        await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _start_process(self) -> Process:
        """Start the ACP server subprocess."""
        args = self._args
        cmd = [self._command, *args]
        self.log.info("Starting ACP subprocess", command=cmd)
        env = {**os.environ, **self._env_vars}
        cwd = str(self._cwd) if self._cwd else None
        self._process = await anyio.open_process(cmd, env=env, cwd=cwd)
        if not self._process.stdin or not self._process.stdout:
            raise RuntimeError("Failed to create subprocess pipes")
        return self._process

    async def _initialize(self) -> None:
        """Initialize the ACP connection."""
        from acp.client.connection import ClientSideConnection
        from agentpool.agents.acp_agent.client_handler import ACPClientHandler

        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("Process not started")

        self._state = ACPSessionState(session_id="")
        self._client_handler = ACPClientHandler(self, self._state, self._input_provider)

        def client_factory(agent: ACPAgentProtocol) -> Client:
            return self._client_handler  # type: ignore[return-value]

        self._connection = ClientSideConnection(
            to_client=client_factory,
            input_stream=self._process.stdin,
            output_stream=self._process.stdout,
        )
        init_response = await self._connection.initialize(self._init_request)
        self._agent_info = init_response.agent_info
        self._caps = init_response.agent_capabilities
        self.log.info("ACP connection initialized", agent_info=self._agent_info)

    async def _create_session(self) -> None:
        """Create a new ACP session with configured MCP servers."""
        from acp.schema import NewSessionRequest
        from agentpool.agents.acp_agent.acp_converters import mcp_config_to_acp
        from agentpool.agents.acp_agent.helpers import filter_servers_by_capabilities

        if not self._connection:
            raise AgentNotInitializedError

        # Collect all MCP servers (extra + from mcp_servers list)
        all_servers = self._extra_mcp_servers[:]
        # Add servers from mcp_servers (converted to ACP format)
        if self.mcp.servers:
            all_servers.extend([mcp_config_to_acp(config) for config in self.mcp.servers])
        mcp_servers = filter_servers_by_capabilities(all_servers, self._caps, logger=self.log)
        cwd = self._cwd or str(Path.cwd())
        session_request = NewSessionRequest(cwd=cwd, mcp_servers=mcp_servers)
        response = await self._connection.new_session(session_request)
        self._sdk_session_id = response.session_id
        if self._state:
            self._state.session_id = self._sdk_session_id
            if response.config_options:
                self._state.config_options = list(response.config_options)
            if response.models:
                self._state.models = response.models
                self._state.current_model_id = response.models.current_model_id
            self._state.modes = response.modes
        model = self._state.current_model_id if self._state else None
        self.log.info("ACP session created", session_id=self._sdk_session_id, model=model)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._tool_bridge._mcp is not None:
            await self._tool_bridge.stop()
        self._extra_mcp_servers.clear()
        if self._client_handler:
            await self._client_handler.cleanup()
            self._client_handler = None
        if self._connection:
            try:
                await self._connection.close()
            except Exception:
                self.log.exception("Error closing ACP connection")
            self._connection = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception:
                self.log.exception("Error terminating ACP process")
            self._process = None

    async def _stream_events(  # noqa: PLR0915
        self,
        prompts: list[UserContent],
        *,
        user_msg: ChatMessage[Any],
        message_history: MessageHistory,
        effective_parent_id: str | None,
        message_id: str | None = None,
        session_id: str | None = None,
        parent_id: str | None = None,
        input_provider: InputProvider | None = None,
        deps: TDeps | None = None,
        wait_for_connections: bool | None = None,
        store_history: bool = True,
    ) -> AsyncIterator[RichAgentStreamEvent[str]]:
        from acp.schema import ForkSessionRequest, PromptRequest
        from agentpool.agents.acp_agent.acp_converters import (
            convert_to_acp_content,
            to_finish_reason,
        )

        # Update input provider if provided
        if input_provider is not None and self._client_handler:
            self._client_handler._input_provider = input_provider
        if not self._connection or not self._sdk_session_id or not self._state:
            raise AgentNotInitializedError

        run_id = str(uuid.uuid4())
        self._state.clear()
        model_messages: list[ModelResponse | ModelRequest] = []
        initial_request = ModelRequest(parts=[UserPromptPart(content=prompts)])
        model_messages.append(initial_request)
        current_response_parts: list[TextPart | ThinkingPart | ToolCallPart] = []
        text_chunks: list[str] = []
        file_tracker = FileTracker()
        assert self.session_id is not None
        yield RunStartedEvent(session_id=self.session_id, run_id=run_id, agent_name=self.name)
        final_blocks = convert_to_acp_content(prompts)
        # Handle ephemeral execution (fork session if store_history=False)
        session_id = self._sdk_session_id
        if not store_history and self._sdk_session_id:
            cwd = self._cwd or str(Path.cwd())
            fork_request = ForkSessionRequest(session_id=self._sdk_session_id, cwd=cwd)
            fork_response = await self._connection.fork_session(fork_request)
            session_id = fork_response.session_id
            self.log.debug("Forked session", parent=self._sdk_session_id, fork=session_id)
        prompt_request = PromptRequest(session_id=session_id, prompt=final_blocks)
        self.log.debug("Starting streaming prompt", num_blocks=len(final_blocks))
        prompt_task = asyncio.create_task(self._connection.prompt(prompt_request))
        self._prompt_task = prompt_task

        async def poll_acp_events() -> AsyncIterator[RichAgentStreamEvent[str]]:
            """Poll raw updates from ACP state, convert to events, until prompt completes."""
            from agentpool.agents.acp_agent.acp_converters import acp_to_native_event

            assert self._state
            while not prompt_task.done():
                if self._client_handler:
                    try:
                        await self._client_handler._update_event.wait_with_timeout(0.05)
                        self._client_handler._update_event.clear()
                    except TimeoutError:
                        pass
                while (update := self._state.pop_update()) is not None:
                    if native_event := acp_to_native_event(update):
                        yield native_event
            while (update := self._state.pop_update()) is not None:
                if native_event := acp_to_native_event(update):
                    yield native_event

        tool_metadata: dict[str, dict[str, Any]] = {}
        try:
            async with (
                self._tool_bridge.set_run_context(deps, input_provider, prompt=prompts),
                merge_queue_into_iterator(poll_acp_events(), self._event_queue) as merged_events,
            ):
                async for event in file_tracker(merged_events):
                    if isinstance(event, ToolResultMetadataEvent):
                        tool_metadata[event.tool_call_id] = event.metadata
                        continue
                    if self._cancelled:
                        self.log.info("Stream cancelled by user")
                        break
                    if isinstance(event, ToolCallCompleteEvent):
                        enriched_event = event
                        if not enriched_event.agent_name:
                            enriched_event = replace(enriched_event, agent_name=self.name)
                        if (
                            enriched_event.metadata is None
                            and enriched_event.tool_call_id in tool_metadata
                        ):
                            enriched_event = replace(
                                enriched_event, metadata=tool_metadata[enriched_event.tool_call_id]
                            )
                        event = enriched_event  # noqa: PLW2901
                    part = event_to_part(event)
                    if isinstance(part, TextPart):
                        text_chunks.append(part.content)
                    if part:
                        current_response_parts.append(part)
                    yield event
        except asyncio.CancelledError:
            self.log.info("Stream cancelled via task cancellation")
            self._cancelled = True

        if self._cancelled:
            message = ChatMessage[str](
                content="".join(text_chunks),
                role="assistant",
                name=self.name,
                message_id=message_id or str(uuid.uuid4()),
                session_id=self.session_id,
                parent_id=user_msg.message_id,
                model_name=self.model_name,
                messages=model_messages,
                metadata=file_tracker.get_metadata(),
                finish_reason="stop",
            )
            yield StreamCompleteEvent(message=message)
            self._prompt_task = None
            return

        response = await prompt_task
        finish_reason = to_finish_reason(response.stop_reason)
        if current_response_parts:
            model_messages.append(
                ModelResponse(
                    parts=current_response_parts,
                    finish_reason=finish_reason,
                    model_name=self.model_name,
                    provider_name=self._provider_type,
                )
            )

        text_content = "".join(text_chunks)
        usage, cost_info = await calculate_usage_from_parts(
            input_parts=prompts,
            response_parts=current_response_parts,
            text_content=text_content,
            model_name=self.model_name,
            provider=self._provider_type,
        )

        message = ChatMessage[str](
            content=text_content,
            role="assistant",
            name=self.name,
            message_id=message_id or str(uuid.uuid4()),
            session_id=self.session_id,
            parent_id=user_msg.message_id,
            model_name=self.model_name,
            messages=model_messages,
            metadata=file_tracker.get_metadata(),
            finish_reason=finish_reason,
            usage=usage,
            cost_info=cost_info,
        )
        yield StreamCompleteEvent(message=message)

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        return model_id if self._state and (model_id := self._state.current_model_id) else None

    async def set_model(self, model: str) -> None:
        """Update the model for the current session via ACP protocol."""
        await self._set_mode(model, "model")

    async def set_auto_approve(self, auto_approve: bool) -> None:
        """Set auto-approve mode for permission requests.

        Args:
            auto_approve: If True, automatically approve all permission requests.
                         If False, forward to callback/input_provider.
        """
        self.auto_approve = auto_approve
        if self._client_handler:
            self._client_handler.auto_approve = auto_approve
        self.log.info("Auto-approve mode changed", auto_approve=auto_approve)

    async def _interrupt(self) -> None:
        """Send CancelNotification to remote ACP server and cancel local tasks."""
        from acp.schema import CancelNotification

        if self._connection and self._sdk_session_id:
            try:
                cancel_notification = CancelNotification(session_id=self._sdk_session_id)
                await self._connection.cancel(cancel_notification)
                self.log.info("Sent cancel notification to ACP server")
            except Exception:
                self.log.exception("Failed to send cancel notification to ACP server")

        if self._prompt_task and not self._prompt_task.done():
            self._prompt_task.cancel()
            self.log.info("Cancelled prompt task")
        if self._current_stream_task and not self._current_stream_task.done():
            self._current_stream_task.cancel()

    async def get_available_models(self) -> list[ModelInfo] | None:
        """Get available models from the ACP session state."""
        from tokonomics.model_discovery.model_info import ModelInfo

        if not self._state or not self._state.models:
            return None
        return [
            ModelInfo(id=m.model_id, name=m.name, description=m.description)
            for m in self._state.models.available_models
        ]

    async def get_modes(self) -> list[ModeCategory]:
        """Get available modes from the ACP session state."""
        from agentpool.agents.acp_agent.acp_converters import get_modes

        if not self._state:
            return []

        return get_modes(
            self._state.config_options,
            available_modes=self._state.modes,
            available_models=self._state.models,
        )

    async def _set_mode(self, mode_id: str, category_id: str) -> None:
        """Forward mode change to remote ACP server."""
        from acp.schema import (
            SetSessionConfigOptionRequest,
            SetSessionModelRequest,
            SetSessionModeRequest,
        )
        from agentpool.agents.modes import ConfigOptionChanged

        if not self._connection or not self._sdk_session_id or not self._state:
            raise RuntimeError("Not connected to ACP server")
        available_modes = await self.get_modes()
        if matching_category := next((c for c in available_modes if c.id == category_id), None):
            valid_ids = {m.id for m in matching_category.available_modes}
            if mode_id not in valid_ids:
                raise UnknownModeError(mode_id, sorted(valid_ids))
        else:
            available_cats = {c.id for c in available_modes}
            raise UnknownCategoryError(category_id, sorted(available_cats))
        if self._state.config_options:
            assert category_id
            config_request = SetSessionConfigOptionRequest(
                session_id=self._sdk_session_id,
                config_id=category_id,
                value=mode_id,
            )
            response = await self._connection.set_session_config_option(config_request)
            if response.config_options:
                self._state.config_options = list(response.config_options)
        elif category_id == "mode":
            mode_request = SetSessionModeRequest(session_id=self._sdk_session_id, mode_id=mode_id)
            await self._connection.set_session_mode(mode_request)
            if self._state.modes:
                self._state.modes.current_mode_id = mode_id
        elif category_id == "model":
            request = SetSessionModelRequest(session_id=self._sdk_session_id, model_id=mode_id)
            if await self._connection.set_session_model(request):
                self._state.current_model_id = mode_id
                self.log.info("Model changed via legacy set_session_model")
            else:
                raise RuntimeError("Remote ACP agent does not support model changes.")
        else:
            raise UnknownCategoryError(category_id, ["mode", "model"])
        self.log.info("Config option changed", config_id=category_id, value=mode_id)
        change = ConfigOptionChanged(config_id=category_id, value_id=mode_id)
        await self.state_updated.emit(change)

    async def list_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[SessionData]:
        """List sessions from the remote ACP server."""
        from acp.schema import ListSessionsRequest
        from agentpool.sessions.models import SessionData

        if not self._connection:
            raise RuntimeError("Not connected to ACP server")
        request = ListSessionsRequest(cwd=cwd)
        try:
            response = await self._connection.list_sessions(request)
        except Exception:
            self.log.exception("Failed to list sessions from ACP server")
            return []
        else:
            result: list[SessionData] = []
            for acp_session in response.sessions:
                updated_at = get_updated_at(acp_session.updated_at)
                meta = acp_session.field_meta or {}
                created_at = updated_at
                if meta_created := meta.get("created_at"):
                    created_at = get_updated_at(meta_created)
                session_data = SessionData(
                    session_id=acp_session.session_id,
                    agent_name=self.name,
                    cwd=acp_session.cwd,
                    created_at=created_at,
                    last_active=updated_at,
                    pool_id=meta.get("pool_id"),
                    project_id=meta.get("project_id"),
                    parent_id=meta.get("parent_id"),
                    version=meta.get("version", "1"),
                    metadata={"title": acp_session.title} if acp_session.title else {},
                )
                result.append(session_data)
            self._sessions_cache = result
            if limit is not None:
                result = result[:limit]
            return result

    async def load_session(self, session_id: str) -> SessionData | None:
        """Load and restore a session from the remote ACP server."""
        from acp.schema import LoadSessionRequest
        from agentpool.agents.acp_agent.acp_converters import (
            acp_notifications_to_messages,
            mcp_config_to_acp,
        )
        from agentpool.agents.acp_agent.helpers import filter_servers_by_capabilities
        from agentpool.sessions.models import SessionData
        from agentpool.utils.time_utils import get_now

        if not self._connection:
            self.log.error("Cannot load session: not connected to ACP server")
            return None

        if not self._state:
            self.log.error("Cannot load session: state not initialized")
            return None
        all_servers = self._extra_mcp_servers[:]
        if self.mcp.servers:
            all_servers.extend([mcp_config_to_acp(config) for config in self.mcp.servers])
        mcp_servers = filter_servers_by_capabilities(all_servers, self._caps, logger=self.log)
        cwd = self._cwd or str(Path.cwd())
        load_request = LoadSessionRequest(
            session_id=session_id, cwd=cwd, mcp_servers=mcp_servers or None
        )
        try:
            self._state.start_load()
            response = await self._connection.load_session(load_request)
            raw_updates = self._state.finish_load()

            self._sdk_session_id = session_id
            self._state.session_id = session_id

            if response.config_options:
                self._state.config_options = list(response.config_options)
            if response.models:
                self._state.models = response.models
                self._state.current_model_id = response.models.current_model_id
            if response.modes:
                self._state.modes = response.modes

            if raw_updates:
                messages = acp_notifications_to_messages(
                    raw_updates,
                    session_id=session_id,
                    agent_name=self.name,
                    model_name=self.model_name,
                )
                self.conversation.chat_messages.clear()
                self.conversation.chat_messages.extend(messages)
                self.log.info("Restored session", session_id=session_id, msg_count=len(messages))
            else:
                self.log.debug("No conversation history to restore", session_id=session_id)

            self.log.info("Session loaded from ACP server", session_id=session_id)

            def find_in_cache(sid: str) -> SessionData | None:
                if self._sessions_cache is None:
                    return None
                return next((s for s in self._sessions_cache if s.session_id == sid), None)

            if session_info := find_in_cache(session_id):
                return session_info

            try:
                await self.list_sessions()
                if session_info := find_in_cache(session_id):
                    return session_info
            except Exception:  # noqa: BLE001
                self.log.debug("Could not fetch session metadata", session_id=session_id)
            now = get_now()
            return SessionData(
                session_id=session_id,
                agent_name=self.name,
                cwd=cwd,
                last_active=now,
                created_at=now,
            )

        except Exception:
            if self._state:
                self._state.finish_load()
            self.log.exception("Failed to load session from ACP server")
            return None


if __name__ == "__main__":

    async def main() -> None:
        async with ACPAgent(command="claude-code-acp") as agent:
            async for event in agent.run_stream("hello"):
                print(event)

    asyncio.run(main())
