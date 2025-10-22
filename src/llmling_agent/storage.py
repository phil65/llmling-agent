"""Storage manager for handling multiple providers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Self

from llmling_agent.log import get_logger
from llmling_agent.utils.tasks import TaskManager
from llmling_agent_config.storage import (
    FileStorageConfig,
    Mem0Config,
    MemoryStorageConfig,
    SQLStorageConfig,
    TextLogConfig,
)


if TYPE_CHECKING:
    from datetime import datetime
    from types import TracebackType

    from llmling_agent.common_types import JsonValue
    from llmling_agent.messaging.messages import ChatMessage, TokenCost
    from llmling_agent.tools import ToolCallInfo
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.storage import (
        BaseStorageProviderConfig,
        StorageConfig,
    )
    from llmling_agent_storage.base import StorageProvider

logger = get_logger(__name__)


class StorageManager:
    """Manages multiple storage providers.

    Handles:
    - Provider initialization and cleanup
    - Message distribution to providers
    - History loading from capable providers
    - Global logging filters
    """

    def __init__(self, config: StorageConfig):
        """Initialize storage manager.

        Args:
            config: Storage configuration including providers and filters
        """
        self.config = config
        self.task_manager = TaskManager()
        self.providers = [
            self._create_provider(cfg) for cfg in self.config.effective_providers
        ]

    async def __aenter__(self) -> Self:
        """Initialize all providers."""
        for provider in self.providers:
            await provider.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Clean up all providers."""
        errors = []
        for provider in self.providers:
            try:
                await provider.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                errors.append(e)
                logger.exception("Error cleaning up provider: %r", provider)

        await self.task_manager.cleanup_tasks()

        if errors:
            msg = "Provider cleanup errors"
            raise ExceptionGroup(msg, errors)

    def cleanup(self):
        """Clean up all providers."""
        for provider in self.providers:
            try:
                provider.cleanup()
            except Exception:
                logger.exception("Error cleaning up provider: %r", provider)
        self.providers.clear()

    def _create_provider(self, config: BaseStorageProviderConfig) -> StorageProvider:
        """Create provider instance from configuration."""
        # Extract common settings from BaseStorageProviderConfig
        match self.config.filter_mode:
            case "and" if self.config.agents and config.agents:
                logged_agents = self.config.agents & config.agents
            case "and":
                logged_agents = self.config.agents or config.agents or set()
            case "override":
                logged_agents = (
                    config.agents
                    if config.agents is not None
                    else self.config.agents or set()
                )

        provider_config = config.model_copy(
            update={
                "log_messages": config.log_messages and self.config.log_messages,
                "log_conversations": config.log_conversations
                and self.config.log_conversations,
                "log_tool_calls": config.log_tool_calls and self.config.log_tool_calls,
                "log_commands": config.log_commands and self.config.log_commands,
                "log_context": config.log_context and self.config.log_context,
                "agents": logged_agents,
            }
        )

        match provider_config:
            case SQLStorageConfig(url=url, pool_size=pool_size):
                from sqlmodel import create_engine

                from llmling_agent_storage.sql_provider import SQLModelProvider

                engine = create_engine(str(url), pool_size=pool_size)
                return SQLModelProvider(provider_config, engine)
            case FileStorageConfig():
                from llmling_agent_storage.file_provider import FileProvider

                return FileProvider(provider_config)
            case TextLogConfig():
                from llmling_agent_storage.text_log_provider import TextLogProvider

                return TextLogProvider(provider_config)

            case Mem0Config():
                from llmling_agent_storage.mem0 import Mem0StorageProvider

                return Mem0StorageProvider(provider_config)

            case MemoryStorageConfig():
                from llmling_agent_storage.memory_provider import MemoryStorageProvider

                return MemoryStorageProvider(provider_config)
            case _:
                msg = f"Unknown provider type: {provider_config}"
                raise ValueError(msg)

    def get_history_provider(self, preferred: str | None = None) -> StorageProvider:
        """Get provider for loading history.

        Args:
            preferred: Optional preferred provider name

        Returns:
            First capable provider based on priority:
            1. Preferred provider if specified and capable
            2. Default provider if specified and capable
            3. First capable provider
            4. Raises error if no capable provider found
        """

        # Function to find capable provider by name
        def find_provider(name: str) -> StorageProvider | None:
            for p in self.providers:
                if (
                    not getattr(p, "write_only", False)
                    and p.can_load_history
                    and p.__class__.__name__.lower() == name.lower()
                ):
                    return p
            return None

        # Try preferred provider
        if preferred and (provider := find_provider(preferred)):
            return provider

        # Try default provider
        if self.config.default_provider:
            if provider := find_provider(self.config.default_provider):
                return provider
            msg = "Default provider %s not found or not capable of loading history"
            logger.warning(msg, self.config.default_provider)

        # Find first capable provider
        for provider in self.providers:
            if not getattr(provider, "write_only", False) and provider.can_load_history:
                return provider

        msg = "No capable provider found for loading history"
        raise RuntimeError(msg)

    async def filter_messages(
        self,
        query: SessionQuery,
        preferred_provider: str | None = None,
    ) -> list[ChatMessage[str]]:
        """Get messages matching query.

        Args:
            query: Filter criteria
            preferred_provider: Optional preferred provider to use
        """
        provider = self.get_history_provider(preferred_provider)
        return await provider.filter_messages(query)

    async def log_message(
        self,
        *,
        conversation_id: str,
        message_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: TokenCost | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
    ):
        """Log message to all providers."""
        if not self.config.log_messages:
            return

        for provider in self.providers:
            if provider.should_log_agent(name or "no name"):
                self.task_manager.create_task(
                    provider.log_message(
                        conversation_id=conversation_id,
                        message_id=message_id,
                        content=content,
                        role=role,
                        name=name,
                        cost_info=cost_info,
                        model=model,
                        response_time=response_time,
                        forwarded_from=forwarded_from,
                    )
                )

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ):
        """Log conversation to all providers."""
        if not self.config.log_conversations:
            return

        for provider in self.providers:
            self.task_manager.create_task(
                provider.log_conversation(
                    conversation_id=conversation_id,
                    node_name=node_name,
                    start_time=start_time,
                )
            )

    async def log_tool_call(
        self,
        *,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ):
        """Log tool call to all providers."""
        if not self.config.log_tool_calls:
            return

        for provider in self.providers:
            self.task_manager.create_task(
                provider.log_tool_call(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    tool_call=tool_call,
                )
            )

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ):
        """Log command to all providers."""
        if not self.config.log_commands:
            return

        for provider in self.providers:
            self.task_manager.create_task(
                provider.log_command(
                    agent_name=agent_name,
                    session_id=session_id,
                    command=command,
                    context_type=context_type,
                    metadata=metadata,
                )
            )

    async def log_context_message(
        self,
        *,
        conversation_id: str,
        content: str,
        role: str,
        name: str | None = None,
        model: str | None = None,
    ):
        """Log context message to all providers."""
        for provider in self.providers:
            self.task_manager.create_task(
                provider.log_context_message(
                    conversation_id=conversation_id,
                    content=content,
                    role=role,
                    name=name,
                    model=model,
                )
            )

    async def reset(
        self,
        *,
        agent_name: str | None = None,
        hard: bool = False,
    ) -> tuple[int, int]:
        """Reset storage in all providers concurrently."""

        async def reset_provider(provider: StorageProvider) -> tuple[int, int]:
            try:
                return await provider.reset(agent_name=agent_name, hard=hard)
            except Exception:
                msg = "Error resetting provider: %r"
                logger.exception(msg, provider.__class__.__name__)
                return (0, 0)

        results = await asyncio.gather(
            *(reset_provider(provider) for provider in self.providers)
        )
        # Return the counts from the last provider (maintaining existing behavior)
        return results[-1] if results else (0, 0)

    async def get_conversation_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get counts from primary provider."""
        provider = self.get_history_provider()
        return await provider.get_conversation_counts(agent_name=agent_name)

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
        preferred_provider: str | None = None,
    ) -> list[str]:
        """Get command history."""
        if not self.config.log_commands:
            return []

        provider = self.get_history_provider(preferred_provider)
        return await provider.get_commands(
            agent_name=agent_name,
            session_id=session_id,
            limit=limit,
            current_session_only=current_session_only,
        )

    # Sync wrappers
    def reset_sync(self, *args, **kwargs) -> tuple[int, int]:
        """Sync wrapper for reset."""
        return self.task_manager.run_task_sync(self.reset(*args, **kwargs))

    def get_conversation_counts_sync(self, *args, **kwargs) -> tuple[int, int]:
        """Sync wrapper for get_conversation_counts."""
        return self.task_manager.run_task_sync(
            self.get_conversation_counts(*args, **kwargs)
        )

    def log_conversation_sync(self, *args, **kwargs):
        """Sync wrapper for log_conversation."""
        for provider in self.providers:
            provider.log_conversation_sync(*args, **kwargs)

    def log_message_sync(self, *args, **kwargs):
        """Sync wrapper for log_message."""
        for provider in self.providers:
            provider.log_message_sync(*args, **kwargs)

    def log_tool_call_sync(self, *args, **kwargs):
        """Sync wrapper for log_tool_call."""
        for provider in self.providers:
            provider.log_tool_call_sync(*args, **kwargs)

    def log_command_sync(self, *args, **kwargs):
        """Sync wrapper for log_command."""
        for provider in self.providers:
            provider.log_command_sync(*args, **kwargs)

    def get_commands_sync(self, *args, **kwargs) -> list[str]:
        """Sync wrapper for get_commands."""
        provider = self.get_history_provider()
        return provider.get_commands_sync(*args, **kwargs)

    def filter_messages_sync(self, *args, **kwargs) -> list[ChatMessage[str]]:
        """Sync wrapper for filter_messages."""
        provider = self.get_history_provider()
        return provider.filter_messages_sync(*args, **kwargs)

    def log_context_message_sync(self, *args, **kwargs):
        """Sync wrapper for log_context_message."""
        for provider in self.providers:
            provider.log_context_message_sync(*args, **kwargs)
