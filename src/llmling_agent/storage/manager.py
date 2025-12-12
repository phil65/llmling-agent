"""Storage manager for handling multiple providers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self

from anyenv import method_spawner
from pydantic_ai import Agent

from llmling_agent.log import get_logger
from llmling_agent.storage.serialization import serialize_messages
from llmling_agent.utils.tasks import TaskManager
from llmling_agent_config.storage import (
    FileStorageConfig,
    MemoryStorageConfig,
    SQLStorageConfig,
    TextLogConfig,
)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from types import TracebackType

    from llmling_agent.common_types import JsonValue
    from llmling_agent.messaging import ChatMessage
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.storage import BaseStorageProviderConfig, StorageConfig
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

    def __init__(self, config: StorageConfig) -> None:
        """Initialize storage manager.

        Args:
            config: Storage configuration including providers and filters
        """
        self.config = config
        self.task_manager = TaskManager()
        self.providers = [self._create_provider(cfg) for cfg in self.config.effective_providers]

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
    ) -> None:
        """Clean up all providers."""
        errors = []
        for provider in self.providers:
            try:
                await provider.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                errors.append(e)
                logger.exception("Error cleaning up provider", provider=provider)

        await self.task_manager.cleanup_tasks()

        if errors:
            msg = "Provider cleanup errors"
            raise ExceptionGroup(msg, errors)

    def cleanup(self) -> None:
        """Clean up all providers."""
        for provider in self.providers:
            try:
                provider.cleanup()
            except Exception:
                logger.exception("Error cleaning up provider", provider=provider)
        self.providers.clear()

    def _create_provider(self, config: BaseStorageProviderConfig) -> StorageProvider:
        """Create provider instance from configuration."""
        # Extract common settings from BaseStorageProviderConfig
        match self.config.filter_mode:
            case "and" if self.config.agents and config.agents:
                logged_agents: set[str] | None = self.config.agents & config.agents
            case "and":
                # If either is None, use the other; if both None, use None (log all)
                if self.config.agents is None and config.agents is None:
                    logged_agents = None
                else:
                    logged_agents = self.config.agents or config.agents or set()
            case "override":
                logged_agents = config.agents if config.agents is not None else self.config.agents

        provider_config = config.model_copy(
            update={
                "log_messages": config.log_messages and self.config.log_messages,
                "log_conversations": config.log_conversations and self.config.log_conversations,
                "log_commands": config.log_commands and self.config.log_commands,
                "log_context": config.log_context and self.config.log_context,
                "agents": logged_agents,
            }
        )

        match provider_config:
            case SQLStorageConfig() as config:
                from llmling_agent_storage.sql_provider import SQLModelProvider

                return SQLModelProvider(provider_config)
            case FileStorageConfig():
                from llmling_agent_storage.file_provider import FileProvider

                return FileProvider(provider_config)
            case TextLogConfig():
                from llmling_agent_storage.text_log_provider import TextLogProvider

                return TextLogProvider(provider_config)

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
            msg = "Default provider not found or not capable of loading history"
            logger.warning(msg, provider=self.config.default_provider)

        # Find first capable provider
        for provider in self.providers:
            if not getattr(provider, "write_only", False) and provider.can_load_history:
                return provider

        msg = "No capable provider found for loading history"
        raise RuntimeError(msg)

    @method_spawner
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

    @method_spawner
    async def log_message(self, message: ChatMessage[Any]) -> None:
        """Log message to all providers."""
        if not self.config.log_messages:
            return

        for provider in self.providers:
            if provider.should_log_agent(message.name or "no name"):
                await provider.log_message(
                    conversation_id=message.conversation_id or "",
                    message_id=message.message_id,
                    content=str(message.content),
                    role=message.role,
                    name=message.name,
                    cost_info=message.cost_info,
                    model=message.model_name,
                    response_time=message.response_time,
                    forwarded_from=message.forwarded_from,
                    provider_name=message.provider_name,
                    provider_response_id=message.provider_response_id,
                    messages=serialize_messages(message.messages),
                    finish_reason=message.finish_reason,
                )

    @method_spawner
    async def log_conversation(
        self,
        *,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Log conversation to all providers."""
        if not self.config.log_conversations:
            return

        for provider in self.providers:
            await provider.log_conversation(
                conversation_id=conversation_id,
                node_name=node_name,
                start_time=start_time,
            )

    @method_spawner
    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        """Log command to all providers."""
        if not self.config.log_commands:
            return

        for provider in self.providers:
            await provider.log_command(
                agent_name=agent_name,
                session_id=session_id,
                command=command,
                context_type=context_type,
                metadata=metadata,
            )

    @method_spawner
    async def log_context_message(
        self,
        *,
        conversation_id: str,
        content: str,
        role: str,
        name: str | None = None,
        model: str | None = None,
    ) -> None:
        """Log context message to all providers."""
        for provider in self.providers:
            await provider.log_context_message(
                conversation_id=conversation_id,
                content=content,
                role=role,
                name=name,
                model=model,
            )

    @method_spawner
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
                cls_name = provider.__class__.__name__
                logger.exception("Error resetting provider", provider=cls_name)
                return (0, 0)

        results = await asyncio.gather(*(reset_provider(provider) for provider in self.providers))
        # Return the counts from the last provider (maintaining existing behavior)
        return results[-1] if results else (0, 0)

    @method_spawner
    async def get_conversation_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get counts from primary provider."""
        provider = self.get_history_provider()
        return await provider.get_conversation_counts(agent_name=agent_name)

    @method_spawner
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

    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str,
    ) -> None:
        """Update conversation title in all providers.

        Args:
            conversation_id: ID of the conversation to update
            title: New title for the conversation
        """
        for provider in self.providers:
            await provider.update_conversation_title(conversation_id, title)

    async def get_conversation_title(
        self,
        conversation_id: str,
    ) -> str | None:
        """Get the title of a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            The conversation title, or None if not set.
        """
        provider = self.get_history_provider()
        return await provider.get_conversation_title(conversation_id)

    async def generate_conversation_title(
        self,
        conversation_id: str,
        messages: Sequence[ChatMessage[Any]],
    ) -> str | None:
        """Generate and store a title for a conversation.

        Uses the configured title generation model to create a short,
        descriptive title based on the conversation content.

        Args:
            conversation_id: ID of the conversation to title
            messages: Messages to use for title generation

        Returns:
            The generated title, or None if title generation is disabled.
        """
        if not self.config.title_generation_model:
            return None

        # Check if title already exists
        existing = await self.get_conversation_title(conversation_id)
        if existing:
            return existing

        # Format messages for the prompt
        formatted = "\n".join(
            f"{msg.role}: {msg.content[:500]}"
            for msg in messages[:4]  # Limit context
        )

        try:
            agent: Agent[None, str] = Agent(
                model=self.config.title_generation_model,
                instructions=self.config.title_generation_prompt,
            )
            result = await agent.run(formatted)
            title = result.output.strip().strip("\"'")  # Remove quotes if present

            # Store the title
            await self.update_conversation_title(conversation_id, title)
            logger.debug(
                "Generated conversation title",
                conversation_id=conversation_id,
                title=title,
            )
        except Exception:
            logger.exception(
                "Failed to generate conversation title",
                conversation_id=conversation_id,
            )
            return None
        else:
            return title
