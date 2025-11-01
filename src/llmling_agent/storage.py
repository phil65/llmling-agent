"""Storage manager for handling multiple providers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Self

from anyenv import method_spawner
from pydantic import ConfigDict, TypeAdapter

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
    from collections.abc import Sequence
    from datetime import datetime
    from types import TracebackType

    from pydantic_ai import ModelRequestPart, ModelResponsePart

    from llmling_agent.common_types import JsonValue
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.storage import (
        BaseStorageProviderConfig,
        StorageConfig,
    )
    from llmling_agent_storage.base import StorageProvider

logger = get_logger(__name__)

# Type adapter for serializing ModelResponsePart sequences
parts_adapter: TypeAdapter = TypeAdapter(
    list,
    config=ConfigDict(ser_json_bytes="base64", val_json_bytes="base64"),
)


def deserialize_parts(parts_json: str | None) -> Sequence[ModelResponsePart]:
    """Deserialize pydantic-ai message parts from JSON string.

    Args:
        parts_json: JSON string representation of parts or None if empty

    Returns:
        Sequence of ModelResponsePart objects, empty if deserialization fails
    """
    if not parts_json:
        return []

    try:
        # Deserialize using pydantic's JSON deserialization
        return parts_adapter.validate_json(parts_json.encode())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to deserialize message parts: %s", e)
        return []  # Return empty list on failure


def serialize_parts(parts: Sequence[ModelResponsePart | ModelRequestPart]) -> str | None:
    """Serialize pydantic-ai message parts from ChatMessage.

    Args:
        parts: Sequence of ModelResponsePart from ChatMessage.parts

    Returns:
        JSON string representation of parts or None if empty
    """
    if not parts:
        return None

    try:
        # Convert parts to serializable format
        serializable_parts = []
        for part in parts:
            # Handle RetryPromptPart context serialization issues
            from pydantic_ai.messages import RetryPromptPart

            if isinstance(part, RetryPromptPart) and isinstance(part.content, list):
                for content in part.content:
                    if isinstance(content, dict) and "ctx" in content:
                        content["ctx"] = {k: str(v) for k, v in content["ctx"].items()}
            serializable_parts.append(part)

        # Serialize using pydantic's JSON serialization
        return parts_adapter.dump_json(serializable_parts).decode()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to serialize message parts: %s", e)
        return str(parts)  # Fallback to string representation


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
                logged_agents: set[str] | None = self.config.agents & config.agents
            case "and":
                # If either is None, use the other; if both None, use None (log all)
                if self.config.agents is None and config.agents is None:
                    logged_agents = None
                else:
                    logged_agents = self.config.agents or config.agents or set()
            case "override":
                logged_agents = (
                    config.agents if config.agents is not None else self.config.agents
                )

        provider_config = config.model_copy(
            update={
                "log_messages": config.log_messages and self.config.log_messages,
                "log_conversations": config.log_conversations
                and self.config.log_conversations,
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
    async def log_message(self, message: ChatMessage):
        """Log message to all providers."""
        if not self.config.log_messages:
            return

        for provider in self.providers:
            if provider.should_log_agent(message.name or "no name"):
                self.task_manager.create_task(
                    provider.log_message(
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
                        parts=serialize_parts(message.parts),
                        finish_reason=message.finish_reason,
                    )
                )

    @method_spawner
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

    @method_spawner
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

    @method_spawner
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
                msg = "Error resetting provider: %r"
                logger.exception(msg, provider.__class__.__name__)
                return (0, 0)

        results = await asyncio.gather(
            *(reset_provider(provider) for provider in self.providers)
        )
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
