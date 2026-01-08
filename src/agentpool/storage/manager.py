"""Storage manager for handling multiple providers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self

from anyenv import method_spawner
from pydantic_ai import Agent

from agentpool.log import get_logger
from agentpool.messaging import ChatMessage
from agentpool.storage.serialization import serialize_messages
from agentpool.utils.tasks import TaskManager
from agentpool_config.session import SessionQuery
from agentpool_config.storage import (
    ClaudeStorageConfig,
    FileStorageConfig,
    MemoryStorageConfig,
    OpenCodeStorageConfig,
    SQLStorageConfig,
    TextLogConfig,
    ZedStorageConfig,
)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from types import TracebackType

    from agentpool.common_types import JsonValue
    from agentpool.sessions.models import ProjectData
    from agentpool_config.storage import BaseStorageProviderConfig, StorageConfig
    from agentpool_storage.base import StorageProvider

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

    def _create_provider(self, config: BaseStorageProviderConfig) -> StorageProvider:  # noqa: PLR0911
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
                from agentpool_storage.sql_provider import SQLModelProvider

                return SQLModelProvider(provider_config)
            case FileStorageConfig():
                from agentpool_storage.file_provider import FileProvider

                return FileProvider(provider_config)
            case TextLogConfig():
                from agentpool_storage.text_log_provider import TextLogProvider

                return TextLogProvider(provider_config)

            case MemoryStorageConfig():
                from agentpool_storage.memory_provider import MemoryStorageProvider

                return MemoryStorageProvider(provider_config)
            case ClaudeStorageConfig():
                from agentpool_storage.claude_provider import ClaudeStorageProvider

                return ClaudeStorageProvider(provider_config)
            case OpenCodeStorageConfig():
                from agentpool_storage.opencode_provider import OpenCodeStorageProvider

                return OpenCodeStorageProvider(provider_config)
            case ZedStorageConfig():
                from agentpool_storage.zed_provider import ZedStorageProvider

                return ZedStorageProvider(provider_config)
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
                if p.can_load_history and p.__class__.__name__.lower() == name.lower():
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
            if provider.can_load_history:
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
                    parent_id=message.parent_id,
                    cost_info=message.cost_info,
                    model=message.model_name,
                    response_time=message.response_time,
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
        initial_prompt: str | None = None,
    ) -> None:
        """Log conversation to all providers.

        Args:
            conversation_id: Unique conversation identifier
            node_name: Name of the node/agent
            start_time: Optional start time
            initial_prompt: Optional initial prompt to trigger title generation
        """
        if not self.config.log_conversations:
            return

        for provider in self.providers:
            await provider.log_conversation(
                conversation_id=conversation_id,
                node_name=node_name,
                start_time=start_time,
            )

        # Trigger title generation if prompt provided and model configured
        if initial_prompt and self.config.title_generation_model:
            self.task_manager.create_task(
                self._generate_title_from_prompt(conversation_id, initial_prompt),
                name=f"title_gen_{conversation_id[:8]}",
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

    async def get_conversation_titles(
        self,
        conversation_ids: list[str],
    ) -> dict[str, str | None]:
        """Get titles for multiple conversations.

        Args:
            conversation_ids: List of conversation IDs

        Returns:
            Dict mapping conversation_id to title (or None if not set)
        """
        if not conversation_ids:
            return {}

        provider = self.get_history_provider()
        titles: dict[str, str | None] = {}
        for conv_id in conversation_ids:
            try:
                titles[conv_id] = await provider.get_conversation_title(conv_id)
            except Exception:  # noqa: BLE001
                titles[conv_id] = None
        return titles

    async def get_message_counts(
        self,
        conversation_ids: list[str],
    ) -> dict[str, int]:
        """Get message counts for multiple conversations.

        Args:
            conversation_ids: List of conversation IDs

        Returns:
            Dict mapping conversation_id to message count
        """
        if not conversation_ids:
            return {}

        counts: dict[str, int] = {}
        for conv_id in conversation_ids:
            try:
                query = SessionQuery(name=conv_id)
                messages = await self.filter_messages(query)
                counts[conv_id] = len(messages) if messages else 0
            except Exception:  # noqa: BLE001
                counts[conv_id] = 0
        return counts

    @method_spawner
    async def delete_conversation_messages(
        self,
        conversation_id: str,
    ) -> int:
        """Delete all messages for a conversation in all providers.

        Used for compaction - removes existing messages so they can be
        replaced with compacted versions.

        Args:
            conversation_id: ID of the conversation to clear

        Returns:
            Total number of messages deleted across all providers
        """
        total_deleted = 0
        for provider in self.providers:
            try:
                deleted = await provider.delete_conversation_messages(conversation_id)
                total_deleted += deleted
            except NotImplementedError:
                # Provider doesn't support deletion (e.g., write-only log)
                pass
            except Exception:
                logger.exception(
                    "Error deleting messages from provider",
                    provider=provider.__class__.__name__,
                    conversation_id=conversation_id,
                )
        return total_deleted

    @method_spawner
    async def replace_conversation_messages(
        self,
        conversation_id: str,
        messages: Sequence[ChatMessage[Any]],
    ) -> tuple[int, int]:
        """Replace all messages for a conversation with new ones.

        Deletes existing messages and logs new ones. Used for compaction
        where the full history is replaced with a compacted version.

        Args:
            conversation_id: ID of the conversation
            messages: New messages to store

        Returns:
            Tuple of (deleted_count, added_count)
        """
        # First delete existing messages
        deleted = await self.delete_conversation_messages(conversation_id)

        # Then log new messages
        added = 0
        for message in messages:
            # Ensure conversation_id is set on the message
            msg_to_log: ChatMessage[Any] = message
            if not message.conversation_id:
                msg_to_log = ChatMessage(
                    content=message.content,
                    role=message.role,
                    name=message.name,
                    conversation_id=conversation_id,
                    message_id=message.message_id,
                    parent_id=message.parent_id,
                    model_name=message.model_name,
                    cost_info=message.cost_info,
                    response_time=message.response_time,
                    timestamp=message.timestamp,
                    provider_name=message.provider_name,
                    provider_response_id=message.provider_response_id,
                    messages=message.messages,
                    finish_reason=message.finish_reason,
                )
            await self.log_message(msg_to_log)
            added += 1

        return deleted, added

    async def _generate_title_from_prompt(
        self,
        conversation_id: str,
        prompt: str,
    ) -> str | None:
        """Generate title from initial prompt (internal, fire-and-forget).

        Called automatically by log_conversation when initial_prompt is provided.

        Args:
            conversation_id: ID of the conversation to title
            prompt: The initial user prompt

        Returns:
            The generated title, or None if generation fails/disabled.
        """
        if not self.config.title_generation_model:
            return None

        # Check if title already exists
        existing = await self.get_conversation_title(conversation_id)
        if existing:
            return existing

        try:
            from llmling_models.models.helpers import infer_model

            model = infer_model(self.config.title_generation_model)
            agent: Agent[None, str] = Agent(
                model=model,
                instructions=self.config.title_generation_prompt,
            )
            # Just use the prompt directly - simpler than formatting messages
            result = await agent.run(f"user: {prompt[:500]}")
            title = result.output.strip().strip("\"'")  # Remove quotes if present
            await self.update_conversation_title(conversation_id, title)
            logger.debug("Generated session title", conversation_id=conversation_id, title=title)
            return title
        except Exception:
            logger.exception("Failed to generate session title", conversation_id=conversation_id)
            return None

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
        formatted = "\n".join(f"{i.role}: {i.content[:500]}" for i in messages[:4])  # Limit context
        try:
            from llmling_models.models.helpers import infer_model

            model = infer_model(self.config.title_generation_model)
            agent: Agent[None, str] = Agent(
                model=model,
                instructions=self.config.title_generation_prompt,
            )
            result = await agent.run(formatted)
            title = result.output.strip().strip("\"'")  # Remove quotes if present
            # Store the title
            await self.update_conversation_title(conversation_id, title)
            logger.debug("Generated session title", conversation_id=conversation_id, title=title)
        except Exception:
            logger.exception("Failed to generate session title", conversation_id=conversation_id)
            return None
        else:
            return title

    # Project methods

    def get_project_provider(self) -> StorageProvider:
        """Get provider capable of storing projects.

        Returns:
            First provider that supports project storage.

        Raises:
            RuntimeError: If no capable provider found.
        """
        if self.providers:
            return self.providers[0]
        msg = "No provider found that supports project storage"
        raise RuntimeError(msg)

    @method_spawner
    async def save_project(self, project: ProjectData) -> None:
        """Save or update a project in all capable providers.

        Args:
            project: Project data to persist
        """
        for provider in self.providers:
            try:
                await provider.save_project(project)
            except NotImplementedError:
                pass
            except Exception:
                logger.exception(
                    "Error saving project",
                    provider=provider.__class__.__name__,
                    project_id=project.project_id,
                )

    @method_spawner
    async def get_project(self, project_id: str) -> ProjectData | None:
        """Get a project by ID.

        Args:
            project_id: Project identifier

        Returns:
            Project data if found, None otherwise
        """
        provider = self.get_project_provider()
        return await provider.get_project(project_id)

    @method_spawner
    async def get_project_by_worktree(self, worktree: str) -> ProjectData | None:
        """Get a project by worktree path.

        Args:
            worktree: Absolute path to the project worktree

        Returns:
            Project data if found, None otherwise
        """
        provider = self.get_project_provider()
        return await provider.get_project_by_worktree(worktree)

    @method_spawner
    async def get_project_by_name(self, name: str) -> ProjectData | None:
        """Get a project by friendly name.

        Args:
            name: Project name

        Returns:
            Project data if found, None otherwise
        """
        provider = self.get_project_provider()
        return await provider.get_project_by_name(name)

    @method_spawner
    async def list_projects(self, limit: int | None = None) -> list[ProjectData]:
        """List all projects, ordered by last_active descending.

        Args:
            limit: Maximum number of projects to return

        Returns:
            List of project data objects
        """
        provider = self.get_project_provider()
        return await provider.list_projects(limit=limit)

    @method_spawner
    async def delete_project(self, project_id: str) -> bool:
        """Delete a project from all providers.

        Args:
            project_id: Project identifier

        Returns:
            True if project was deleted from at least one provider
        """
        deleted = False
        for provider in self.providers:
            try:
                if await provider.delete_project(project_id):
                    deleted = True
            except NotImplementedError:
                pass
            except Exception:
                logger.exception(
                    "Error deleting project",
                    provider=provider.__class__.__name__,
                    project_id=project_id,
                )
        return deleted

    @method_spawner
    async def touch_project(self, project_id: str) -> None:
        """Update project's last_active timestamp in all providers.

        Args:
            project_id: Project identifier
        """
        for provider in self.providers:
            try:
                await provider.touch_project(project_id)
            except NotImplementedError:
                pass
            except Exception:
                logger.exception(
                    "Error touching project",
                    provider=provider.__class__.__name__,
                    project_id=project_id,
                )
