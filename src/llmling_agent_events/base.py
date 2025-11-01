"""Event sources for LLMling agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from llmling_agent.log import get_logger


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from llmling_agent.messaging.events import EventData
    from llmling_agent_config.events import EventConfig


class EventSource(ABC):
    """Base class for event sources."""

    @abstractmethod
    async def connect(self):
        """Initialize connection to event source."""

    @abstractmethod
    async def disconnect(self):
        """Close connection to event source."""

    @abstractmethod
    def events(self) -> AsyncGenerator[EventData]:
        """Get event iterator.

        Returns:
            AsyncIterator yielding events from this source

        Note: This is a coroutine that returns an AsyncIterator
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: EventConfig) -> EventSource:
        """Create event source from configuration.

        Args:
            config: Event source configuration

        Returns:
            Configured event source instance

        Raises:
            ValueError: If source type is unknown or disabled
        """
        from llmling_agent_config.events import (
            EmailConfig,
            FileWatchConfig,
            TimeEventConfig,
            WebhookConfig,
        )

        if not config.enabled:
            msg = f"Source {config.name} is disabled"
            raise ValueError(msg)
        logger.info("Creating event source", name=config.name, provider=config.type)
        match config:
            case FileWatchConfig():
                from llmling_agent_events.file_watcher import FileSystemEventSource

                return FileSystemEventSource(config)
            case WebhookConfig():
                from llmling_agent_events.webhook_watcher import WebhookEventSource

                return WebhookEventSource(config)
            case EmailConfig():
                from llmling_agent_events.email_watcher import EmailEventSource

                return EmailEventSource(config)
            case TimeEventConfig():
                from llmling_agent_events.timed_watcher import TimeEventSource

                return TimeEventSource(config)
            case _:
                msg = f"Unknown event source type: {config.type}"
                raise ValueError(msg)
