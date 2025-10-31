from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from acp.schema import (
        AgentRequest,
        ClientResponse,
        SessionNotification,
    )


class BaseClient(Protocol):
    """Base client interface for ACP with clean union signature."""

    async def handle_request(self, request: AgentRequest) -> ClientResponse:
        """Handle any agent request and return appropriate response."""
        ...

    async def handle_notification(self, notification: SessionNotification) -> None:
        """Handle session update notification."""
        ...


class Client(BaseClient):
    """ACP Client interface.

    Uses unified handle_request method with type-safe signatures.
    All requests are routed through a single entry point for cleaner implementation.
    """
