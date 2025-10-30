"""Agent ACP Protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from acp.schema import CancelNotification
    from acp.schema.agent_responses import AgentResponse
    from acp.schema.client_requests import ClientRequest


class BaseAgent(Protocol):
    """Base agent interface for ACP with clean union signature."""

    async def handle_request(self, request: ClientRequest) -> AgentResponse:
        """Handle any client request and return appropriate response."""
        ...

    async def cancel(self, params: CancelNotification) -> None:
        """Handle cancellation request."""
        ...


class Agent(BaseAgent):
    """ACP Agent interface.

    Uses unified handle_request method with type-safe @overload signatures.
    All requests are routed through a single entry point for cleaner implementation.
    """
