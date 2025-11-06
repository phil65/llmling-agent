"""Base server class for LLMLing Agent servers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self


if TYPE_CHECKING:
    from types import TracebackType

    from llmling_agent import AgentPool


class BaseServer:
    """Base class for all LLMLing Agent servers."""

    def __init__(self, pool: AgentPool[Any], *args: Any, **kwargs: Any) -> None:
        """Initialize base server with agent pool."""
        self.pool = pool

    async def __aenter__(self) -> Self:
        """Enter async context and initialize the agent pool."""
        await self.pool.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context and cleanup the agent pool."""
        await self.pool.__aexit__(exc_type, exc_val, exc_tb)
