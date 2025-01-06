from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from slashed import BaseCommand, CommandStore, DefaultOutputWriter, ExitCommandError
from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.result import StreamedRunResult

    from llmling_agent.agent import Agent
    from llmling_agent.delegation.pool import AgentPool
    from llmling_agent.models.context import AgentContext


logger = get_logger(__name__)

TContext = TypeVar("TContext")
TDeps = TypeVar("TDeps")
TResult = TypeVar("TResult", default=str)


class SlashedAgent[TDeps, TContext]:
    """Wraps an agent with slash command support."""

    def __init__(
        self,
        agent: Agent[TDeps],
        *,
        command_context: TContext | None = None,
        command_history_path: str | None = None,
        output: DefaultOutputWriter | None = None,
    ):
        self.agent = agent
        assert self.agent._context, "Agent must have a context!"
        assert self.agent._context.pool, "Agent must have a pool!"

        self.commands = CommandStore(
            history_file=command_history_path,
            enable_system_commands=True,
        )
        self.command_context: TContext = command_context or self  # type: ignore
        self.output = output or DefaultOutputWriter()

    @property
    def pool(self) -> AgentPool:
        """Get agent's pool from context."""
        assert self.agent._context.pool
        return self.agent._context.pool

    def register_command(self, command: BaseCommand) -> None:
        """Register additional command."""
        self.commands.register_command(command)

    async def handle_command(
        self,
        command: str,
        output: DefaultOutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str]:
        """Handle a slash command."""
        try:
            await self.commands.execute_command_with_context(
                command,
                context=self.command_context,
                output_writer=output or self.output,
                metadata=metadata,
            )
            return ChatMessage(content="", role="system")
        except ExitCommandError:
            raise
        except Exception as e:  # noqa: BLE001
            msg = f"Command error: {e}"
            return ChatMessage(content=msg, role="system")

    async def run[TResult](
        self,
        content: str,
        *,
        result_type: type[TResult] | None = None,
        output: DefaultOutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatMessage[TResult]:
        """Run agent or handle command based on prefix."""
        if content.startswith("/"):
            return await self.handle_command(  # type: ignore
                content[1:],
                output=output,
                metadata=metadata,
            )

        return await self.agent.run(content, result_type=result_type, **kwargs)

    @asynccontextmanager
    async def run_stream[TResult](
        self,
        content: str,
        *,
        result_type: type[TResult] | None = None,
        output: DefaultOutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamedRunResult[AgentContext[TDeps], TResult]]:
        """Stream agent response."""
        if content.startswith("/"):
            await self.handle_command(
                content[1:],
                output=output,
                metadata=metadata,
            )
            return

        async with self.agent.run_stream(
            content, result_type=result_type, **kwargs
        ) as stream:
            yield stream
