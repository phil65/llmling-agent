"""Agent provider implementations."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from llmling import ToolError
import logfire
from slashed import CommandStore, DefaultOutputWriter, parse_command

from llmling_agent.log import get_logger
from llmling_agent.prompts.convert import format_prompts
from llmling_agent_providers.base import AgentProvider, ProviderResponse
from llmling_agent_providers.human.utils import get_textual_streaming_app


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent.agent.context import AgentContext
    from llmling_agent.common_types import ModelType
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.models.content import Content
    from llmling_agent_providers.base import UsageLimits


logger = get_logger(__name__)


class HumanProvider(AgentProvider):
    """Provider for human-in-the-loop responses."""

    model = None
    NAME = "human"

    def __init__(
        self,
        *,
        name: str = "human",
        context: AgentContext | None = None,
        timeout: int | None = None,
        show_context: bool = True,
        command_store: CommandStore | None = None,
        use_promptantic: bool = True,
        debug: bool = False,
    ):
        """Initialize human provider."""
        from llmling_agent_commands import get_commands

        super().__init__(context=context)
        self.name = name or "human"
        self._debug = debug
        self._timeout = timeout
        self._show_context = show_context
        self.use_promptantic = use_promptantic
        self.commands = command_store or CommandStore()
        for cmd in get_commands():
            self.commands.register_command(cmd)

    def __repr__(self) -> str:
        return f"Human({self.name!r})"

    @logfire.instrument("Human input. result type {result_type}")
    async def generate_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelType = None,
        system_prompt: str | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Get response through human inut."""
        # Show prompt and get response
        formatted = await format_prompts(prompts)
        content = await self.context.get_input_provider().get_input(
            self.context,
            formatted,
            result_type=result_type,
            message_history=message_history,
        )
        return ProviderResponse(content=content)

    async def stream_events(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelType = None,
        system_prompt: str | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream response events keystroke by keystroke."""
        from pydantic_ai.messages import PartDeltaEvent, TextPartDelta
        from pydantic_ai.run import AgentRunResultEvent

        prompt = await format_prompts(prompts)
        print(f"\n{prompt}")
        if result_type:
            print(f"(Please provide response as {result_type.__name__})")

        chunk_queue: asyncio.Queue[str] = asyncio.Queue()

        async def handle_chunk(chunk: str):
            await chunk_queue.put(chunk)

        # Emit start event
        from pydantic_ai.messages import PartStartEvent, TextPart
        from pydantic_ai.run import AgentRunResult

        yield PartStartEvent(index=0, part=TextPart(content=""))

        collected_chunks = []

        try:
            # Run textual app in background
            textual_app = get_textual_streaming_app()
            app = textual_app(handle_chunk)

            # Start the app task
            app_task = asyncio.create_task(app.run_async())

            # Stream chunks as they arrive
            while not app_task.done():
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.1)
                    collected_chunks.append(chunk)
                    # Emit delta event for each chunk
                    yield PartDeltaEvent(
                        index=0, delta=TextPartDelta(content_delta=chunk)
                    )
                except TimeoutError:
                    continue

            # Get final content from completed app
            content = await app_task

            # Emit any remaining chunks
            while not chunk_queue.empty():
                chunk = await chunk_queue.get()
                collected_chunks.append(chunk)
                yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=chunk))

            # Parse structured response if needed
            if result_type:
                try:
                    content = result_type.model_validate_json(content)
                except Exception as e:
                    logger.exception("Failed to parse structured response")
                    error_msg = f"Invalid response format: {e}"
                    raise ToolError(error_msg) from e

            final_content = str(content)

            result = AgentRunResult(output=final_content)
            yield AgentRunResultEvent(result=result)

        except Exception:
            logger.exception("Error in human streaming")
            raise

    async def handle_input(self, content: str):
        """Handle all human input."""
        if not content.strip():
            return

        try:
            if content.startswith("/"):
                # Regular command
                parsed = parse_command(content[1:])

                await self.commands.execute_command_with_context(
                    parsed.name,
                    context=self.context,
                    output_writer=DefaultOutputWriter(),
                )

            elif content.startswith("@"):
                # Agent-specific interaction
                agent_name, message = content[1:].split(maxsplit=1)
                if not self.context.pool:
                    logger.error("No agent pool available")
                    return

                agent = self.context.pool.get_agent(agent_name)
                if message.startswith("/"):
                    # Command for specific agent
                    parsed = parse_command(message[1:])

                    await self.commands.execute_command_with_context(
                        parsed.name,
                        context=agent.context,
                        output_writer=DefaultOutputWriter(),
                    )
                else:
                    await agent.run(message)

        except Exception:
            logger.exception("Failed to handle input")
