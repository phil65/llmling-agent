"""Slash command wrapper for Agent that injects command events into streams."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any, cast

from slashed.events import (
    CommandExecutedEvent,
    CommandOutputEvent as SlashedCommandOutputEvent,
)

from llmling_agent.agents.events import CommandCompleteEvent, CommandOutputEvent
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from slashed import CommandContext, CommandStore
    from slashed.events import CommandStoreEvent

    from llmling_agent.agents.base_agent import BaseAgent
    from llmling_agent.agents.events import SlashedAgentStreamEvent
    from llmling_agent.common_types import PromptCompatible


logger = get_logger(__name__)
SLASH_PATTERN = re.compile(r"^/([\w-]+)(?:\s+(.*))?$")


def _parse_slash_command(command_text: str) -> tuple[str, str] | None:
    """Parse slash command into name and args.

    Args:
        command_text: Full command text

    Returns:
        Tuple of (cmd_name, args) or None if invalid
    """
    if match := SLASH_PATTERN.match(command_text.strip()):
        cmd_name = match.group(1)
        args = match.group(2) or ""
        return cmd_name, args.strip()
    return None


class SlashedAgent[TDeps, OutputDataT]:
    """Wrapper around Agent that handles slash commands in streams.

    Uses the "commands first" strategy from the ACP adapter:
    1. Execute all slash commands first
    2. Then process remaining content through wrapped agent
    3. If only commands, end without LLM processing
    """

    def __init__(
        self,
        agent: BaseAgent[TDeps, OutputDataT],
        command_store: CommandStore | None = None,
        *,
        context_data_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize with wrapped agent and command store.

        Args:
            agent: The agent to wrap
            command_store: Command store for slash commands (creates default if None)
            context_data_factory: Optional factory for creating command context data
        """
        self.agent = agent
        self._context_data_factory = context_data_factory
        self._event_queue: asyncio.Queue[CommandStoreEvent] | None = None

        # Create store with our streaming event handler
        if command_store is None:
            from slashed import CommandStore

            from llmling_agent_commands import get_commands

            cmds = get_commands()
            self.command_store = CommandStore(event_handler=self._emit_event, commands=cmds)
        else:
            self.command_store = command_store

    async def _emit_event(self, event: CommandStoreEvent) -> None:
        """Bridge store events to async queue during command execution."""
        if self._event_queue:
            await self._event_queue.put(event)

    def _is_slash_command(self, text: str) -> bool:
        """Check if text starts with a slash command.

        Args:
            text: Text to check

        Returns:
            True if text is a slash command
        """
        return bool(SLASH_PATTERN.match(text.strip()))

    async def _execute_slash_command_streaming(
        self, command_text: str
    ) -> AsyncGenerator[CommandOutputEvent | CommandCompleteEvent]:
        """Execute a single slash command and yield events as they happen.

        Args:
            command_text: Full command text including slash

        Yields:
            Command output and completion events
        """
        parsed = _parse_slash_command(command_text)
        if not parsed:
            logger.warning("Invalid slash command", command=command_text)
            yield CommandCompleteEvent(command="unknown", success=False)
            return

        cmd_name, args = parsed

        # Set up event queue for this command execution
        self._event_queue = asyncio.Queue()
        context_data = (  # Create command context
            self._context_data_factory() if self._context_data_factory else self.agent.context
        )

        cmd_ctx = self.command_store.create_context(data=context_data)
        command_str = f"{cmd_name} {args}".strip()
        execute_task = asyncio.create_task(self.command_store.execute_command(command_str, cmd_ctx))

        success = True
        try:
            # Yield events from queue as command runs
            while not execute_task.done():
                try:
                    # Wait for events with short timeout to check task completion
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                    # Convert store events to our stream events
                    match event:
                        case SlashedCommandOutputEvent(output=output):
                            yield CommandOutputEvent(command=cmd_name, output=output)
                        case CommandExecutedEvent(success=False, error=error) if error:
                            output = f"Command error: {error}"
                            yield CommandOutputEvent(command=cmd_name, output=output)
                            success = False
                except TimeoutError:
                    continue

            # Ensure command task completes and handle any remaining events
            try:
                await execute_task
            except Exception as e:
                logger.exception("Command execution failed", command=cmd_name)
                success = False
                yield CommandOutputEvent(command=cmd_name, output=f"Command error: {e}")

            # Drain any remaining events from queue
            while not self._event_queue.empty():
                try:
                    match self._event_queue.get_nowait():
                        case SlashedCommandOutputEvent(output=output):
                            yield CommandOutputEvent(command=cmd_name, output=output)
                except asyncio.QueueEmpty:
                    break

            # Always yield completion event
            yield CommandCompleteEvent(command=cmd_name, success=success)

        finally:
            # Clean up event queue
            self._event_queue = None

    async def run_stream(
        self,
        *prompts: PromptCompatible,
        **kwargs: Any,
    ) -> AsyncGenerator[SlashedAgentStreamEvent[OutputDataT]]:
        """Run agent with slash command support.

        Separates slash commands from regular prompts, executes commands first,
        then processes remaining content through the wrapped agent.

        Args:
            *prompts: Input prompts (may include slash commands)
            **kwargs: Additional arguments passed to agent.run_stream

        Yields:
            Stream events from command execution and agent processing
        """
        # Separate slash commands from regular content
        commands: list[str] = []
        regular_prompts: list[Any] = []

        for prompt in prompts:
            if isinstance(prompt, str) and self._is_slash_command(prompt):
                logger.debug("Found slash command", command=prompt)
                commands.append(prompt.strip())
            else:
                regular_prompts.append(prompt)

        # Execute all commands first with streaming
        if commands:
            for command in commands:
                logger.info("Processing slash command", command=command)
                async for cmd_event in self._execute_slash_command_streaming(command):
                    yield cmd_event

        # If we have regular content, process it through the agent
        if regular_prompts:
            logger.debug("Processing prompts through agent", num_prompts=len(regular_prompts))
            async for event in self.agent.run_stream(*regular_prompts, **kwargs):
                # ACPAgent always returns str, cast to match OutputDataT
                yield cast("SlashedAgentStreamEvent[OutputDataT]", event)

        # If we only had commands and no regular content, we're done
        # (no additional events needed)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped agent.

        Args:
            name: Attribute name

        Returns:
            Attribute value from wrapped agent
        """
        return getattr(self.agent, name)


if __name__ == "__main__":
    import asyncio

    from llmling_agent import Agent

    async def main() -> None:
        agent = Agent("test-agent", model="test", session=False)
        slashed = SlashedAgent(agent)  # Uses built-in commands by default

        # Add a simple test command that outputs multiple lines
        @slashed.command_store.command(name="test-streaming", category="test")
        async def test_streaming(ctx: CommandContext[Any], *args: Any, **kwargs: Any) -> None:
            """Test command that outputs multiple lines."""
            await ctx.print("Starting streaming test...")
            for i in range(3):
                await ctx.print(f"Output line {i + 1}")
                await asyncio.sleep(0.1)  # Small delay to simulate work
            await ctx.print("Streaming test complete!")

        print("Testing SlashedAgent streaming:")
        async for event in slashed.run_stream("/test-streaming"):
            print(f"Event: {event}")

    asyncio.run(main())
