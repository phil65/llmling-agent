"""Slash command wrapper for Agent that injects command events into streams."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from llmling_agent.agent.events import CommandCompleteEvent, CommandOutputEvent
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from slashed import CommandStore, OutputWriter

    from llmling_agent.agent.agent import Agent
    from llmling_agent.agent.events import SlashedAgentStreamEvent


logger = get_logger(__name__)
SLASH_PATTERN = re.compile(r"^/([\w-]+)(?:\s+(.*))?$")


def _parse_slash_command(command_text: str) -> tuple[str, str] | None:
    """Parse slash command into name and args.

    Args:
        command_text: Full command text

    Returns:
        Tuple of (command_name, args) or None if invalid
    """
    if match := SLASH_PATTERN.match(command_text.strip()):
        command_name = match.group(1)
        args = match.group(2) or ""
        return command_name, args.strip()
    return None


class StreamingOutputWriter:
    """OutputWriter that captures command output and emits it as events."""

    def __init__(
        self,
        command: str,
        event_sink: Sequence[CommandOutputEvent | CommandCompleteEvent],
    ) -> None:
        """Initialize with command name and event sink.

        Args:
            command: Name of the command being executed
            event_sink: List to append output events to
        """
        self.command = command
        self.event_sink = list(event_sink)

    async def print(self, message: str = "", **kwargs: Any) -> None:
        """Capture print output as command output event.

        Args:
            message: The message to output
            **kwargs: Additional print arguments (ignored)
        """
        if message:  # Only emit non-empty messages
            event = CommandOutputEvent(command=self.command, output=message)
            self.event_sink.append(event)


class SlashedAgent[TDeps, OutputDataT]:
    """Wrapper around Agent that handles slash commands in streams.

    Uses the "commands first" strategy from the ACP adapter:
    1. Execute all slash commands first
    2. Then process remaining content through wrapped agent
    3. If only commands, end without LLM processing
    """

    def __init__(
        self,
        agent: Agent[TDeps, OutputDataT],
        command_store: CommandStore | None = None,
        *,
        output_writer_factory: Callable[[str], OutputWriter] | None = None,
        context_data_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize with wrapped agent and command store.

        Args:
            agent: The agent to wrap
            command_store: Command store for slash commands(creates default if None)
            output_writer_factory: Optional factory for creating output writers
            context_data_factory: Optional factory for creating command context data
        """
        from llmling_agent_commands import create_default_command_store

        self.agent = agent
        self.command_store = command_store or create_default_command_store()
        self._output_writer_factory = output_writer_factory
        self._context_data_factory = context_data_factory

    def _is_slash_command(self, text: str) -> bool:
        """Check if text starts with a slash command.

        Args:
            text: Text to check

        Returns:
            True if text is a slash command
        """
        return bool(SLASH_PATTERN.match(text.strip()))

    async def _execute_slash_command(
        self, command_text: str
    ) -> list[CommandOutputEvent | CommandCompleteEvent]:
        """Execute a single slash command and return events.

        Args:
            command_text: Full command text including slash

        Returns:
            List of events from command execution
        """
        events: list[CommandOutputEvent | CommandCompleteEvent] = []

        parsed = _parse_slash_command(command_text)
        if not parsed:
            logger.warning("Invalid slash command", command=command_text)
            events.append(CommandCompleteEvent(command="unknown", success=False))
            return events

        command_name, args = parsed

        # Create output writer that captures to events
        if self._output_writer_factory:
            output_writer = self._output_writer_factory(command_name)
        else:
            output_writer = StreamingOutputWriter(command_name, events)

        # Create command context
        context_data = (
            self._context_data_factory()
            if self._context_data_factory
            else self.agent.context
        )

        cmd_ctx = self.command_store.create_context(
            data=context_data, output_writer=output_writer
        )

        # Execute command
        command_str = f"{command_name} {args}".strip()
        success = True
        try:
            await self.command_store.execute_command(command_str, cmd_ctx)
        except Exception as e:
            logger.exception("Command execution failed", command=command_name)
            events.append(
                CommandOutputEvent(command=command_name, output=f"Command error: {e}")
            )
            success = False

        # Add completion event
        events.append(CommandCompleteEvent(command=command_name, success=success))
        return events

    async def run_stream(
        self, *prompts: Any, **kwargs: Any
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
                logger.debug("Found slash command: %r", prompt)
                commands.append(prompt.strip())
            else:
                regular_prompts.append(prompt)

        # Execute all commands first
        if commands:
            for command in commands:
                logger.info("Processing slash command", command=command)
                command_events = await self._execute_slash_command(command)
                for cmd_event in command_events:
                    yield cmd_event

        # If we have regular content, process it through the agent
        if regular_prompts:
            logger.debug(
                "Processing prompts through agent", num_prompts=len(regular_prompts)
            )
            async for event in self.agent.run_stream(*regular_prompts, **kwargs):
                yield event

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

    async def main():
        agent = Agent("test-agent", model="test", session=False)
        slashed = SlashedAgent(agent)  # Uses built-in commands by default

        print("Testing SlashedAgent with built-in commands:")
        async for event in slashed.run_stream("/list-tools"):
            print(event)

    asyncio.run(main())
