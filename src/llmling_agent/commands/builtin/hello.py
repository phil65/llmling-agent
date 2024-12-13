"""Simple hello command for testing."""

from __future__ import annotations

from llmling_agent.commands.base import Command, CommandContext


async def hello(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Say hello to someone."""
    name = args[0] if args else "World"
    greeting = kwargs.get("greeting", "Hello")
    await ctx.output.print(f"{greeting}, {name}!")


hello_command = Command(
    name="hello",
    description="Say hello",
    execute_func=hello,
    usage="[name] [--greeting msg]",
    help_text="Greet someone. Use --greeting to customize the message.",
)