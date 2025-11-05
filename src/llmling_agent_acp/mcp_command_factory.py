"""Factory for converting MCP prompts to slashed commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from slashed import CommandContext

    from llmling_agent.mcp_server.manager import Prompt
    from llmling_agent_acp.session import ACPSession


logger = get_logger(__name__)


def create_mcp_command(prompt: Prompt, session: ACPSession) -> Command:
    """Convert MCP prompt to slashed Command.

    Args:
        prompt: MCP prompt to wrap
        session: ACP session for execution context

    Returns:
        Slashed Command that executes the prompt
    """

    async def execute_prompt(
        ctx: CommandContext,
        args: list[str],
        kwargs: dict[str, str],
    ) -> None:
        """Execute the MCP prompt with parsed arguments."""
        # Map parsed args to prompt parameters
        arguments = _map_to_prompt_args(prompt, args, kwargs)

        try:
            # Get prompt components
            components = await prompt.get_components(arguments)
            session.add_staged_parts(components)

            # Send confirmation
            staged_count = session.get_staged_parts_count()
            await ctx.print(
                f"✅ Prompt '{prompt.name}' staged ({staged_count} total parts)"
            )

        except Exception as e:
            logger.exception("MCP prompt execution failed", prompt=prompt.name)
            await ctx.print(f"❌ Prompt error: {e}")

    return Command(
        execute_func=execute_prompt,
        name=prompt.name,
        description=prompt.description or f"MCP prompt: {prompt.name}",
        category="mcp",
        usage=_generate_usage_hint(prompt),
    )


def _map_to_prompt_args(
    prompt: Prompt,
    args: list[str],
    kwargs: dict[str, str],
) -> dict[str, str] | None:
    """Map parsed command args to prompt parameters.

    Args:
        prompt: MCP prompt with argument definitions
        args: Positional arguments from command line
        kwargs: Keyword arguments from command line

    Returns:
        Mapped arguments for prompt, or None if no arguments
    """
    if not prompt.arguments:
        return None

    result = {}

    # Map positional args to prompt parameter names
    for i, arg_value in enumerate(args):
        if i < len(prompt.arguments):
            param_name = prompt.arguments[i]["name"]
            result[param_name] = arg_value

    # Add keyword arguments
    result.update(kwargs)

    return result if result else None


def _generate_usage_hint(prompt: Prompt) -> str | None:
    """Generate usage hint from prompt arguments.

    Args:
        prompt: MCP prompt with argument definitions

    Returns:
        Usage hint string or None if no arguments
    """
    if not prompt.arguments:
        return None

    return " ".join(f"<{arg['name']}>" for arg in prompt.arguments)
