"""Environment management commands."""

from __future__ import annotations

import webbrowser

from llmling.config.runtime import RuntimeConfig
from upath import UPath

from llmling_agent.commands.base import Command, CommandContext, CommandError
from llmling_agent.commands.completers import PathCompleter


async def show_env(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Show current environment file path."""
    if not ctx.session._agent._context:
        msg = "No agent context available"
        raise CommandError(msg)

    config = ctx.session._agent._context.config
    try:
        resolved_path = config.get_environment_path()
        await ctx.output.print(f"Current environment file: {resolved_path}")
    except Exception as e:
        msg = f"Failed to get environment file: {e}"
        raise CommandError(msg) from e


async def set_env(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Change the environment file path."""
    if not args:
        await ctx.output.print("Usage: /set-env <path>")
        return

    env_path = args[0]
    if not UPath(env_path).exists():
        msg = f"Environment file not found: {env_path}"
        raise CommandError(msg)

    try:
        # Get current agent configuration
        agent = ctx.session._agent
        if not agent._context or not agent._context.config:
            msg = "No agent context available"
            raise CommandError(msg)  # noqa: TRY301

        # Update environment path in config
        config = agent._context.config
        config = config.model_copy(update={"environment": env_path})

        # Create new runtime with updated config
        async with RuntimeConfig.open(config.get_config()) as new_runtime:
            # Create new agent with updated runtime
            new_agent = agent.__class__(
                runtime=new_runtime,
                context=agent._context,
                **agent._context.config.get_agent_kwargs(),
            )

            # Update session's agent
            ctx.session._agent = new_agent
            # Reset session state but keep history
            ctx.session._tool_states = new_agent.list_tools()

            await ctx.output.print(
                f"Environment changed to: {env_path}\n"
                "Session updated with new configuration."
            )

    except Exception as e:
        msg = f"Failed to change environment: {e}"
        raise CommandError(msg) from e


async def edit_env(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Open agent's environment file in default application."""
    if not ctx.session._agent._context:
        msg = "No agent context available"
        raise CommandError(msg)

    config = ctx.session._agent._context.config
    try:
        resolved_path = config.get_environment_path()
        webbrowser.open(resolved_path)
        await ctx.output.print(f"Opening environment file: {resolved_path}")
    except Exception as e:
        msg = f"Failed to open environment file: {e}"
        raise CommandError(msg) from e


show_env_cmd = Command(
    name="show-env",
    description="Show current environment file path",
    execute_func=show_env,
    help_text=(
        "Display the path to the current environment configuration file.\n"
        "This file defines available tools and resources."
    ),
    category="environment",
)

set_env_cmd = Command(
    name="set-env",
    description="Change the environment configuration file",
    execute_func=set_env,
    usage="<path>",
    help_text=(
        "Change the environment configuration file for the current session.\n\n"
        "The environment file defines:\n"
        "- Available tools\n"
        "- Resource configurations\n"
        "- Other runtime settings\n\n"
        "Example: /set-env configs/new_env.yml\n\n"
        "Note: This will reload the runtime configuration and update available tools."
    ),
    category="environment",
    completer=PathCompleter(file_patterns=["*.yml", "*.yaml"]),
)

edit_env_cmd = Command(
    name="open-env-file",
    description="Open the agent's environment configuration",
    execute_func=edit_env,
    help_text=(
        "Open the agent's environment configuration file in the default editor.\n"
        "This allows you to modify:\n"
        "- Available tools\n"
        "- Resources\n"
        "- Other environment settings"
    ),
    category="environment",
    completer=PathCompleter(file_patterns=["*.yml", "*.yaml"]),
)