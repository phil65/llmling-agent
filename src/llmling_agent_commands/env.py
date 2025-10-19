"""Environment management commands."""

from __future__ import annotations

import webbrowser

from llmling import Config, RuntimeConfig
from slashed import Command, CommandContext, CommandError, PathCompleter  # noqa: TC002

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent_config.environment import FileEnvironment, InlineEnvironment


SET_ENV_HELP = """\
Change the environment configuration file for the current session.

The environment file defines:
- Available tools
- Resource configurations
- Other runtime settings

Example: /set-env configs/new_env.yml

Note: This will reload the runtime configuration and update available tools.
"""

EDIT_ENV_HELP = """\
Open the agent's environment configuration file in the default editor.
This allows you to modify:
- Available tools
- Resources
- Other environment settings
"""


async def set_env(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Change the environment file path."""
    from upath import UPath

    if not args:
        await ctx.output.print("**Usage:** `/set-env <path>`")
        return

    env_path = args[0]
    if not UPath(env_path).exists():
        msg = f"Environment file not found: {env_path}"
        raise CommandError(msg)

    try:
        agent = ctx.context.agent
        if not agent.context.config:
            msg = "No agent context available"
            raise CommandError(msg)  # noqa: TRY301

        # Manually remove runtime tools
        tools = [name for name, info in agent.tools.items() if info.source == "runtime"]
        for name in tools:
            del agent.tools[name]

        # Clean up old runtime if we own it
        if agent._owns_runtime and agent.context.runtime:
            await agent.context.runtime.__aexit__(None, None, None)

        # Create and initialize new runtime
        config = Config.from_file(env_path)
        runtime = RuntimeConfig.from_config(config)
        agent.context.runtime = runtime
        agent._owns_runtime = True  # type: ignore

        # Re-initialize agent with new runtime
        await agent.__aenter__()

        await ctx.output.print(
            f"✅ **Environment changed to:** `{env_path}`\n"
            f"🔧 **Replaced runtime tools:** `{', '.join(tools)}`"
        )

    except Exception as e:
        msg = f"Failed to change environment: {e}"
        raise CommandError(msg) from e


async def edit_env(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Open agent's environment file in default application."""
    if not ctx.context.agent.context:
        msg = "No agent context available"
        raise CommandError(msg)

    config = ctx.context.agent.context.config
    match config.environment:
        case FileEnvironment(uri=uri):
            # For file environments, open in browser
            try:
                webbrowser.open(uri)
                await ctx.output.print(f"🌐 **Opening environment file:** `{uri}`")
            except Exception as e:
                msg = f"Failed to open environment file: {e}"
                raise CommandError(msg) from e
        case InlineEnvironment() as cfg:
            # For inline environments, display the configuration
            yaml_config = cfg.model_dump_yaml()
            await ctx.output.print(
                f"📝 **Inline environment configuration:**\n\n```yaml\n{yaml_config}\n```"
            )
        case str() as path:
            # Legacy string path
            try:
                resolved = config._resolve_environment_path(path, config.config_file_path)
                webbrowser.open(resolved)
                await ctx.output.print(f"🌐 **Opening environment file:** `{resolved}`")
            except Exception as e:
                msg = f"Failed to open environment file: {e}"
                raise CommandError(msg) from e
        case None:
            await ctx.output.print("ℹ️ **No environment configured**")  #  noqa: RUF001


set_env_cmd = Command(
    name="set-env",
    description="Change the environment configuration file",
    execute_func=set_env,
    usage="<path>",
    help_text=SET_ENV_HELP,
    category="environment",
    completer=PathCompleter(file_patterns=["*.yml", "*.yaml"]),
)

edit_env_cmd = Command(
    name="open-env-file",
    description="Open the agent's environment configuration",
    execute_func=edit_env,
    help_text=EDIT_ENV_HELP,
    category="environment",
    completer=PathCompleter(file_patterns=["*.yml", "*.yaml"]),
)
