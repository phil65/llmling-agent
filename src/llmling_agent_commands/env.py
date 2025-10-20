"""Environment management commands."""

from __future__ import annotations

import webbrowser

from llmling import Config, RuntimeConfig
from slashed import CommandContext, CommandError, SlashedCommand  # noqa: TC002
from slashed.completers import PathCompleter

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


class SetEnvCommand(SlashedCommand):
    """Change the environment configuration file for the current session.

    The environment file defines:
    - Available tools
    - Resource configurations
    - Other runtime settings

    Example: /set-env configs/new_env.yml

    Note: This will reload the runtime configuration and update available tools.
    """

    name = "set-env"
    category = "environment"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        path: str,
    ):
        """Change the environment file path.

        Args:
            ctx: Command context
            path: Path to environment file
        """
        from upath import UPath

        if not UPath(path).exists():
            msg = f"Environment file not found: {path}"
            raise CommandError(msg)

        try:
            agent = ctx.context.agent
            if not agent.context.config:
                msg = "No agent context available"
                raise CommandError(msg)  # noqa: TRY301

            # Manually remove runtime tools
            tools = [
                name for name, info in agent.tools.items() if info.source == "runtime"
            ]
            for name in tools:
                del agent.tools[name]

            # Clean up old runtime if we own it
            if agent._owns_runtime and agent.context.runtime:
                await agent.context.runtime.__aexit__(None, None, None)

            # Create and initialize new runtime
            config = Config.from_file(path)
            runtime = RuntimeConfig.from_config(config)
            agent.context.runtime = runtime
            agent._owns_runtime = True  # type: ignore

            # Re-initialize agent with new runtime
            await agent.__aenter__()

            await ctx.output.print(
                f"✅ **Environment changed to:** `{path}`\n"
                f"🔧 **Replaced runtime tools:** `{', '.join(tools)}`"
            )

        except Exception as e:
            msg = f"Failed to change environment: {e}"
            raise CommandError(msg) from e

    def get_completer(self):
        """Get completer for YAML files."""
        return PathCompleter(file_patterns=["*.yml", "*.yaml"])


class OpenEnvFileCommand(SlashedCommand):
    """Open the agent's environment configuration file in the default editor.

    This allows you to modify:
    - Available tools
    - Resources
    - Other environment settings
    """

    name = "open-env-file"
    category = "environment"

    async def execute_command(self, ctx: CommandContext[AgentContext]):
        """Open agent's environment file in default application.

        Args:
            ctx: Command context
        """
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
                    "📝 **Inline environment configuration:**\n\n"
                    f"```yaml\n{yaml_config}\n```"
                )
            case str() as path:
                # Legacy string path
                try:
                    resolved = config._resolve_environment_path(
                        path, config.config_file_path
                    )
                    webbrowser.open(resolved)
                    await ctx.output.print(
                        f"🌐 **Opening environment file:** `{resolved}`"
                    )
                except Exception as e:
                    msg = f"Failed to open environment file: {e}"
                    raise CommandError(msg) from e
            case None:
                await ctx.output.print("ℹ️ **No environment configured**")  #  noqa: RUF001

    def get_completer(self):
        """Get completer for YAML files."""
        return PathCompleter(file_patterns=["*.yml", "*.yaml"])
