"""Agent management commands."""

from __future__ import annotations

from slashed import CommandContext, CommandError, SlashedCommand  # noqa: TC002
from slashed.completers import CallbackCompleter

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.messaging.context import NodeContext  # noqa: TC001
from llmling_agent_commands.completers import get_available_agents
from llmling_agent_commands.markdown_utils import format_table


CREATE_AGENT_HELP = """\
Create a new agent in the current session.

Creates a temporary agent that inherits the current agent's model.
The new agent will exist only for this session.

Options:
  --system-prompt "prompt"   System instructions for the agent (required)
  --model model_name        Override model (default: same as current agent)
  --role role_name         Agent role (assistant/specialist/overseer)
  --description "text"     Optional description of the agent
  --tools "import_path1|import_path2"   Optional list tools (by import path)

Examples:
  # Create poet using same model as current agent
  /create-agent poet --system-prompt "Create poems from any text"

  # Create analyzer with different model
  /create-agent analyzer --system-prompt "Analyze in detail" --model gpt-5

  # Create specialized helper
  /create-agent helper --system-prompt "Debug code" --role specialist
"""

SHOW_AGENT_HELP_TEXT = """\
Display the complete configuration of the current agent as YAML.
Shows:
- Basic agent settings
- Model configuration (with override indicators)
- Environment settings (including inline environments)
- System prompts
- Response type configuration
- Other settings

Fields that have been overridden at runtime are marked with comments.
"""

LIST_AGENTS_HELP = """\
Show all agents defined in the current configuration.
Displays:
- Agent name
- Model used (if specified)
- Description (if available)
"""

SWITCH_AGENT_HELP = """\
Switch the current chat session to a different agent.
Use /list-agents to see available agents.

Example: /switch-agent url_opener
"""


class CreateAgentCommand(SlashedCommand):
    """Create a new agent in the current session.

    Creates a temporary agent that inherits the current agent's model.
    The new agent will exist only for this session.

    Options:
      --system-prompt "prompt"   System instructions for the agent (required)
      --model model_name        Override model (default: same as current agent)
      --role role_name         Agent role (assistant/specialist/overseer)
      --description "text"     Optional description of the agent
      --tools "import_path1|import_path2"   Optional list tools (by import path)

    Examples:
      # Create poet using same model as current agent
      /create-agent poet --system-prompt "Create poems from any text"

      # Create analyzer with different model
      /create-agent analyzer --system-prompt "Analyze in detail" --model gpt-5

      # Create specialized helper
      /create-agent helper --system-prompt "Debug code" --role specialist
    """

    name = "create-agent"
    category = "agents"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        agent_name: str,
        *,
        system_prompt: str,
        model: str | None = None,
        role: str | None = None,
        description: str | None = None,
        tools: str | None = None,
    ):
        """Create a new agent in the current session.

        Args:
            ctx: Command context
            agent_name: Name for the new agent
            system_prompt: System instructions for the agent
            model: Override model (default: same as current agent)
            role: Agent role (assistant/specialist/overseer)
            description: Optional description of the agent
            tools: Optional pipe-separated list of tools (by import path)
        """
        try:
            if not ctx.context.pool:
                msg = "No agent pool available"
                raise CommandError(msg)

            # Get model from args or current agent
            current_agent = ctx.context.agent

            # Parse tools if provided
            tool_list = None
            if tools:
                tool_list = [t.strip() for t in tools.split("|")]

            # Create and register the new agent
            await ctx.context.pool.add_agent(
                name=agent_name,
                model=model or current_agent.model_name,
                system_prompt=system_prompt or (),
                description=description,
                tools=tool_list,
            )

            msg = f"✅ **Created agent** `{agent_name}`"
            if tool_list:
                msg += f" with tools: `{', '.join(tool_list)}`"
            await ctx.output.print(
                f"{msg}\n\n💡 Use `/connect {agent_name}` to forward messages"
            )

        except ValueError as e:
            msg = f"Failed to create agent: {e}"
            raise CommandError(msg) from e


class ShowAgentCommand(SlashedCommand):
    """Display the complete configuration of the current agent as YAML.

    Shows:
    - Basic agent settings
    - Model configuration (with override indicators)
    - Environment settings (including inline environments)
    - System prompts
    - Response type configuration
    - Other settings

    Fields that have been overridden at runtime are marked with comments.
    """

    name = "show-agent"
    category = "agents"

    async def execute_command(self, ctx: CommandContext[NodeContext]):
        """Show current agent's configuration.

        Args:
            ctx: Command context
        """
        import yamling

        if not ctx.context.node.context:
            await ctx.output.print("❌ **No node context available**")
            return

        # Get the agent's config with current overrides
        config = ctx.context.node.context.config

        # Get base config as dict
        config_dict = config.model_dump(exclude_none=True)

        # Format as annotated YAML
        yaml_config = yamling.dump_yaml(
            config_dict,
            sort_keys=False,
            indent=2,
            default_flow_style=False,
            allow_unicode=True,
        )
        # Add header and format for display
        sections = [
            "\n[bold]Current Node Configuration:[/]",
            "```yaml",
            yaml_config,
            "```",
        ]

        await ctx.output.print("\n".join(sections))


class ListAgentsCommand(SlashedCommand):
    """Show all agents defined in the current configuration.

    Displays:
    - Agent name
    - Model used (if specified)
    - Description (if available)
    """

    name = "list-agents"
    category = "agents"

    async def execute_command(self, ctx: CommandContext[AgentContext]):
        """List all available agents.

        Args:
            ctx: Command context
        """
        if not ctx.context.pool:
            msg = "No agent pool available"
            raise CommandError(msg)

        rows = []
        for name, agent in ctx.context.pool.agents.items():
            typ = "dynamic" if name not in ctx.context.definition.agents else "static"
            rows.append({
                "Name": name,
                "Model": str(agent.model_name or ""),
                "Type": typ,
                "Description": agent.description or "",
            })

        headers = ["Name", "Model", "Type", "Description"]
        table = format_table(headers, rows)
        await ctx.output.print(f"## 🤖 Available Agents\n\n{table}")


class SwitchAgentCommand(SlashedCommand):
    """Switch the current chat session to a different agent.

    Use /list-agents to see available agents.

    Example: /switch-agent url_opener
    """

    name = "switch-agent"
    category = "agents"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        agent_name: str,
    ):
        """Switch to a different agent.

        Args:
            ctx: Command context
            agent_name: Name of the agent to switch to
        """
        msg = "Temporarily disabled"
        raise RuntimeError(msg)

    def get_completer(self):
        """Get completer for agent names."""
        return CallbackCompleter(get_available_agents)
