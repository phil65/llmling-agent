"""Model-related commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slashed import CommandContext  # noqa: TC002
from slashed.completers import CallbackCompleter

from llmling_agent.agents.context import AgentContext  # noqa: TC001
from llmling_agent_commands.base import NodeCommand
from llmling_agent_commands.completers import get_model_names


if TYPE_CHECKING:
    from llmling_agent.messaging import MessageNode


class SetModelCommand(NodeCommand):
    """Change the language model for the current conversation.

    The model change takes effect immediately for all following messages.
    Previous messages and their context are preserved.

    Examples:
      /set-model gpt-5
      /set-model openai:gpt-5-mini
      /set-model claude-2

    Note: Available models depend on your configuration and API access.
    """

    name = "set-model"
    category = "model"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext],
        model: str,
    ) -> None:
        """Change the model for the current conversation.

        Args:
            ctx: Command context
            model: Model name to switch to
        """
        try:
            # Create new session with model override
            ctx.context.native_agent.set_model(model)
            await ctx.print(f"✅ **Model changed to:** `{model}`")
        except Exception as e:  # noqa: BLE001
            await ctx.print(f"❌ **Failed to change model:** {e}")

    def get_completer(self) -> CallbackCompleter:
        """Get completer for model names."""
        return CallbackCompleter(get_model_names)

    @classmethod
    def supports_node(cls, node: MessageNode[Any, Any]) -> bool:
        from llmling_agent import Agent

        return isinstance(node, Agent)
