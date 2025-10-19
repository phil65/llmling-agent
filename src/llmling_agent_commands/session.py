"""Agent session slash commands."""

from __future__ import annotations

from slashed import CommandContext, SlashedCommand  # noqa: TC002

from llmling_agent.agent.context import AgentContext  # noqa: TC001


RESET_HELP = """\
Reset the entire session state:
- Clears chat history
- Restores default tool settings
- Resets any session-specific configurations
"""

CLEAR_HELP = """\
Clear the current chat session history.
This removes all previous messages but keeps tools and settings.
"""


class ClearCommand(SlashedCommand):
    """Clear the current chat session history.

    This removes all previous messages but keeps tools and settings.
    """

    name = "clear"
    category = "session"

    async def execute_command(self, ctx: CommandContext[AgentContext]):
        """Clear chat history.

        Args:
            ctx: Command context
        """
        ctx.context.agent.conversation.clear()
        await ctx.output.print("🧹 **Chat history cleared**")


class ResetCommand(SlashedCommand):
    """Reset the entire session state.

    - Clears chat history
    - Restores default tool settings
    - Resets any session-specific configurations
    """

    name = "reset"
    category = "session"

    async def execute_command(self, ctx: CommandContext[AgentContext]):
        """Reset session state.

        Args:
            ctx: Command context
        """
        ctx.context.agent.reset()
        await ctx.output.print(
            "🔄 **Session state reset** - history cleared, tools and settings restored"
        )
