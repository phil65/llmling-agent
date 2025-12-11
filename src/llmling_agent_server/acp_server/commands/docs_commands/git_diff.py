"""Fetch git diff for a specific commit."""

from __future__ import annotations

import uuid

from pydantic_ai import UserPromptPart
from slashed import CommandContext  # noqa: TC002

from llmling_agent.log import get_logger
from llmling_agent.messaging.context import NodeContext  # noqa: TC001
from llmling_agent_commands.base import NodeCommand
from llmling_agent_server.acp_server.session import ACPSession  # noqa: TC001


logger = get_logger(__name__)


class GitDiffCommand(NodeCommand):
    """Fetch git diff for a specific commit.

    Executes git diff client-side to get the complete diff
    and displays it in a structured format.
    """

    name = "git-diff"
    category = "docs"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        commit: str,
        *,
        base_commit: str | None = None,
        cwd: str | None = None,
    ) -> None:
        """Fetch git diff for a specific commit.

        Args:
            ctx: Command context with ACP session
            commit: Commit hash or reference to diff
            base_commit: Base commit to compare against (defaults to commit~1)
            cwd: Working directory to run git in (defaults to session cwd)
        """
        session = ctx.context.data
        assert session
        tool_call_id = f"git-diff-{uuid.uuid4().hex[:8]}"
        try:
            caps = session.client_capabilities  # Check if we have terminal access for running git
            if not (caps and caps.terminal and session.acp_agent.terminal_access):
                msg = "❌ **Terminal access not available for git operations**"
                await session.notifications.send_agent_text(msg)
                return

            # Build git diff command
            if base_commit:
                git_command = ["git", "diff", base_commit, commit]
                display_title = f"Git diff: {base_commit}..{commit}"
            else:
                git_command = ["git", "diff", f"{commit}~1", commit]
                display_title = f"Git diff: {commit}"

            # Start tool call
            await session.notifications.tool_call_start(
                tool_call_id=tool_call_id,
                title=display_title,
                kind="execute",
            )

            # Run git diff command
            output, exit_code = await session.requests.run_command(
                command=git_command[0],
                args=git_command[1:],
                cwd=cwd or session.cwd,
            )

            # Check if git command succeeded
            if exit_code != 0:
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="failed",
                    title=f"Git diff failed (exit code: {exit_code})",
                )
                return
            diff_content = output.strip()
            if not diff_content:
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="completed",
                    title="No changes found in diff",
                )
                return

            # Stage the diff content for use in agent context
            staged_part = UserPromptPart(content=f"Git diff for {display_title}:\n\n{diff_content}")
            session.add_staged_parts([staged_part])
            # Send successful result - wrap in code block for proper display
            staged_count = session.get_staged_parts_count()
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="completed",
                title=f"Git diff fetched and staged ({staged_count} total parts)",
                content=[f"```diff\n{diff_content}\n```"],
            )
        except Exception as e:
            logger.exception("Unexpected error fetching git diff", commit=commit)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="failed",
                title=f"Error: {e}",
            )
