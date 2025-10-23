"""ACP-specific slash commands for session management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from slashed import SlashedCommand  # noqa: TC002


if TYPE_CHECKING:
    from llmling_agent_acp.session import ACPSession


@dataclass
class ACPCommandContext:
    """Context for ACP-specific commands that includes session information."""

    session: ACPSession
    """The ACP session associated with this command context."""


# class ListSessionsCommand(SlashedCommand):
#     """List all available ACP sessions.

#     Shows:
#     - Session ID and status (active/stored)
#     - Agent name and working directory
#     - Creation time and message count
#     - Storage information
#     """

#     name = "list-sessions"
#     category = "acp"

#     async def execute_command(
#         self,
#         ctx: CommandContext[ACPCommandContext],
#         *,
#         active: bool = False,
#         stored: bool = False,
#     ):
#         """List available ACP sessions.

#         Args:
#             ctx: Command context with ACP session
#             active: Show only active sessions
#             stored: Show only stored sessions
#         """
#         session = ctx.context.session

#         # Check if we have access to session manager
#         if not session.manager:
#             await ctx.output.print("❌ **Session manager not available**")
#             return

#         # If no filter specified, show both
#         if not active and not stored:
#             active = stored = True

#         try:
#             output_lines = ["## 📋 ACP Sessions\n"]

#             # Show active sessions
#             if active:
#                 output_lines.append("### 🟢 Active Sessions")
#                 active_sessions = session.manager._sessions

#                 if not active_sessions:
#                     output_lines.append("*No active sessions*\n")
#                 else:
#                     for session_id, sess in active_sessions.items():
#                         agent_name = getattr(sess, "current_agent_name", "unknown")
#                         cwd = getattr(sess, "cwd", "unknown")
#                         msg_count = len(getattr(sess, "_conversation_history", []))

#                         output_lines.append(f"- **{session_id}**")
#                         output_lines.append(f"  - Agent: `{agent_name}`")
#                         output_lines.append(f"  - Directory: `{cwd}`")
#                         output_lines.append(f"  - Messages: {msg_count}")
#                     output_lines.append("")

#             # Show stored sessions
#             if stored and session.manager._persistent_manager:
#                 output_lines.append("### 💾 Stored Sessions")

#                 try:
#                     stored_sessions = (
#                         await session.manager._persistent_manager.store.list_sessions()
#                     )

#                     if not stored_sessions:
#                         output_lines.append("*No stored sessions*\n")
#                     else:
#                         for session_id in stored_sessions:
#                             store = session.manager._persistent_manager.store
#                             session_data = await store.load_session(session_id)
#                             if session_data:
#                                 msg_count = len(session_data.conversation)
#                                 created = session_data.metadata.get(
#                                     "created_at", "unknown"
#                                 )

#                                 output_lines.append(f"- **{session_id}**")
#                                 output_lines.append(
#                                     f"  - Agent: `{session_data.agent_name or 'unknown'}`"  # noqa: E501
#                                 )
#                                 output_lines.append(
#                                     f"  - Directory: `{session_data.cwd}`"
#                                 )
#                                 output_lines.append(f"  - Messages: {msg_count}")
#                                 output_lines.append(f"  - Created: {created}")
#                         output_lines.append("")
#                 except Exception as e:
#                     output_lines.append(f"*Error loading stored sessions: {e}*\n")

#             await ctx.output.print("\n".join(output_lines))

#         except Exception as e:
#             await ctx.output.print(f"❌ **Error listing sessions:** {e}")


# class LoadSessionCommand(SlashedCommand):
#     """Load a previous ACP session with conversation replay.

#     This command will:
#     1. Look up the session by ID
#     2. Replay the entire conversation history
#     3. Restore the session context (agent, working directory, MCP servers)
#     """

#     name = "load-session"
#     category = "acp"

#     async def execute_command(
#         self,
#         ctx: CommandContext[ACPCommandContext],
#         session_id: str,
#         *,
#         preview: bool = False,
#         no_replay: bool = False,
#     ):
#         """Load a previous ACP session.

#         Args:
#             ctx: Command context with ACP session
#             session_id: Session identifier to load
#             preview: Show session info without loading
#             no_replay: Load session without replaying conversation
#         """
#         session = ctx.context.session
#         if not session.manager:
#             await ctx.output.print("❌ **Session manager not available**")
#             return

#         try:
#             # Check if session exists
#             if session.manager._persistent_manager:
#                 session_data = (
#                     await session.manager._persistent_manager.load_session_data(
#                         session_id
#                     )
#                 )
#                 if not session_data:
#                     await ctx.output.print(f"❌ **Session not found:** `{session_id}`")
#                     return

#                 if preview:
#                     # Show session preview
#                     msg_count = len(session_data.conversation)
#                     created = session_data.metadata.get("created_at", "unknown")
#                     mcp_count = len(session_data.mcp_servers)

#                     preview_lines = [
#                         f"## 📋 Session Preview: `{session_id}`\n",
#                         f"**Agent:** `{session_data.agent_name or 'unknown'}`",
#                         f"**Directory:** `{session_data.cwd}`",
#                         f"**Messages:** {msg_count}",
#                         f"**MCP Servers:** {mcp_count}",
#                         f"**Created:** {created}",
#                     ]

#                     if session_data.metadata:
#                         metadata_json = json.dumps(session_data.metadata, indent=2)
#                         preview_lines.append(
#                             f"**Metadata:** ```json\n{metadata_json}\n```"
#                         )

#                     await ctx.output.print("\n".join(preview_lines))
#                     return

#                 # Actually load the session
#                 if no_replay:
#                     await ctx.output.print(
#                         f"🔄 **Loading session `{session_id}` without replay...**"
#                     )
#                     await ctx.output.print(
#                         f"✅ **Session `{session_id}` is available for loading**"
#                     )
#                 else:
#                     load_msg = f"🔄 **Loading session `{session_id}` with replay...**"
#                     await ctx.output.print(load_msg)

#                     msg_count = len(session_data.conversation)
#                     await ctx.output.print(f"📽️ **Replaying {msg_count} messages...**")
#                     await ctx.output.print(
#                         f"✅ **Session `{session_id}` loaded successfully**"
#                     )
#             else:
#                 await ctx.output.print("❌ **Session persistence not enabled**")

#         except Exception as e:
#             await ctx.output.print(f"❌ **Error loading session:** {e}")


# class SaveSessionCommand(SlashedCommand):
#     """Save the current ACP session to persistent storage.

#     This will save:
#     - Complete conversation history
#     - Current agent configuration
#     - Working directory and MCP server setup
#     - Session metadata

#     The session can later be loaded with /load-session.
#     """

#     name = "save-session"
#     category = "acp"

#     async def execute_command(
#         self,
#         ctx: CommandContext[ACPCommandContext],
#         *,
#         description: str | None = None,
#     ):
#         """Save the current ACP session.

#         Args:
#             ctx: Command context with ACP session
#             description: Optional description for the session
#         """
#         session = ctx.context.session

#         if not session.manager:
#             await ctx.output.print("❌ **Session manager not available**")
#             return

#         try:
#             if session.manager._persistent_manager:
#                 await session.manager._persistent_manager.save_session(session)

#                 msg_count = len(getattr(session, "_conversation_history", []))
#                 await ctx.output.print(
#                     f"💾 **Session `{session.session_id}` saved successfully**"
#                 )
#                 await ctx.output.print(f"📊 **Saved {msg_count} messages**")

#                 if description:
#                     await ctx.output.print(f"📝 **Description:** {description}")
#             else:
#                 await ctx.output.print("❌ **Session persistence not enabled**")

#         except Exception as e:
#             await ctx.output.print(f"❌ **Error saving session:** {e}")


# class DeleteSessionCommand(SlashedCommand):
#     """Delete a stored ACP session.

#     This permanently removes the session from storage.
#     Use with caution as this action cannot be undone.
#     """

#     name = "delete-session"
#     category = "acp"

#     async def execute_command(
#         self,
#         ctx: CommandContext[ACPCommandContext],
#         session_id: str,
#         *,
#         confirm: bool = False,
#     ):
#         """Delete a stored ACP session.

#         Args:
#             ctx: Command context with ACP session
#             session_id: Session identifier to delete
#             confirm: Skip confirmation prompt
#         """
#         session = ctx.context.session
#         if not session.manager:
#             await ctx.output.print("❌ **Session manager not available**")
#             return

#         try:
#             if session.manager._persistent_manager:
#                 # Check if session exists
#                 session_data = (
#                     await session.manager._persistent_manager.load_session_data(
#                         session_id
#                     )
#                 )
#                 if not session_data:
#                     await ctx.output.print(f"❌ **Session not found:** `{session_id}`")
#                     return

#                 if not confirm:
#                     msg_count = len(session_data.conversation)
#                     await ctx.output.print(
#                         f"⚠️  **About to delete session `{session_id}`**"
#                     )
#                     await ctx.output.print(
#                         f"📊 **This session has {msg_count} messages**"
#                     )
#                     await ctx.output.print(
#                         f"**To confirm, run:** `/delete-session {session_id} --confirm`"
#                     )
#                     return

#                 # Delete the session
#                 await session.manager._persistent_manager.store.delete_session(session_id)  # noqa: E501
#                 await ctx.output.print(
#                     f"🗑️  **Session `{session_id}` deleted successfully**"
#                 )
#             else:
#                 await ctx.output.print("❌ **Session persistence not enabled**")

#         except Exception as e:
#             await ctx.output.print(f"❌ **Error deleting session:** {e}")


def get_acp_commands() -> list[type[SlashedCommand]]:
    """Get all ACP-specific slash commands."""
    return [
        # ListSessionsCommand,
        # LoadSessionCommand,
        # SaveSessionCommand,
        # DeleteSessionCommand,
    ]
