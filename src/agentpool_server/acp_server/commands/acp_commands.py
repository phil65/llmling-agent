"""ACP-specific slash commands for session management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import CommandContext  # noqa: TC002

from agentpool.messaging.context import NodeContext  # noqa: TC001
from agentpool_commands.base import NodeCommand
from agentpool_config.session import SessionQuery
from agentpool_server.acp_server.session import ACPSession  # noqa: TC001


if TYPE_CHECKING:
    from pydantic_ai import ModelRequest, ModelResponse


class ListSessionsCommand(NodeCommand):
    """List all available ACP sessions.

    Shows:
    - Session ID and status (active/stored)
    - Agent name and working directory
    - Creation time and last activity

    Options:
      --active    Show only active sessions
      --stored    Show only stored sessions
    """

    name = "list-sessions"
    category = "acp"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        *,
        active: bool = False,
        stored: bool = False,
    ) -> None:
        """List available ACP sessions.

        Args:
            ctx: Command context with ACP session
            active: Show only active sessions
            stored: Show only stored sessions
        """
        session = ctx.context.data
        if not session:
            raise RuntimeError("Session not available in command context")

        if not session.manager:
            await ctx.output.print("❌ **Session manager not available**")
            return

        # If no filter specified, show both
        if not active and not stored:
            active = stored = True

        try:
            output_lines = ["## 📋 ACP Sessions\n"]

            # Show active sessions
            if active:
                output_lines.append("### 🟢 Active Sessions")
                active_sessions = session.manager._active

                if not active_sessions:
                    output_lines.append("*No active sessions*\n")
                else:
                    for session_id, sess in active_sessions.items():
                        agent_name = sess.current_agent_name
                        cwd = sess.cwd or "unknown"
                        is_current = session_id == session.session_id

                        # Get title from SessionData
                        session_data = await session.manager.session_manager.store.load(session_id)
                        title = session_data.title if session_data else None

                        status = " *(current)*" if is_current else ""
                        title_text = f": {title}" if title else ""
                        output_lines.append(f"- **{session_id}**{status}{title_text}")
                        output_lines.append(f"  - Agent: `{agent_name}`")
                        output_lines.append(f"  - Directory: `{cwd}`")
                    output_lines.append("")

            # Show stored sessions
            if stored:
                output_lines.append("### 💾 Stored Sessions")

                try:
                    stored_session_ids = await session.manager.session_manager.store.list_sessions()
                    # Filter out active ones if we already showed them
                    if active:
                        stored_session_ids = [
                            sid for sid in stored_session_ids if sid not in session.manager._active
                        ]

                    if not stored_session_ids:
                        output_lines.append("*No stored sessions*\n")
                    else:
                        for session_id in stored_session_ids:
                            session_data = await session.manager.session_manager.store.load(
                                session_id
                            )
                            if session_data:
                                title_text = f": {session_data.title}" if session_data.title else ""
                                output_lines.append(f"- **{session_id}**{title_text}")
                                output_lines.append(f"  - Agent: `{session_data.agent_name}`")
                                output_lines.append(
                                    f"  - Directory: `{session_data.cwd or 'unknown'}`"
                                )
                                output_lines.append(
                                    f"  - Last active: {session_data.last_active.strftime('%Y-%m-%d %H:%M')}"  # noqa: E501
                                )
                        output_lines.append("")
                except Exception as e:  # noqa: BLE001
                    output_lines.append(f"*Error loading stored sessions: {e}*\n")

            await ctx.output.print("\n".join(output_lines))

        except Exception as e:  # noqa: BLE001
            await ctx.output.print(f"❌ **Error listing sessions:** {e}")


class LoadSessionCommand(NodeCommand):
    """Load a previous ACP session with conversation replay.

    This command will:
    1. Look up the session by ID
    2. Replay the conversation history via ACP notifications
    3. Restore the session context (agent, working directory)

    Options:
      --preview     Show session info without loading
      --no-replay   Load session without replaying conversation

    Examples:
      /load-session sess_abc123def456
      /load-session sess_abc123def456 --preview
      /load-session sess_abc123def456 --no-replay
    """

    name = "load-session"
    category = "acp"

    async def execute_command(  # noqa: PLR0915
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        session_id: str,
        *,
        preview: bool = False,
        no_replay: bool = False,
    ) -> None:
        """Load a previous ACP session.

        Args:
            ctx: Command context with ACP session
            session_id: Session identifier to load
            preview: Show session info without loading
            no_replay: Load session without replaying conversation
        """
        session = ctx.context.data
        if not session:
            raise RuntimeError("Session not available in command context")

        if not session.manager:
            await ctx.output.print("❌ **Session manager not available**")
            return

        try:
            # Load session data from storage
            session_data = await session.manager.session_manager.store.load(session_id)

            if not session_data:
                await ctx.output.print(f"❌ **Session not found:** `{session_id}`")
                return

            # Get conversation history from storage
            storage = session.agent_pool.storage
            messages = []
            if storage:
                query = SessionQuery(name=session_data.conversation_id)
                messages = await storage.filter_messages(query)

            if preview:
                # Show session preview without loading
                preview_lines = [
                    f"## 📋 Session Preview: `{session_id}`\n",
                ]

                if session_data.title:
                    preview_lines.append(f"**Title:** {session_data.title}")

                preview_lines.extend([
                    f"**Agent:** `{session_data.agent_name}`",
                    f"**Directory:** `{session_data.cwd or 'unknown'}`",
                    f"**Created:** {session_data.created_at.strftime('%Y-%m-%d %H:%M')}",
                    f"**Last active:** {session_data.last_active.strftime('%Y-%m-%d %H:%M')}",
                    f"**Conversation ID:** `{session_data.conversation_id}`",
                    f"**Messages:** {len(messages)}",
                ])

                if session_data.metadata:
                    preview_lines.append(
                        f"**Protocol:** {session_data.metadata.get('protocol', 'unknown')}"
                    )

                await ctx.output.print("\n".join(preview_lines))
                return

            # Actually load the session
            await ctx.output.print(f"🔄 **Loading session `{session_id}`...**")

            # Switch to the session's agent if different
            if session_data.agent_name != session.current_agent_name:
                if session_data.agent_name in session.agent_pool.all_agents:
                    await session.switch_active_agent(session_data.agent_name)
                    await ctx.output.print(f"📌 **Switched to agent:** `{session_data.agent_name}`")
                else:
                    await ctx.output.print(
                        f"⚠️ **Agent `{session_data.agent_name}` not found, keeping current agent**"
                    )

            # Update working directory if specified
            if session_data.cwd and session_data.cwd != session.cwd:
                session.cwd = session_data.cwd
                await ctx.output.print(f"📂 **Working directory:** `{session_data.cwd}`")

            # Replay conversation history unless disabled
            if not no_replay and messages:
                await ctx.output.print(f"📽️ **Replaying {len(messages)} messages...**")

                # Extract ModelRequest/ModelResponse from ChatMessage.messages field

                model_messages: list[ModelRequest | ModelResponse] = []
                for chat_msg in messages:
                    if chat_msg.messages:
                        model_messages.extend(chat_msg.messages)

                if model_messages:
                    # Use ACPNotifications.replay() which handles all content types properly
                    try:
                        await session.notifications.replay(model_messages)
                        await ctx.output.print(
                            f"✅ **Replayed {len(model_messages)} model messages**"
                        )
                    except Exception as e:  # noqa: BLE001
                        session.log.warning("Failed to replay conversation history", error=str(e))
                        await ctx.output.print(f"⚠️ **Failed to replay messages:** {e}")
                else:
                    await ctx.output.print("📭 **No model messages to replay**")
            elif no_replay:
                await ctx.output.print("⏭️ **Skipped conversation replay**")
            else:
                await ctx.output.print("📭 **No conversation history to replay**")

            await ctx.output.print(f"✅ **Session `{session_id}` loaded successfully**")

        except Exception as e:  # noqa: BLE001
            await ctx.output.print(f"❌ **Error loading session:** {e}")


class SaveSessionCommand(NodeCommand):
    """Save the current ACP session to persistent storage.

    This will save:
    - Current agent configuration
    - Working directory
    - Session metadata

    Note: Conversation history is automatically saved if storage is enabled.

    Options:
      --description "text"   Optional description for the session

    Examples:
      /save-session
      /save-session --description "Working on feature X"
    """

    name = "save-session"
    category = "acp"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        *,
        description: str | None = None,
    ) -> None:
        """Save the current ACP session.

        Args:
            ctx: Command context with ACP session
            description: Optional description for the session
        """
        session = ctx.context.data
        if not session:
            raise RuntimeError("Session not available in command context")

        if not session.manager:
            await ctx.output.print("❌ **Session manager not available**")
            return

        try:
            # Load current session data
            session_data = await session.manager.session_manager.store.load(session.session_id)

            if session_data:
                # Update metadata if description provided
                if description:
                    session_data = session_data.with_metadata(description=description)

                # Touch to update last_active
                session_data.touch()

                # Save back
                await session.manager.session_manager.save(session_data)

                await ctx.output.print(f"💾 **Session `{session.session_id}` saved successfully**")
                if description:
                    await ctx.output.print(f"📝 **Description:** {description}")
            else:
                await ctx.output.print(f"⚠️ **Session `{session.session_id}` not found in storage**")

        except Exception as e:  # noqa: BLE001
            await ctx.output.print(f"❌ **Error saving session:** {e}")


class DeleteSessionCommand(NodeCommand):
    """Delete a stored ACP session.

    This permanently removes the session from storage.
    Use with caution as this action cannot be undone.

    Options:
      --confirm   Skip confirmation prompt

    Examples:
      /delete-session sess_abc123def456
      /delete-session sess_abc123def456 --confirm
    """

    name = "delete-session"
    category = "acp"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        session_id: str,
        *,
        confirm: bool = False,
    ) -> None:
        """Delete a stored ACP session.

        Args:
            ctx: Command context with ACP session
            session_id: Session identifier to delete
            confirm: Skip confirmation prompt
        """
        session = ctx.context.data
        if not session:
            raise RuntimeError("Session not available in command context")

        if not session.manager:
            await ctx.output.print("❌ **Session manager not available**")
            return

        # Prevent deleting current session
        if session_id == session.session_id:
            await ctx.output.print("❌ **Cannot delete the current active session**")
            return

        try:
            # Check if session exists
            session_data = await session.manager.session_manager.store.load(session_id)

            if not session_data:
                await ctx.output.print(f"❌ **Session not found:** `{session_id}`")
                return

            if not confirm:
                await ctx.output.print(f"⚠️  **About to delete session `{session_id}`**")
                await ctx.output.print(f"📌 **Agent:** `{session_data.agent_name}`")
                await ctx.output.print(
                    f"📅 **Last active:** {session_data.last_active.strftime('%Y-%m-%d %H:%M')}"
                )
                await ctx.output.print(
                    f"**To confirm, run:** `/delete-session {session_id} --confirm`"
                )
                return

            # Delete the session
            deleted = await session.manager.session_manager.store.delete(session_id)

            if deleted:
                await ctx.output.print(f"🗑️  **Session `{session_id}` deleted successfully**")
            else:
                await ctx.output.print(f"⚠️ **Failed to delete session `{session_id}`**")

        except Exception as e:  # noqa: BLE001
            await ctx.output.print(f"❌ **Error deleting session:** {e}")


class ListPoolsCommand(NodeCommand):
    """List available agent pool configurations.

    Shows:
    - Stored configurations from ConfigStore (name -> path mapping)
    - Currently active pool configuration
    - Available agents in the current pool

    Examples:
      /list-pools
    """

    name = "list-pools"
    category = "acp"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
    ) -> None:
        """List available pool configurations.

        Args:
            ctx: Command context with ACP session
        """
        from agentpool_cli import agent_store

        session = ctx.context.data
        if not session:
            raise RuntimeError("Session not available in command context")

        try:
            output_lines = ["## 🏊 Agent Pool Configurations\n"]

            # Show current pool info
            output_lines.append("### 📍 Current Pool")
            current_config = (
                session.acp_agent.server.config_path if session.acp_agent.server else None
            )
            if current_config:
                output_lines.append(f"**Config:** `{current_config}`")
            else:
                output_lines.append("**Config:** *(default/built-in)*")

            # Show agents in current pool
            agent_names = list(session.agent_pool.all_agents.keys())
            output_lines.append(f"**Agents:** {', '.join(f'`{n}`' for n in agent_names)}")
            output_lines.append(f"**Active agent:** `{session.current_agent_name}`")
            output_lines.append("")

            # Show stored configurations
            output_lines.append("### 💾 Stored Configurations")
            stored_configs = agent_store.list_configs()
            active_config = agent_store.get_active()

            if not stored_configs:
                output_lines.append("*No stored configurations*")
                output_lines.append("")
                output_lines.append("Use `agentpool add <name> <path>` to add configurations.")
            else:
                for name, path in stored_configs:
                    is_active = active_config and active_config.name == name
                    is_current = current_config and path == current_config
                    markers = []
                    if is_active:
                        markers.append("default")
                    if is_current:
                        markers.append("current")
                    marker_text = f" *({', '.join(markers)})*" if markers else ""
                    output_lines.append(f"- **{name}**{marker_text}")
                    output_lines.append(f"  `{path}`")

            output_lines.append("")
            output_lines.append("*Use `/set-pool <name>` or `/set-pool <path>` to switch pools.*")

            await ctx.output.print("\n".join(output_lines))

        except Exception as e:  # noqa: BLE001
            await ctx.output.print(f"❌ **Error listing pools:** {e}")


class SetPoolCommand(NodeCommand):
    """Switch to a different agent pool configuration.

    This command will:
    1. Close all active sessions
    2. Load the new pool configuration
    3. Initialize the new pool with all agents

    The configuration can be specified as:
    - A stored config name (from `agentpool add`)
    - A direct path to a configuration file

    Options:
      --agent <name>   Specify which agent to use as default

    Examples:
      /set-pool prod
      /set-pool /path/to/agents.yml
      /set-pool dev --agent=coder
    """

    name = "set-pool"
    category = "acp"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        config: str,
        *,
        agent: str | None = None,
    ) -> None:
        """Switch to a different agent pool.

        Args:
            ctx: Command context with ACP session
            config: Config name (from store) or path to config file
            agent: Optional specific agent to use as default
        """
        from pathlib import Path

        from agentpool_cli import agent_store

        session = ctx.context.data
        if not session:
            raise RuntimeError("Session not available in command context")

        if not session.acp_agent.server:
            await ctx.output.print("❌ **Server reference not available - cannot switch pools**")
            return

        try:
            # Resolve config path
            config_path: str | None = None
            config_name: str | None = None

            # First try as stored config name
            try:
                config_path = agent_store.get_config(config)
                config_name = config
            except KeyError:
                # Not a stored config, try as direct path
                path = Path(config)
                if path.exists() and path.is_file():
                    config_path = str(path.resolve())
                else:
                    await ctx.output.print(f"❌ **Config not found:** `{config}`")
                    await ctx.output.print("Provide a stored config name or a valid file path.")
                    return

            # Show what we're doing
            if config_name:
                await ctx.output.print(f"🔄 **Switching pool to `{config_name}`...**")
            else:
                await ctx.output.print(f"🔄 **Switching pool to `{config_path}`...**")

            # Perform the swap
            agent_names = await session.acp_agent.swap_pool(config_path, agent)

            # Report success
            await ctx.output.print("✅ **Pool switched successfully**")
            await ctx.output.print(f"**Agents:** {', '.join(f'`{n}`' for n in agent_names)}")
            if agent:
                await ctx.output.print(f"**Default agent:** `{agent}`")
            else:
                await ctx.output.print(f"**Default agent:** `{agent_names[0]}`")

            await ctx.output.print("")
            await ctx.output.print("*Note: A new session will be created on your next message.*")

        except FileNotFoundError as e:
            await ctx.output.print(f"❌ **Config file not found:** {e}")
        except ValueError as e:
            await ctx.output.print(f"❌ **Invalid configuration:** {e}")
        except Exception as e:  # noqa: BLE001
            await ctx.output.print(f"❌ **Error switching pool:** {e}")


def get_acp_commands() -> list[type[NodeCommand]]:
    """Get all ACP-specific slash commands."""
    return [
        ListSessionsCommand,
        LoadSessionCommand,
        SaveSessionCommand,
        DeleteSessionCommand,
        ListPoolsCommand,
        SetPoolCommand,
    ]
