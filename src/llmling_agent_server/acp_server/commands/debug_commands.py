"""Debug commands for ACP notification replay and testing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import anyenv
from pydantic import TypeAdapter
from slashed import CommandContext  # noqa: TC002

from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ContentToolCallContent,
    SessionNotification,
    SessionUpdate,
    TextContentBlock,
    ToolCallKind,  # noqa: TC001
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)
from llmling_agent.log import get_logger
from llmling_agent.messaging.context import NodeContext  # noqa: TC001
from llmling_agent_commands.base import NodeCommand
from llmling_agent_server.acp_server.session import ACPSession  # noqa: TC001


logger = get_logger(__name__)

# TypeAdapter for auto-constructing SessionUpdate variants from discriminator
SessionUpdateAdapter: TypeAdapter[SessionUpdate] = TypeAdapter(SessionUpdate)


class DebugSendTextCommand(NodeCommand):
    """Send a text chunk notification for debugging.

    Useful for testing client rendering of different message types.
    """

    name = "debug-send-text"
    category = "debug"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        text: str,
        *,
        chunk_type: str = "agent",
    ) -> None:
        """Send a text chunk notification.

        Args:
            ctx: Command context
            text: Text content to send
            chunk_type: Type of chunk ('agent', 'user', 'thought')
        """
        session = ctx.context.data
        assert session
        try:
            content = TextContentBlock(text=text)

            if chunk_type == "agent":
                update: SessionUpdate = AgentMessageChunk(content=content)
            elif chunk_type == "user":
                update = UserMessageChunk(content=content)
            elif chunk_type == "thought":
                update = AgentThoughtChunk(content=content)
            else:
                await ctx.print(f"❌ **Invalid chunk type:** `{chunk_type}`")
                return

            notification = SessionNotification(session_id=session.session_id, update=update)
            await session.client.session_update(notification)  # pyright: ignore[reportArgumentType]
            await ctx.print(f"✅ **Sent {chunk_type} text chunk:** {text[:50]}...")

        except Exception as e:
            logger.exception("Failed to send debug text chunk")
            await ctx.print(f"❌ **Failed to send text chunk:** {e}")


class DebugSendToolCallCommand(NodeCommand):
    """Send a tool call notification for debugging.

    Tests the client's tool call visualization and status handling.
    """

    name = "debug-send-tool-call"
    category = "debug"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        title: str,
        *,
        kind: ToolCallKind = "other",
    ) -> None:
        """Send a tool call notification.

        Args:
            ctx: Command context
            title: Tool call title/description
            kind: Tool kind ('read', 'edit', 'delete', 'move', 'search',
                  'execute', 'think', 'fetch', 'other')
        """
        session = ctx.context.data
        assert session
        try:
            id_ = f"debug-{hash(title)}"
            await session.notifications.tool_call_start(id_, title=title, kind=kind)
            await ctx.print(f"✅ **Sent tool call:** {title}")
        except Exception as e:
            logger.exception("Failed to send debug tool call")
            await ctx.print(f"❌ **Failed to send tool call:** {e}")


# class DebugUpdateToolCallCommand(SlashedCommand):
#     """Send a tool call update notification for debugging.

#     Tests tool call progress updates and result display.
#     """

#     name = "debug-update-tool"
#     category = "debug"

#     async def execute_command(
#         self,
#         ctx: CommandContext[AgentContext[ACPSession]],
#         tool_call_id: str,
#         *,
#         status: ToolCallStatus = "completed",
#         content: str = "",
#     ):
#         """Send a tool call update notification.

#         Args:
#             ctx: Command context
#             tool_call_id: ID of tool call to update
#             status: New status
#             content: Content to include in update
#         """
#         session = ctx.context.data
#         assert session
#         try:
#             tool_content = []
#             if content:
#                 tool_content = [
#                     ContentToolCallContent(content=TextContentBlock(text=content))
#                 ]
#             await session.notifications.tool_call_progress(
#                 tool_call_id,
#                 status,
#                 content=tool_content,
#             )
#             await ctx.print(f"✅ **Updated tool call {tool_call_id}:** {status}")

#         except Exception as e:
#             logger.exception("Failed to update debug tool call")
#             await ctx.print(f"❌ **Failed to update tool call:** {e}")


class DebugReplaySequenceCommand(NodeCommand):
    """Replay a sequence of ACP notifications from a JSON file.

    Allows testing complex interaction flows by replaying recorded sequences.
    """

    name = "debug-replay"
    category = "debug"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        file_path: str,
    ) -> None:
        """Replay a sequence of ACP notifications from a JSON file.

        Args:
            ctx: Command context
            file_path: Path to JSON file containing notification sequence
        """
        session = ctx.context.data
        assert session
        try:
            path = Path(file_path)
            if not path.exists():
                await ctx.print(f"❌ **File not found:** `{file_path}`")
                return

            with path.open() as f:
                sequence_data = json.load(f)

            if not isinstance(sequence_data, dict) or "notifications" not in sequence_data:
                await ctx.print("❌ **Invalid replay file.** Expected: `{'notifications': [...]}`")
                return

            notifications = sequence_data["notifications"]
            count = 0
            delay_ms = sequence_data.get("delay_ms", 0)

            for notification_data in notifications:
                try:
                    # Auto-construct the correct SessionUpdate type via discriminator
                    update = SessionUpdateAdapter.validate_python(notification_data)
                    notification = SessionNotification(session_id=session.session_id, update=update)
                    await session.client.session_update(notification)  # pyright: ignore[reportArgumentType]
                    count += 1
                    if delay_ms:
                        await asyncio.sleep(delay_ms / 1000)

                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to replay notification", error=e)
                    continue

            await ctx.print(f"✅ **Replayed {count} notifications from** `{file_path}`")

        except Exception as e:
            logger.exception("Failed to replay debug sequence")
            await ctx.print(f"❌ **Failed to replay sequence:** {e}")


class DebugSessionInfoCommand(NodeCommand):
    """Show current ACP session debugging information.

    Displays session state, client capabilities, and configuration details.
    """

    name = "debug-session-info"
    category = "debug"

    async def execute_command(self, ctx: CommandContext[NodeContext[ACPSession]]) -> None:
        """Show current ACP session debugging information."""
        session = ctx.context.data
        assert session
        try:
            info = {
                "session_id": session.session_id,
                "current_agent": session.current_agent_name,
                "available_agents": list(session.agent_pool.agents.keys()),
                "cwd": session.cwd,
                "client_capabilities": (
                    session.client_capabilities.model_dump()
                    if session.client_capabilities
                    else None
                ),
            }

            text = anyenv.dump_json(info, indent=True)
            await ctx.print(f"## 🔍 Session Debug Info\n\n```json\n{text}\n```")

        except Exception as e:
            logger.exception("Failed to get session info")
            await ctx.print(f"❌ **Failed to get session info:** {e}")


class DebugCreateTemplateCommand(NodeCommand):
    """Create a template JSON file for debugging notification sequences.

    Generates a sample replay file with common notification types.
    """

    name = "debug-create-template"
    category = "debug"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        *,
        file_path: str = "debug_replay_template.json",
    ) -> None:
        """Create a template JSON file for debugging notification sequences.

        Args:
            ctx: Command context
            file_path: Path where to create the template file
        """
        try:
            # Create proper BaseModel instances
            message_chunk = AgentMessageChunk.text(text="Hello, this is a debug message!")

            tool_start = ToolCallStart(
                tool_call_id="debug-tool-1",
                title="Debug Tool Call",
                status="in_progress",
                kind="other",
                content=None,
                locations=None,
            )

            tool_update = ToolCallProgress(
                tool_call_id="debug-tool-1",
                status="completed",
                content=[
                    ContentToolCallContent.text(text="Tool completed successfully!"),
                ],
                title="tool_call_update",
            )

            # Create notifications using proper SessionNotification models
            notifications = [
                SessionNotification(session_id="template", update=message_chunk),
                SessionNotification(session_id="template", update=tool_start),
                SessionNotification(session_id="template", update=tool_update),
            ]

            # Convert to JSON-serializable format
            template = {
                "description": "ACP notification replay sequence for debugging",
                "delay_ms": 100,
                "notifications": [notif.model_dump()["update"] for notif in notifications],
            }

            with Path(file_path).open("w") as f:
                json.dump(template, f, indent=2)

            await ctx.print(f"✅ **Created replay template:** `{file_path}`")

        except Exception as e:
            logger.exception("Failed to create replay template")
            await ctx.print(f"❌ **Failed to create template:** {e}")


class DebugSendRawCommand(NodeCommand):
    """Send a raw ACP notification from JSON string.

    For advanced debugging - send arbitrary notification structures.
    """

    name = "debug-send-raw"
    category = "debug"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        notification_json: str,
    ) -> None:
        """Send a raw ACP notification from JSON string.

        Args:
            ctx: Command context
            notification_json: JSON string of the notification to send
        """
        session = ctx.context.data
        assert session
        try:
            data = anyenv.load_json(notification_json, return_type=dict)

            # Validate it has the expected structure
            if "update" not in data:
                msg = "❌ **Notification JSON must contain 'update' field**"
                await ctx.print(msg)
                return

            notification = SessionNotification(session_id=session.session_id, **data)
            await session.client.session_update(notification)
            await ctx.print("✅ **Sent raw notification**")
        except json.JSONDecodeError as e:
            await ctx.print(f"❌ **Invalid JSON:** {e}")
        except Exception as e:
            logger.exception("Failed to send raw notification")
            await ctx.print(f"❌ **Failed to send raw notification:** {e}")


def get_debug_commands() -> list[type[NodeCommand]]:
    """Get all ACP debug commands."""
    return [
        DebugSendTextCommand,
        DebugSendToolCallCommand,
        # DebugUpdateToolCallCommand,
        DebugReplaySequenceCommand,
        DebugSessionInfoCommand,
        DebugCreateTemplateCommand,
        DebugSendRawCommand,
    ]
