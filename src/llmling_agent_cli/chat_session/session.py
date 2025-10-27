"""Interactive chat session implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import logfire
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console
from rich.markdown import Markdown
from slashed import DefaultOutputWriter, ExitCommandError
from slashed.prompt_toolkit_completer import PromptToolkitCompleter

from llmling_agent import ChatMessage
from llmling_agent_cli.chat_session.base import AgentPoolView
from llmling_agent_cli.chat_session.formatting import MessageFormatter
from llmling_agent_cli.chat_session.history import SessionHistory
from llmling_agent_cli.chat_session.welcome import create_welcome_messages


if TYPE_CHECKING:
    from llmling_agent import Agent, AgentPool, AnyAgent, Tool, ToolCallInfo
    from llmling_agent.agent.conversation import ConversationManager


class InteractiveSession:
    """Interactive chat session using prompt_toolkit."""

    def __init__(
        self,
        agent: AnyAgent[Any, Any],
        *,
        pool: AgentPool | None = None,
        show_log_in_chat: bool = False,
        stream: bool = False,
        render_markdown: bool = False,
    ):
        """Initialize interactive session.

        Args:
            agent: The LLMling agent to use
            pool: Optional agent pool for multi-agent interactions
            show_log_in_chat: Whether to show logs in chat
            stream: Whether to use streaming mode
            render_markdown: Whether to render markdown in responses
        """
        self.agent = agent
        self._stream = stream
        self._render_markdown = render_markdown

        # Setup console and formatting
        self.console = Console()
        self.formatter = MessageFormatter(self.console)

        # Setup session management
        self._chat_session: AgentPoolView | None = None
        self._prompt: PromptSession | None = None
        self.pool = pool
        self._completer: PromptToolkitCompleter | None = None

    def _connect_signals(self):
        """Connect to chat session signals."""
        assert self._chat_session is not None
        self._chat_session.history_cleared.connect(self._on_history_cleared)
        self.agent.agent_reset.connect(self._on_session_reset)
        self.agent.tools.events.added.connect(self._on_tool_added)
        self.agent.tools.events.removed.connect(self._on_tool_removed)
        self.agent.tools.events.changed.connect(self._on_tool_changed)
        self.agent.connections.node_connected.connect(self._on_agent_connected)
        if self.pool:
            # Connect to all agents in pool (including main agent)
            for agent in self.pool.agents.values():
                agent.message_sent.connect(self._on_message)
                agent.tool_used.connect(self._on_tool_call)
        else:
            # Only if no pool, connect directly to main agent
            self.agent.tool_used.connect(self._on_tool_call)

    def _on_agent_connected(self, agent: Agent[Any]):
        """Handle newly connected agent."""
        agent.message_sent.connect(self._on_message)
        agent.tool_used.connect(self._on_tool_call)

    @logfire.instrument("Received message in UI from agent {message.name}")
    def _on_message(self, message: ChatMessage):
        """Handle messages from any agent."""
        # Format with agent name if it's not the main agent
        if message.name and message.name != self.agent.name:
            self.formatter.print_message_start(message)
            self.formatter.print_message_content(message.content)
            self.formatter.print_message_end(message)

    def _on_tool_added(self, name: str, tool: Tool):
        """Handle tool addition."""
        self.console.print(f"\nTool added: {tool.name}")

    def _on_tool_removed(self, tool_name: str):
        """Handle tool removal."""
        self.console.print(f"\nTool removed: {tool_name}")

    @logfire.instrument("Received tool call in UI from agent {tool_call.agent_name}")
    def _on_tool_call(self, tool_call: ToolCallInfo):
        """Handle tool usage signal."""
        self.formatter.print_tool_call(tool_call)

    def _on_tool_changed(self, name: str, tool: Tool):
        """Handle tool state changes."""
        state = "enabled" if tool.enabled else "disabled"
        self.console.print(f"\nTool {name!r} {state}")

    def _on_history_cleared(self, event: ConversationManager.HistoryCleared):
        """Handle history cleared event."""
        self.console.print("\nChat history cleared")

    def _on_session_reset(self, event: Agent.AgentReset):
        """Handle session reset event."""
        self.console.print("\nSession reset.")

    def _setup_prompt(self):
        """Setup prompt toolkit session."""
        session = self._chat_session
        assert session is not None

        history = SessionHistory(session)
        self._completer = PromptToolkitCompleter[AgentPoolView](session.commands, session)

        self._prompt = PromptSession[str](
            "You: ",
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=self._completer,
        )

    async def _handle_input(self, content: str):
        """Handle user input."""
        if not content.strip():
            return

        writer = DefaultOutputWriter()
        try:
            assert self._chat_session is not None

            if content.startswith("/"):
                try:
                    result = await self._chat_session.send_message(content, output=writer)
                    if result.content:
                        self.formatter.print_message_start(result)
                        self.formatter.print_message_content(result.content)
                        self.formatter.print_message_end(result)
                except ExitCommandError:
                    self.formatter.print_exit()
                    raise EOFError  # noqa: B904
                return

            # Show user message
            user_msg: ChatMessage[str] = ChatMessage(content=content, role="user")
            self.formatter.print_message_start(user_msg)
            self.formatter.print_message_content(content)
            self.formatter.print_message_end(user_msg)

            if self._stream:
                buffer = ""
                first_chunk = True
                async for chunk in await self._chat_session.send_message(
                    content,
                    stream=True,
                    output=writer,
                ):
                    if chunk.content:
                        if first_chunk:
                            self.formatter.print_message_start(chunk)
                            first_chunk = False

                        new_content = chunk.content
                        if new_content != buffer:
                            diff = new_content[len(buffer) :]
                            self.formatter.print_message_content(diff, end="")
                            buffer = new_content

                self.console.print()  # New line after streaming
                self.formatter.print_message_end(chunk)
            else:
                # Non-streaming mode
                result = await self._chat_session.send_message(content, output=writer)
                if result.content:
                    self.formatter.print_message_start(result)
                    content_to_print = (
                        Markdown(result.content)
                        if self._render_markdown
                        else result.content
                    )
                    self.formatter.print_message_content(content_to_print)
                    self.formatter.print_message_end(result)
        except (httpx.ReadError, GeneratorExit):
            self.formatter.print_connection_error()
        except EOFError:
            raise
        except Exception as e:  # noqa: BLE001
            self.formatter.print_error(e, show_traceback=True)

    async def start(self):
        """Start interactive session."""
        try:
            self._chat_session = AgentPoolView(self.agent, pool=self.pool)
            self._connect_signals()
            self._setup_prompt()

            # Show welcome message
            welcome_info = await create_welcome_messages(
                self._chat_session._agent, streaming=self._stream, rich_format=True
            )
            self.formatter.print_welcome(welcome_info)

            # Main interaction loop
            while True:
                try:
                    assert self._prompt
                    if user_input := await self._prompt.prompt_async():
                        await self._handle_input(user_input)
                except KeyboardInterrupt:
                    self.console.print("\nUse /exit to quit")
                    continue
                except EOFError:
                    break
                except Exception as e:  # noqa: BLE001
                    self.formatter.print_error(e, show_traceback=True)
                    continue

        except Exception as e:  # noqa: BLE001
            self.formatter.print_error(e, show_traceback=True)
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Clean up resources."""
        if self._chat_session:
            await self._chat_session.cleanup()


# Helper function for CLI
async def start_interactive_session(
    agent: AnyAgent[Any, Any],
    *,
    stream: bool = False,
):
    """Start an interactive chat session."""
    session = InteractiveSession(agent, pool=agent.context.pool, stream=stream)
    await session.start()
