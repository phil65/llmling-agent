from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from llmling_agent.chat_session import ChatSessionManager
from llmling_agent.cli.chat_session.base import CommandContext
from llmling_agent.cli.chat_session.config import SessionState, get_history_file


if TYPE_CHECKING:
    from llmling_agent import LLMlingAgent
    from llmling_agent.chat_session.base import AgentChatSession
    from llmling_agent.cli.chat_session.base import Command


logger = logging.getLogger(__name__)


class InteractiveSession:
    """Interactive chat session using prompt_toolkit."""

    def __init__(
        self,
        agent: LLMlingAgent[str],
        *,
        debug: bool = False,
    ) -> None:
        """Initialize interactive session."""
        self.agent = agent
        self.debug = debug
        self.console = Console()

        # Internal state
        self._session_manager = ChatSessionManager()
        self._chat_session: AgentChatSession | None = None
        self._state = SessionState()
        self._commands: dict[str, Command] = {}

        # Setup components
        self._setup_history()
        self._setup_prompt()
        self._register_commands()

    @property
    def chat_session(self) -> AgentChatSession:
        """Get current chat session."""
        if self._chat_session is None:
            msg = "Chat session not initialized"
            raise RuntimeError(msg)
        return self._chat_session

    def _setup_history(self) -> None:
        """Setup command history."""
        history_file = get_history_file(self.agent.name)
        self._history = FileHistory(str(history_file))

    def _setup_prompt(self) -> None:
        """Setup prompt toolkit session."""
        self._prompt = PromptSession(
            history=self._history,
            auto_suggest=AutoSuggestFromHistory(),
            message="You: ",
        )

    def _register_commands(self) -> None:
        """Register available commands."""
        from .commands.builtin import get_builtin_commands

        for command in get_builtin_commands():
            self._commands[command.name] = command

    async def start(self) -> None:
        """Start interactive session."""
        try:
            # Initialize chat session
            self._chat_session = await self._session_manager.create_session(self.agent)
            await self._show_welcome()

            while True:
                try:
                    user_input = await self._prompt.prompt_async()
                    if not user_input:
                        continue

                    if user_input.startswith("/"):
                        await self._handle_command(user_input[1:])
                    else:
                        await self._handle_message(user_input)

                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break

        finally:
            await self._cleanup()

    async def _handle_command(self, input_: str) -> None:
        """Handle command input."""
        cmd, *args = input_.split(maxsplit=1)
        arg = args[0] if args else ""

        command = self._commands.get(cmd)
        if not command:
            self.console.print(f"Unknown command: {cmd}", style="red")
            return

        try:
            context = CommandContext(
                console=self.console,
                session=self.chat_session,
                state=self._state,
                args=arg,
            )
            await command.execute(context)
            self._state.last_command = cmd
        except EOFError:
            raise  # Re-raise EOFError to be caught by main loop
        except Exception as e:  # noqa: BLE001
            self.console.print(f"Error executing command: {e}", style="red")
            if self.debug:
                import traceback

                self.console.print(traceback.format_exc())

    async def _handle_message(self, message: str) -> None:
        """Handle chat message."""
        try:
            self.console.print("\nAssistant:", style="bold blue")
            with Live("", console=self.console) as live:
                response_parts = []
                async for chunk in await self.chat_session.send_message(  # Using property
                    message,
                    stream=True,
                ):
                    response_parts.append(chunk.content)
                    live.update(Markdown("".join(response_parts)))

            # Update session state
            self._state.message_count += 2
            # TODO: Update token count when available

        except Exception as e:  # noqa: BLE001
            self.console.print(f"\nError: {e}", style="red")
            if self.debug:
                import traceback

                self.console.print(traceback.format_exc())

    async def _show_welcome(self) -> None:
        """Show welcome message."""
        self.console.print(f"\nStarted chat with {self.agent.name}")
        self.console.print("Type /help for commands or /exit to quit\n")

        # Show initial state
        tools = self.chat_session.get_tool_states()
        enabled = sum(1 for enabled in tools.values() if enabled)
        self.console.print(
            f"Available tools: {len(tools)} ({enabled} enabled)",
            style="dim",
        )

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._chat_session:
            # Any cleanup needed for chat session
            pass

        # Show session summary
        if self._state.message_count > 0:
            self.console.print("\nSession Summary:")
            self.console.print(f"Messages: {self._state.message_count}")
            self.console.print(f"Total tokens: {self._state.total_tokens:,}")


# Helper function for CLI
async def start_interactive_session(
    agent: LLMlingAgent[str],
    *,
    debug: bool = False,
) -> None:
    """Start an interactive chat session."""
    session = InteractiveSession(agent, debug=debug)
    await session.start()
