from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import httpx
from rich.console import Console
from rich.style import Style
from rich.traceback import Traceback

from llmling_agent_cli.chat_session.exceptions import ChatSessionConfigError


if TYPE_CHECKING:
    from rich.markdown import Markdown

    from llmling_agent import ChatMessage, ToolCallInfo
    from llmling_agent_cli.chat_session.welcome import WelcomeInfo


class MessageFormatter:
    """Format chat messages and related content for CLI display."""

    LINE_WIDTH: ClassVar[int] = 80
    USER_STYLE: ClassVar[Style] = Style(color="blue")
    ASSISTANT_STYLE: ClassVar[Style] = Style(color="green")
    SYSTEM_STYLE: ClassVar[Style] = Style(color="yellow")
    STATS_STYLE: ClassVar[Style] = Style(dim=True)
    ERROR_STYLE: ClassVar[Style] = Style(color="red", bold=True)
    TOOL_STYLE: ClassVar[Style] = Style(color="yellow")

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def print_message_start(self, message: ChatMessage):
        """Print message header."""
        self.console.print()
        sender = self._get_sender_name(message)
        line = f"─── {sender} " + "─" * (self.LINE_WIDTH - len(sender) - 4)
        style = self._get_style(message.role)
        self.console.print(line, style=style)

    def print_message_content(self, content: str | Markdown, end: str = ""):
        """Print message content."""
        self.console.print(content, end=end, soft_wrap=True)

    def print_message_end(self, message: ChatMessage | None = None):
        """Print message footer with stats."""
        self.console.print()

        if message:  # Only show stats if we have a message
            # Collect stats parts
            parts = []

            # Model info
            if message.model_name:
                parts.append(f"Model: {message.model_name}")

            # Token and cost info from cost_info
            if message.cost_info:
                token_usage = message.cost_info.token_usage
                parts.append(f"Tokens: {token_usage['total']:,}")
                parts.append(f"Cost: ${message.cost_info.total_cost:.4f}")

            # Response time
            if message.response_time:
                parts.append(f"Time: {message.response_time:.2f}s")

            # Tool information from metadata
            if message.tool_calls:
                tool_infos = []
                for call in message.tool_calls:
                    tool_info = f"Tool: {call.tool_name}"
                    if call.args:
                        tool_info += f" (args: {call.args})"
                    tool_infos.append(tool_info)
                parts.extend(tool_infos)

            if parts:
                stats_line = " • ".join(parts)
                self.console.print(stats_line, style=self.STATS_STYLE)

        self.console.print("─" * self.LINE_WIDTH)

    def print_error(self, error: Exception, show_traceback: bool = False):
        """Print error message with optional traceback."""
        error_msg = format_error(error)
        self.console.print(f"\n[red bold]Error:[/] {error_msg}")
        if show_traceback:
            self.console.print("\n[dim]Debug traceback:[/]")
            trace = Traceback.from_exception(
                type(error),
                error,
                error.__traceback__,
                show_locals=True,
                width=self.LINE_WIDTH,
            )
            self.console.print(trace)

    def print_tool_call(self, tool_call: ToolCallInfo):
        """Print tool call information."""
        self.console.print()
        self.console.print("Tool Call:", style=self.TOOL_STYLE)
        self.console.print(f"  Name: {tool_call.tool_name}")
        self.console.print(f"  Args: {tool_call.args}")
        self.console.print(f"  Result: {tool_call.result}")

    def print_welcome(self, welcome_info: WelcomeInfo):
        """Print welcome message sections."""
        for title, lines in welcome_info.all_sections():
            if title:  # Skip empty section titles
                self.console.print(f"\n[bold]{title}[/]")
            for line in lines:
                self.console.print(line)

    def print_session_summary(
        self, messages: int, tokens: dict[str, int], cost: float, duration: str
    ):
        """Print end of session summary."""
        self.console.print("\nSession Summary:")
        self.console.print(f"Messages: {messages}")
        token_info = (
            f"Total tokens: {tokens['total']:,} "
            f"(Prompt: {tokens['prompt']:,}, "
            f"Completion: {tokens['completion']:,})"
        )
        self.console.print(token_info)
        self.console.print(f"Total cost: ${cost:.6f}")
        self.console.print(f"Duration: {duration}")

    def print_connection_error(self):
        """Print connection error message."""
        self.console.print("\nConnection interrupted.", style=self.ERROR_STYLE)

    def print_exit(self):
        """Print exit message."""
        self.console.print("\nGoodbye!")

    def print_chain_status(self, agent_names: list[str], status: str):
        """Print chain processing status."""
        self.console.print(f"\n[dim]Chain ({', '.join(agent_names)}): {status}[/]")

    def _get_sender_name(self, message: ChatMessage) -> str:
        """Get display name for message sender."""
        match message.role:
            case "user":
                return "You"
            case "assistant":
                return message.name or "Assistant"
            case "system":
                return "System"
            case _:
                return message.role.title()

    def _get_style(self, role: str) -> Style:
        """Get style for message role."""
        match role:
            case "user":
                return self.USER_STYLE
            case "assistant":
                return self.ASSISTANT_STYLE
            case "system":
                return self.SYSTEM_STYLE
            case _:
                return Style()


def format_error(error: Exception) -> str:
    """Format error message for display."""
    # Known error types we want to handle specially
    match error:
        case ChatSessionConfigError():
            return f"Chat session error: {error}"
        case ValueError() if "token" in str(error):
            return "Connection interrupted"
        case httpx.ReadError():
            return "Connection lost. Please try again."
        case GeneratorExit():
            return "Response stream interrupted"
        case _:
            return f"Error: {error}"
