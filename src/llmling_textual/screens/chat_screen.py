"""Chat screen."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

from platformdirs import user_data_dir
from slashed import CommandStore, ExitCommandError, OutputWriter
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header

from llmling_agent import Agent, ChatMessage, StructuredAgent
from llmling_agent_commands import get_commands
from llmling_textual.widgets.agent_is_typing import ResponseStatus
from llmling_textual.widgets.chat_view import ChatView
from llmling_textual.widgets.prompt_input import PromptInput


if TYPE_CHECKING:
    from textual.app import ComposeResult

    from llmling_agent import MessageNode


class TextualOutputWriter(OutputWriter):
    """Output writer that writes to Textual widgets."""

    def __init__(self, chat_view: ChatView):
        self.chat_view = chat_view

    async def print(self, text: str):
        """Write text to chat view."""
        msg = ChatMessage(content=text, role="system", name="System")
        await self.chat_view.append_chat_message(msg)

    async def print_error(self, text: str):
        """Write error to chat view."""
        msg = ChatMessage(content=f"Error: {text}", role="system", name="System")
        await self.chat_view.append_chat_message(msg)


class ChatScreen(ModalScreen[None]):
    """Modal screen for chatting with a specific node."""

    BINDINGS: ClassVar = [
        Binding("escape", "app.pop_screen", "Close", show=True),
        Binding("ctrl+j,ctrl+enter", "send_message", "Send", show=True),
    ]

    DEFAULT_CSS = """
    ChatScreen {
        align: center middle;
    }

    #chat-container {
        width: 90%;
        height: 90%;
        background: $surface;
        border: thick $background;
    }

    #chat-view {
        height: 1fr;
        border: heavy $background;
        margin: 1;
    }

    ResponseStatus {
        height: auto;
        margin: 1;
        background: $surface-darken-2;
    }

    #prompt-input {
        height: 20%;
        dock: bottom;
        border: heavy $background;
        margin: 1;
    }
    """

    def __init__(self, node: MessageNode[Any, Any]):
        super().__init__()
        self.node = node
        self._typing_status = ResponseStatus()
        self._typing_status.display = False

        # Initialize command system
        history_dir = pathlib.Path(user_data_dir("llmling", "llmling")) / "cli_history"
        file_path = history_dir / f"{node.name}.history"
        self.commands = CommandStore(history_file=file_path, enable_system_commands=True)
        self.commands._initialize_sync()
        for cmd in get_commands():
            self.commands.register_command(cmd)

    def compose(self) -> ComposeResult:
        """Create screen layout."""
        yield Header(name=f"Chat with {self.node.name}")

        with Vertical(id="chat-container"):
            yield ChatView(id="chat-view")
            yield self._typing_status
            yield PromptInput(id="prompt-input")

        yield Footer()

    async def on_mount(self):
        """Load conversation history when mounted."""
        chat_view = self.query_one(ChatView)
        # Load existing messages
        if isinstance(self.node, Agent | StructuredAgent):
            for msg in self.node.conversation.chat_messages:
                await chat_view.append_chat_message(msg)
        else:
            messages = await self.node._logger.get_message_history()
            for msg in messages:
                await chat_view.append_chat_message(msg)
        # Focus input
        self.query_one(PromptInput).focus()

    async def handle_command(self, command_str: str) -> ChatMessage[str]:
        """Handle a slash command."""
        chat_view = self.query_one(ChatView)
        writer = TextualOutputWriter(chat_view)

        try:
            ctx = self.commands.create_context(self.node.context, output_writer=writer)
            await self.commands.execute_command(command_str, ctx)
            return ChatMessage(content="", role="system")

        except ExitCommandError:
            # Handle exit command specially
            self.app.pop_screen()
            return ChatMessage(content="Session ended", role="system")

        except Exception as e:  # noqa: BLE001
            return ChatMessage(
                content=f"Command error: {e}", role="system", name="System"
            )

    async def on_prompt_input_prompt_submitted(self, event: PromptInput.PromptSubmitted):
        """Handle submitted prompt."""
        message = event.text.strip()
        if not message:
            self.notify("Cannot send empty message", severity="error")
            return

        chat_view = self.query_one(ChatView)
        user_msg = ChatMessage(content=message, role="user", name="You")
        await chat_view.append_chat_message(user_msg)

        async def process_response():
            try:
                self._typing_status.display = True
                self._typing_status.set_node_responding()
                response = await self.node.run(message)
                await chat_view.append_chat_message(response)
            except Exception as e:  # noqa: BLE001
                # Show error in chat
                error_msg = ChatMessage(
                    content=f"Error: {e}",
                    role="assistant",
                    name=self.node.name,
                )
                await chat_view.append_chat_message(error_msg)

            finally:
                # Always hide typing status
                self._typing_status.display = False

        # Run in background to keep UI responsive
        self.app.run_worker(process_response(), name=f"agent_response_{self.node.name}")


if __name__ == "__main__":
    from textualicious import show

    from llmling_agent import Agent

    agent = Agent()
    show(ChatScreen(agent))
