"""Textual main screen."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.screen import Screen
from textual.widgets import Footer, Header

from llmling_textual.screens.chat_screen import ChatScreen
from llmling_textual.widgets.agent_list import NodeList
from llmling_textual.widgets.message_stream import MessageStream


if TYPE_CHECKING:
    from textual.app import ComposeResult

    from llmling_agent import AgentPool
    from llmling_textual.widgets.agent_list import NodeEntry


class MainScreen(Screen):
    """Main application screen."""

    DEFAULT_CSS = """
    MainScreen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 4fr;
        padding: 1;
    }

    #node-list {
        height: 100%;
    }

    #message-stream {
        height: 100%;
    }
    """

    def __init__(self, pool: AgentPool):
        super().__init__()
        self.pool = pool
        self.node_list = NodeList(id="node-list")
        self.message_stream = MessageStream(pool)

        # Connect pool events
        self.pool.node_events.added.connect(self._on_node_change)
        self.pool.node_events.removed.connect(self._on_node_change)

    def on_mount(self):
        """Initialize screen."""
        # Initial node list population
        self.node_list.update_nodes(self.pool)

    def compose(self) -> ComposeResult:
        """Create main layout."""
        yield Header()
        yield self.node_list
        yield self.message_stream
        yield Footer()

    def _on_node_change(self, *_):
        """Handle node added/removed."""
        if self.node_list:  # Check if widget still exists
            self.node_list.update_nodes(self.pool)

    def cleanup(self):
        """Disconnect event handlers."""
        # Disconnect pool events
        self.pool.node_events.added.disconnect(self._on_node_change)
        self.pool.node_events.removed.disconnect(self._on_node_change)

        # Clean up stream
        self.message_stream.cleanup()

    async def on_node_entry_clicked(self, event: NodeEntry.Clicked):
        """Show chat screen for clicked agent."""
        await self.app.push_screen(ChatScreen(event.node))


if __name__ == "__main__":
    from textualicious import show

    from llmling_agent import AgentPool

    pool = AgentPool()
    show(MainScreen(pool))
