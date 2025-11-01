"""Message stream widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.containers import ScrollableContainer
from textual.widgets import Static

from llmling_agent.log import get_logger
from llmling_textual.widgets.message_flow import MessageFlowWidget


if TYPE_CHECKING:
    from llmling_agent import AgentPool
    from llmling_agent.talk.talk import Talk

logger = get_logger(__name__)


class MessageStream(ScrollableContainer):
    """Displays global message flows from the agent pool."""

    DEFAULT_CSS = """
    MessageStream {
        width: 100%;
        height: 100%;
        border: round $background;
        background: $surface;
        overflow-y: scroll;
        padding: 1;
    }

    MessageStream > .placeholder {
        color: $text-muted;
        text-align: center;
        padding: 2;
    }
    """

    def __init__(
        self,
        pool: AgentPool[Any],
        *,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ):
        super().__init__(id=id, classes=classes)
        self.pool = pool
        self._placeholder = Static("No messages yet...", classes="placeholder")
        self.show_placeholder = True
        self._connected = False

    def on_mount(self):
        """Connect to pool's message flow."""
        if not self._connected:
            self.pool.connection_registry.message_flow.connect(self.handle_message_flow)
            self._connected = True
            logger.debug("Connected to global message flow")

        # Show initial placeholder
        self.mount(self._placeholder)

    def cleanup(self):
        """Cleanup signal connections."""
        if self._connected:
            self.pool.connection_registry.message_flow.disconnect(
                self.handle_message_flow
            )
            self._connected = False

    def handle_message_flow(self, event: Talk.ConnectionProcessed):
        """Handle new message flow event."""
        if not self.is_mounted:
            return

        logger.debug("Received message flow event: %r", event)

        # Remove placeholder if needed
        if self.show_placeholder and self._placeholder in self.children:
            self._placeholder.remove()
            self.show_placeholder = False

        # Create and mount new message widget
        widget = MessageFlowWidget(event)
        self.mount(widget)
        widget.scroll_visible()

        # Keep reasonable buffer size
        if len(self.children) > 1000:  # noqa: PLR2004
            self.children[0].remove()


if __name__ == "__main__":
    from textualicious import show

    from llmling_agent import AgentPool

    pool = AgentPool()
    show(MessageStream(pool))
