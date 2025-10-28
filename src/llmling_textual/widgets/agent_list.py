"""Agent list widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.message import Message
from textual.widgets import Static

from llmling_agent import Agent
from llmling_agent.delegation.base_team import BaseTeam


if TYPE_CHECKING:
    from textual.app import ComposeResult

    from llmling_agent import AgentPool, MessageNode


class NodeEntry(Static, can_focus=True):  # type: ignore
    """Individual node entry."""

    DEFAULT_CSS = """
    NodeEntry {
        height: 3;
        padding: 0 1;
        background: $surface;
        border: tall $primary;
        margin: 0 0 0 0;
    }

    NodeEntry:hover {
        background: $accent;
    }

    NodeEntry:focus {
        background: $accent;
        border: tall $accent;
    }

    NodeEntry.busy {
        color: $warning;
    }

    NodeEntry > Horizontal {
        height: 1;
        width: 100%;
    }

    NodeEntry .status {
        width: 2;
        color: $text-muted;
    }

    NodeEntry .name {
        width: auto;
        margin-right: 1;
    }

    NodeEntry .provider {
        color: $text-muted;
    }
    """

    class Clicked(Message):
        """Emitted when entry is clicked."""

        def __init__(self, node: MessageNode[Any, Any]):
            self.node = node
            super().__init__()

    def __init__(self, node: MessageNode[Any, Any]):
        super().__init__("")
        self.node = node
        self.add_class("busy" if node.task_manager.is_busy() else "idle")

    def compose(self) -> ComposeResult:
        """Create entry layout."""
        with Horizontal():
            yield Static(
                "●" if self.node.task_manager.is_busy() else "○", classes="status"
            )
            yield Static(self.node.name, classes="name")
            match self.node:
                case Agent():
                    exra = self.node.name
                case BaseTeam():
                    exra = " | ".join(node.name for node in self.node.agents)
                case _:
                    msg = f"Unknown node type: {type(self.node)}"
                    raise ValueError(msg)
            yield Static(f"({exra})", classes="provider")

    def on_click(self):
        """Handle click event."""
        self.post_message(self.Clicked(self.node))


class NodeList(ScrollableContainer):
    """List of available nodes."""

    BINDINGS: ClassVar = [
        Binding("up", "cursor_up", "Previous node", show=False),
        Binding("down", "cursor_down", "Next node", show=False),
        Binding("enter", "select_node", "Select node", show=False),
    ]

    DEFAULT_CSS = """
    NodeList {
        width: 100%;
        height: 100%;
        background: $surface-darken-1;
        padding: 0;
    }

    NodeList > .header {
        background: $panel;
        padding: 1;
        height: 3;
        border-bottom: solid $panel-darken-2;
    }
    """

    def __init__(
        self,
        *,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ):
        super().__init__(id=id, classes=classes)
        self._entries: dict[str, NodeEntry] = {}

    def compose(self) -> ComposeResult:
        """Create initial layout."""
        yield Static("Available Nodes", classes="header")

    def update_nodes(self, pool: AgentPool):
        """Update node list from pool."""
        current_nodes = set(pool.nodes.keys())
        existing_nodes = set(self._entries.keys())

        # Remove old entries
        for name in existing_nodes - current_nodes:
            if entry := self._entries.pop(name, None):
                entry.remove()

        # Add new entries
        for name in current_nodes - existing_nodes:
            node = pool.nodes[name]
            entry = NodeEntry(node)
            self._entries[name] = entry
            self.mount(entry)

        # Update existing entries
        for name in current_nodes & existing_nodes:
            node = pool.nodes[name]
            entry = self._entries[name]
            entry.add_class("busy" if node.task_manager.is_busy() else "idle")

    def action_cursor_up(self):
        """Move focus to previous node."""
        entries = list(self.query(NodeEntry))
        if not entries:
            return

        focused = self.screen.focused
        if isinstance(focused, NodeEntry):
            try:
                current_idx = entries.index(focused)
                new_idx = (current_idx - 1) % len(entries)
                entries[new_idx].focus()
            except ValueError:
                entries[-1].focus()
        else:
            entries[0].focus()

    def action_cursor_down(self):
        """Move focus to next node."""
        entries = list(self.query(NodeEntry))
        if not entries:
            return

        focused = self.screen.focused
        if isinstance(focused, NodeEntry):
            try:
                current_idx = entries.index(focused)
                new_idx = (current_idx + 1) % len(entries)
                entries[new_idx].focus()
            except ValueError:
                entries[0].focus()
        else:
            entries[0].focus()

    def action_select_node(self):
        """Select focused node."""
        if (focused := self.screen.focused) and isinstance(focused, NodeEntry):
            self.post_message(focused.Clicked(focused.node))


if __name__ == "__main__":
    from textualicious import show

    show(NodeList())
