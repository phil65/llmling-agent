"""Pool list widget."""

from __future__ import annotations

from llmling.config.store import ConfigFile, ConfigStore
from textual.containers import ScrollableContainer
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static
from upathtools import read_path

from llmling_agent.log import get_logger


logger = get_logger(__name__)


class PoolEntry(Static):
    """Individual pool entry in the list."""

    DEFAULT_CSS = """
    PoolEntry {
        height: 3;
        padding: 1;
        background: $surface;
        border: tall $primary;
        margin: 0 0 1 0;
    }

    PoolEntry:hover {
        background: $accent;
    }

    PoolEntry:focus {
        background: $accent;
        border: tall $accent;
    }

    PoolEntry.active {
        border: tall $success;
    }

    PoolEntry .name {
        text-style: bold;
    }

    PoolEntry .path {
        color: $text-muted;
    }
    """

    active = reactive(False)

    class Selected(Message):
        """Emitted when entry is selected."""

        def __init__(self, config: ConfigFile):
            self.config = config
            super().__init__()

    def __init__(self, config: ConfigFile, active: bool = False):
        super().__init__("")
        self.config = config
        self.active = active

    def render(self) -> str:
        """Render the pool entry."""
        active = "✓ " if self.active else "  "
        return f"{active}[b]{self.config.name}[/b]\n   [dim]{self.config.path}[/dim]"

    def on_click(self):
        """Handle click event."""
        self.post_message(self.Selected(self.config))


class PoolList(ScrollableContainer):
    """List of available pools."""

    DEFAULT_CSS = """
    PoolList {
        width: 1fr;
        height: 100%;
        border: round $background;
        background: $surface-darken-1;
        padding: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.store = ConfigStore("agents.json")

    def on_mount(self):
        """Load pools on mount."""
        active = self.store.get_active()
        for name, path in self.store.list_configs():
            config = ConfigFile(name=name, path=path)
            entry = PoolEntry(
                config, active=(config.name == active.name) if active else False
            )
            self.mount(entry)


class PoolPreview(Static):
    """Preview of pool configuration."""

    DEFAULT_CSS = """
    PoolPreview {
        width: 2fr;
        height: 100%;
        border: round $background;
        background: $surface;
        padding: 1;
        overflow: auto scroll;  # Add scrolling
    }
    """

    def __init__(self):
        super().__init__("")
        self._current_path: str | None = None

    async def show_config(self, path: str | None):
        """Show configuration content."""
        from rich.syntax import Syntax

        if not path:
            self.update("No pool selected")
            return

        if path == self._current_path:
            return

        try:
            content = await read_path(path)
            self.update(Syntax(content, "yaml", theme="monokai"))
            self._current_path = path
        except Exception:
            logger.exception("Failed to load config", path=path)
            self.update(f"Error loading {path}")


if __name__ == "__main__":
    from textualicious import show

    show(PoolList())
