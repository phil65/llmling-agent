"""Web interface commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from pydantic import ValidationError
import typer
from upathtools import to_upath

from llmling_agent_cli.cli_types import Provider  # noqa: TC001


def create(
    output: Annotated[
        str | None,
        typer.Option(
            "-o",
            "--output",
            help="Output file path. If not provided, only displays the config.",
        ),
    ] = None,
    add_to_store: Annotated[
        bool, typer.Option("-a", "--add-to-store", help="Add generated config to ConfigStore")
    ] = False,
    model: Annotated[
        str, typer.Option("-m", "--model", help="Model to use for generation")
    ] = "gpt-5",
    provider: Annotated[
        Provider, typer.Option("-p", "--provider", help="Provider to use")
    ] = "pydantic_ai",
) -> None:
    """Interactive config generator for agents and teams."""
    from schemez import YAMLCode
    from textual.app import App
    from textual.binding import Binding
    from textual.containers import ScrollableContainer
    from textual.widgets import Header, Input, Static

    from llmling_agent import Agent, AgentsManifest
    from llmling_agent.agents.architect import create_architect_agent
    from llmling_agent.utils.count_tokens import count_tokens
    from llmling_agent_cli import agent_store

    if TYPE_CHECKING:
        from textual.app import ComposeResult

    class StatsDisplay(Static):
        """Display for token count and validation status."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, markup=kwargs.pop("markup", False), **kwargs)

        def update_stats(self, token_count: int, status: str | None = None) -> None:
            """Update the stats display."""
            text = f"Context tokens: {token_count:,}"
            if status:
                text = f"{status} | {text}"
            self.update(text)

    class YamlDisplay(ScrollableContainer):
        """Display for YAML content with syntax highlighting."""

        def __init__(self) -> None:
            super().__init__()
            self._content = Static("")

        def compose(self) -> ComposeResult:
            """Initial empty content."""
            yield self._content

        def update_yaml(self, content: str) -> None:
            """Update the YAML content with syntax highlighting."""
            from rich.syntax import Syntax

            syntax = Syntax(content, "yaml", theme="monokai")
            self._content.update(syntax)

    class ConfigGeneratorApp(App[None]):
        """Application for generating configuration files."""

        CSS = """
        Screen {
            layout: grid;
            grid-size: 1;
            padding: 1;
        }

        Input {
            dock: top;
            margin: 1 0;
        }

        YamlDisplay {
            height: 1fr;
            border: solid green;
        }

        StatsDisplay {
            dock: bottom;
            height: 3;
            content-align: center middle;
        }
        """

        BINDINGS: ClassVar = [
            Binding("ctrl+s", "save", "Save Config", show=True),
            ("escape", "quit", "Quit"),
        ]

        def __init__(
            self,
            model: str = "openai:gpt-5-mini",
            output_path: str | None = None,
            add_to_store: bool = False,
        ) -> None:
            super().__init__()
            agent = Agent(output_type=YAMLCode)
            self.agent = agent
            self.current_config: str | None = None
            self.output_path = to_upath(output_path) if output_path else None
            self.add_to_store = add_to_store
            self._token_count: int = 0

        def compose(self) -> ComposeResult:
            yield Header()
            yield Input(placeholder="Describe your configuration needs...")
            yield YamlDisplay()
            yield StatsDisplay("Context tokens: calculating...")

        async def on_mount(self) -> None:
            """Load schema and calculate token count."""
            self.agent = await create_architect_agent(model="openai:gpt-5-nano")
            assert self.agent.model_name
            model_name = self.agent.model_name.split(":")[-1]
            context = await self.agent.conversation.format_history()
            self._token_count = count_tokens(context, model_name)
            stats = self.query_one(StatsDisplay)
            stats.update_stats(self._token_count)

        async def on_input_submitted(self, message: Input.Submitted) -> None:
            """Generate config when user hits enter."""
            from yamling import YAMLError

            yaml = await self.agent.run(message.value)
            self.current_config = yaml.content.code
            try:
                AgentsManifest.from_yaml(yaml.content.code)
                status = "✓ Valid configuration"
            except (ValidationError, YAMLError) as e:
                status = f"✗ Invalid: {e}"

            content = self.query_one(YamlDisplay)
            content.update_yaml(yaml.content.code)
            stats = self.query_one(StatsDisplay)
            stats.update_stats(self._token_count, status)

        def action_save(self) -> None:
            """Save current config."""
            if not self.current_config:
                self.notify("No configuration generated yet!")
                return

            if not self.output_path:
                self.notify("No output path specified!")
                return

            self.output_path.write_text(self.current_config)
            if self.add_to_store:
                agent_store.add_config(self.output_path.stem, str(self.output_path))
            self.notify(f"Saved to {self.output_path}")

    app = ConfigGeneratorApp(model=model, provider=provider)  # type: ignore
    app.run()
