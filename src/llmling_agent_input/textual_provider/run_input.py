"""Textual based run call input."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


if TYPE_CHECKING:
    from textual.app import ComposeResult


class InputModal(ModalScreen[str]):
    """Modal for basic input requests."""

    DEFAULT_CSS = """
    InputModal {
        align: center middle;
    }

    .modal-container {
        width: 60%;
        height: auto;
        border: heavy $primary;
        padding: 1;
    }

    #prompt {
        margin: 1;
        text-align: center;
    }

    #input {
        margin: 1;
    }

    .buttons {
        width: 100%;
        height: auto;
        align-horizontal: right;
        margin-top: 1;
    }
    """
    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "submit", "Submit"),
    ]

    def __init__(self, prompt: str, output_type: type | None = None):
        super().__init__()
        self.prompt = prompt
        self.output_type = output_type

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label(self.prompt, id="prompt")
            if self.output_type:
                yield Label(f"(Please provide response as {self.output_type.__name__})")
            yield Input(id="input")
            with Horizontal(classes="buttons"):
                yield Button("Submit", variant="primary", id="submit")
                yield Button("Cancel", variant="error", id="cancel")

    def action_cancel(self):
        """Handle cancel action."""
        self.dismiss(None)

    def action_submit(self):
        """Handle submit action."""
        input_value = self.query_one(Input).value
        self.dismiss(input_value)

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "submit":
            self.action_submit()
        else:
            self.action_cancel()

    def on_input_submitted(self, event: Input.Submitted):
        """Handle Enter key in input."""
        self.action_submit()
