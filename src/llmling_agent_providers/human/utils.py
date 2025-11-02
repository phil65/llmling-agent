"""Agent provider implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling import ToolError

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from pydantic import BaseModel


logger = get_logger(__name__)


async def get_structured_response(
    model_cls: type[BaseModel], use_promptantic: bool = True
) -> BaseModel:
    if use_promptantic:
        from promptantic import ModelGenerator, PromptanticError

        try:
            return await ModelGenerator().apopulate(model_cls)
        except PromptanticError as e:
            logger.exception("Failed to get structured input")
            error_msg = f"Invalid input: {e}"
            raise ToolError(error_msg) from e
        except KeyboardInterrupt:
            msg = "Input cancelled by user"
            raise ToolError(msg)  # noqa: B904
    else:
        # Regular text input
        print(f"(Please provide response as {model_cls.__name__})")
        response = input("> ")
        try:
            return model_cls.model_validate_json(response)
        except Exception as e:
            logger.exception("Failed to parse structured response")
            error_msg = f"Invalid response format: {e}"
            raise ToolError(error_msg) from e


def get_textual_streaming_app():
    from textual.app import App
    from textual.events import Key  # noqa: TC002
    from textual.widgets import Input

    class StreamingInputApp(App):
        def __init__(self, chunk_callback):
            super().__init__()
            self.chunk_callback = chunk_callback
            self.buffer = []
            self.done = False

        def compose(self):
            yield Input(id="input")

        def on_input_changed(self, event: Input.Changed):
            # New character was typed
            if len(event.value) > len(self.buffer):
                new_char = event.value[len(self.buffer) :]
                self.chunk_callback(new_char)
            self.buffer = list(event.value)

        def on_key(self, event: Key):
            if event.key == "enter":
                self.done = True
                self.result = "".join(self.buffer)
                self.exit()

    return StreamingInputApp
