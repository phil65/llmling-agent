"""Web interface for LLMling agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import gradio as gr
from gradio.themes import Base, Default, Glass, Monochrome, Soft
from llmling.config.store import ConfigStore
from upath import UPath

from llmling_agent.chat_session import ChatSessionManager
from llmling_agent_web.ui_state import UIState


THEMES = {
    "base": Base(),
    "soft": Soft(),
    "monochrome": Monochrome(),
    "glass": Glass(),
    "default": Default(),
}

CSS = """
.monospace {
    font-family: ui-monospace, "Cascadia Mono", "Segoe UI Mono",
                "Liberation Mono", Menlo, Monaco, Consolas, monospace;
}
"""


if TYPE_CHECKING:
    from gradio.routes import App


logger = logging.getLogger(__name__)


class AgentUI:
    """Main agent web interface."""

    def __init__(self, theme: str = "soft"):
        """Initialize interface."""
        store = ConfigStore("agents.json")
        self.available_files = [str(UPath(path)) for _, path in store.list_configs()]
        self.state = UIState()
        self.initial_status = "Please select a configuration file"
        self._session_manager = ChatSessionManager()
        self.theme = THEMES.get(theme.lower(), Soft())

    def create_ui(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(theme=self.theme, css=CSS) as app:
            gr.Markdown("# 🤖 LLMling Agent Chat")

            with gr.Row():
                with gr.Column(scale=1):
                    # Config file management
                    with gr.Group(visible=True):
                        upload_button = gr.UploadButton(
                            "📁 Upload Config",
                            file_types=[".yml", ".yaml"],
                            file_count="single",
                            interactive=True,
                        )
                        file_input = gr.Dropdown(
                            choices=self.available_files,
                            label="Agent Configuration File",
                            value=None,  # No default selection
                            interactive=True,
                            show_label=True,
                        )

                    # Agent selection - empty initially
                    agent_input = gr.Dropdown(
                        choices=[],
                        label="Select Agent",
                        interactive=True,
                        show_label=True,
                    )

                    status = gr.Markdown(
                        value=self.initial_status,
                        elem_classes=["status-msg"],
                    )

                    # Model override
                    model_input = gr.Textbox(
                        label="Model Override (optional)",
                        placeholder="e.g. openai:gpt-4",
                        interactive=True,
                        show_label=True,
                    )

                    # Tool management
                    tool_states = gr.Dataframe(
                        headers=["Tool", "Enabled"],
                        label="Available Tools",
                        interactive=True,
                        elem_classes=["monospace"],
                        visible=True,
                    )

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        value=[],
                        label="Chat",
                        height=600,
                        show_copy_button=True,
                        show_copy_all_button=True,
                        type="messages",
                        avatar_images=(None, None),
                        bubble_full_width=False,
                    )

                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="Type your message here...",
                            label="Message",
                            scale=8,
                            container=False,
                            interactive=True,
                        )
                        submit_btn = gr.Button(
                            "Send",
                            scale=1,
                            variant="primary",
                            interactive=True,
                        )

            with gr.Row():
                debug_toggle = gr.Checkbox(
                    label="Debug Mode",
                    value=False,
                    interactive=True,
                )
                debug_logs = gr.Markdown(
                    value=None,
                    visible=True,
                    elem_classes=["monospace"],
                )

            # Event handlers with proper async handling
            async def handle_upload(x: Any) -> list[Any]:
                result = await self.state.handle_upload(x)
                return result.to_updates([
                    file_input,
                    agent_input,
                    status,
                    debug_logs,
                    tool_states,
                ])

            async def handle_file_selection(file_path: str) -> list[Any]:
                result = await self.state.handle_file_selection(file_path)
                return result.to_updates([agent_input, status, debug_logs, tool_states])

            async def handle_agent_selection(*args: Any) -> list[Any]:
                result = await self.state.handle_agent_selection(*args)
                return result.to_updates([status, chatbot, debug_logs, tool_states])

            def handle_debug(x: bool) -> list[Any]:
                result = self.state.toggle_debug(x)
                return result.to_updates([debug_logs, status])

            async def handle_message(*args: Any) -> list[Any]:
                result = await self.state.send_message(*args)
                return result.to_updates([msg_box, chatbot, status, debug_logs])

            async def handle_tool_toggle(evt: gr.SelectData) -> list[Any]:
                if evt.index[1] == 1:  # Second column (Enabled)
                    tool_name = evt.row_value[0]  # First column contains tool name
                    new_state = not evt.value  # Toggle current state

                    result = await self.state.update_tool_states({tool_name: new_state})
                    return result.to_updates([status, tool_states, debug_logs])
                return [gr.update(), gr.update(), None]

            # Connect handlers to UI events
            upload_button.upload(
                fn=handle_upload,
                inputs=[upload_button],
                outputs=[file_input, agent_input, status, debug_logs, tool_states],
            )

            file_input.select(
                fn=handle_file_selection,
                inputs=[file_input],
                outputs=[agent_input, status, debug_logs, tool_states],
            )

            agent_input.select(
                fn=handle_agent_selection,
                inputs=[agent_input, model_input, chatbot],
                outputs=[status, chatbot, debug_logs, tool_states],
            )

            debug_toggle.change(
                fn=handle_debug,
                inputs=[debug_toggle],
                outputs=[debug_logs, status],
            )

            inputs = [msg_box, chatbot, agent_input, model_input]
            outputs = [msg_box, chatbot, status, debug_logs]
            msg_box.submit(fn=handle_message, inputs=inputs, outputs=outputs)
            submit_btn.click(fn=handle_message, inputs=inputs, outputs=outputs)

            tool_states.select(
                fn=handle_tool_toggle,
                outputs=[status, tool_states, debug_logs],
            )

        return app


def setup_logging() -> None:
    """Set up logging configuration."""
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, force=True, format=fmt)
    logging.getLogger("gradio").setLevel(logging.INFO)
    logging.getLogger("llmling_agent").setLevel(logging.DEBUG)
    logging.getLogger("llmling").setLevel(logging.DEBUG)


def create_app(theme: str = "soft") -> gr.Blocks:
    """Create the Gradio interface."""
    ui = AgentUI(theme=theme)
    return ui.create_ui()


def launch_app(
    *,
    theme: str = "soft",
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int | None = None,
    block: bool = True,
) -> tuple[App, str, str]:
    """Launch the LLMling web interface.

    This provides a user-friendly interface to:
    - Load agent configuration files
    - Select and configure agents
    - Chat with agents
    - View chat history and debug logs

    Args:
        theme: Interface theme (default: "soft")
        share: Whether to create a public URL
        server_name: Server hostname (default: "127.0.0.1")
        server_port: Optional server port number
        block: Whether to block the thread. Set to False when using programmatically.
    """
    import asyncio
    import platform

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    setup_logging()
    logger.info("Starting web interface")
    app = create_app(theme=theme)
    app.queue()
    return app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=not block,
    )


if __name__ == "__main__":
    launch_app()