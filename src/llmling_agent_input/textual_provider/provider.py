"""Textual input provider."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any

from textual.app import App

from llmling_agent_input.base import InputProvider
from llmling_agent_input.textual_provider.code_input import CodeInputModal
from llmling_agent_input.textual_provider.confirmation_input import ConfirmationModal
from llmling_agent_input.textual_provider.run_input import InputModal


if TYPE_CHECKING:
    from mcp import types
    from textual.screen import ModalScreen

    from llmling_agent.agent.context import AgentContext, ConfirmationResult
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.tools.base import Tool


class UserCancelledError(Exception):
    """Raised when user cancels input."""


class BaseInputApp(App[str]):
    """Base app for standalone input."""

    def __init__(self, input_screen: ModalScreen[str]):
        super().__init__()
        self._input_screen = input_screen
        self._result: str | None = None

    async def on_mount(self):
        self._result = await self.push_screen_wait(self._input_screen)  # type: ignore
        self.exit()


class InputApp(BaseInputApp):
    """Standalone app for text input."""

    def __init__(self, prompt: str, output_type: type | None = None):
        super().__init__(InputModal(prompt, output_type))


class CodeInputApp(BaseInputApp):
    """Standalone app for code input."""

    def __init__(
        self,
        template: str | None = None,
        language: str = "python",
        description: str | None = None,
    ):
        super().__init__(CodeInputModal(template, language, description))


class ConfirmationApp(BaseInputApp):
    """Standalone app for confirmation."""

    def __init__(self, prompt: str):
        super().__init__(ConfirmationModal(prompt))


class TextualInputProvider(InputProvider):
    """Input provider using Textual modals or standalone app."""

    def __init__(self, app: App | None = None):
        super().__init__()
        self.app = app

    async def get_input(
        self,
        context: AgentContext,
        prompt: str,
        output_type: type | None = None,
        message_history: list[ChatMessage] | None = None,
    ) -> Any:
        if self.app:
            result = await self.app.push_screen_wait(InputModal(prompt, output_type))
            if result is None:
                msg = "Input cancelled"
                raise UserCancelledError(msg)
            return result
        # Standalone mode - create temporary app
        app = InputApp(prompt, output_type)
        app_result = await app.run_async()
        if app_result is None:
            msg = "Input cancelled"
            raise UserCancelledError(msg)
        return app_result

    async def get_code_input(
        self,
        context: AgentContext,
        template: str | None = None,
        language: str = "python",
        description: str | None = None,
    ) -> str:
        if self.app:
            modal = CodeInputModal(template, language=language, description=description)
            result = await self.app.push_screen_wait(modal)
            if result is None:
                msg = "Code input cancelled"
                raise UserCancelledError(msg)
            return result
        app = CodeInputApp(template, language, description)
        app_result = await app.run_async()
        if app_result is None:
            msg = "Code input cancelled"
            raise UserCancelledError(msg)
        return app_result

    async def get_tool_confirmation(
        self,
        context: AgentContext,
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage] | None = None,
    ) -> ConfirmationResult:
        import anyenv

        prompt = dedent(f"""
            Tool Execution Confirmation
            -------------------------
            Tool: {tool.name}
            Description: {tool.description or "No description"}
            Agent: {context.node_name}

            Arguments:
            {anyenv.dump_json(args, indent=True)}
        """).strip()

        if self.app:
            result = await self.app.push_screen_wait(ConfirmationModal(prompt))
            return result or "skip"  # type: ignore
        app = ConfirmationApp(prompt)
        app_result = await app.run_async()
        return app_result or "skip"  # type: ignore

    async def get_elicitation(
        self,
        context: AgentContext,
        params: types.ElicitRequestParams,
        message_history: list[ChatMessage] | None = None,
    ) -> types.ElicitResult | types.ErrorData:
        """Get user response to elicitation request using Textual UI."""
        import anyenv
        from mcp import types

        try:
            prompt = f"{params.message}\nPlease provide response as JSON:"

            if self.app:
                result: str | None = await self.app.push_screen_wait(
                    InputModal(prompt, None)  # type: ignore
                )
                if result is None:
                    return types.ElicitResult(action="cancel")
            else:
                app = InputApp(prompt, None)
                result = await app.run_async()
                if result is None:
                    return types.ElicitResult(action="cancel")

            # Parse JSON response
            try:
                content = anyenv.load_json(result, return_type=dict)
                return types.ElicitResult(action="accept", content=content)
            except anyenv.JsonLoadError as e:
                return types.ErrorData(
                    code=types.INVALID_REQUEST, message=f"Invalid JSON: {e}"
                )

        except UserCancelledError:
            return types.ElicitResult(action="cancel")
        except Exception as e:  # noqa: BLE001
            return types.ErrorData(
                code=types.INVALID_REQUEST, message=f"Elicitation failed: {e}"
            )
