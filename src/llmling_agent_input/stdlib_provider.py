"""Stdlib input provider."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any

from llmling import ToolError
from mcp import types

from llmling_agent.log import get_logger
from llmling_agent_input.base import InputProvider


if TYPE_CHECKING:
    from pydantic import BaseModel

    from llmling_agent.agent.context import AgentContext, ConfirmationResult
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.tools.base import Tool


logger = get_logger(__name__)


class StdlibInputProvider(InputProvider):
    """Input provider using only Python stdlib functionality."""

    async def get_text_input(
        self,
        context: AgentContext,
        prompt: str,
        message_history: list[ChatMessage] | None = None,
    ) -> str:
        return input(f"{prompt}\n> ")

    async def get_structured_input(
        self,
        context: AgentContext,
        prompt: str,
        output_type: type[BaseModel],
        message_history: list[ChatMessage] | None = None,
    ) -> BaseModel:
        """Get structured input, with promptantic and fallback handling."""
        if result := await _get_promptantic_result(output_type):
            return result

        # Fallback: Get raw input and validate
        prompt = f"{prompt}\n(Please provide response as {output_type.__name__})"
        raw_input = await self.get_input(context, prompt, message_history=message_history)
        try:
            return output_type.model_validate_json(raw_input)
        except Exception as e:
            msg = f"Invalid response format: {e}"
            raise ToolError(msg) from e

    async def get_tool_confirmation(
        self,
        context: AgentContext,
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage] | None = None,
    ) -> ConfirmationResult:
        import anyenv

        agent_name = context.node_name
        prompt = dedent(f"""
            Tool Execution Confirmation
            -------------------------
            Tool: {tool.name}
            Description: {tool.description or "No description"}
            Agent: {agent_name}

            Arguments:
            {anyenv.dump_json(args, indent=True)}

            Options:
            - y: allow execution
            - n/skip: skip this tool
            - abort: abort current run
            - quit: abort entire chain
            """).strip()

        response = input(f"{prompt}\nChoice [y/n/abort/quit]: ").lower()
        match response:
            case "y" | "yes":
                return "allow"
            case "abort":
                return "abort_run"
            case "quit":
                return "abort_chain"
            case _:
                return "skip"

    async def get_elicitation(
        self,
        context: AgentContext,
        params: types.ElicitRequestParams,
        message_history: list[ChatMessage] | None = None,
    ) -> types.ElicitResult | types.ErrorData:
        """Get user response to elicitation request using stdlib input."""
        try:
            print(f"\n{params.message}")

            # Handle structured input with schema
            print("Please provide response as JSON:")
            if params.requestedSchema:
                import anyenv

                schema_json = anyenv.dump_json(params.requestedSchema, indent=True)
                print(f"Expected schema:\n{schema_json}")
            response = input("> ")
            try:
                import anyenv

                content = anyenv.load_json(response, return_type=dict)
                return types.ElicitResult(action="accept", content=content)
            except anyenv.JsonLoadError as e:
                return types.ErrorData(
                    code=types.INVALID_REQUEST, message=f"Invalid JSON: {e}"
                )

        except KeyboardInterrupt:
            return types.ElicitResult(action="cancel")
        except Exception as e:  # noqa: BLE001
            return types.ErrorData(
                code=types.INVALID_REQUEST, message=f"Elicitation failed: {e}"
            )

    async def get_code_input(
        self,
        context: AgentContext,
        template: str | None = None,
        language: str = "python",
        description: str | None = None,
    ) -> str:
        msg = (
            "Multi-line code input not supported in stdlib provider. "
            "Use prompt-toolkit or textual provider instead."
        )
        raise NotImplementedError(msg)


async def _get_promptantic_result(output_type: type[BaseModel]) -> BaseModel | None:
    """Helper to get structured input via promptantic.

    Returns None if promptantic is not available or fails.
    """
    try:
        from promptantic import ModelGenerator

        return await ModelGenerator().apopulate(output_type)
    except ImportError:
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Promptantic failed: %s", e)
        return None
