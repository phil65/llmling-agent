"""Stdlib input provider."""

from __future__ import annotations

import sys
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import anyenv
from mcp import types

from llmling_agent.log import get_logger
from llmling_agent.tools.exceptions import ToolError
from llmling_agent.ui.base import InputProvider


if TYPE_CHECKING:
    from pydantic import BaseModel

    from llmling_agent.agents.context import ConfirmationResult
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.tools.base import Tool


logger = get_logger(__name__)


class StdlibInputProvider(InputProvider):
    """Input provider using only Python stdlib functionality."""

    async def get_text_input(
        self,
        context: NodeContext,
        prompt: str,
        message_history: list[ChatMessage[Any]] | None = None,
    ) -> str:
        print(f"{prompt}\n> ", end="", file=sys.stderr, flush=True)
        return input()

    async def get_structured_input(
        self,
        context: NodeContext,
        prompt: str,
        output_type: type[BaseModel],
        message_history: list[ChatMessage[Any]] | None = None,
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
        context: NodeContext,
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage[Any]] | None = None,
    ) -> ConfirmationResult:
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

        print(f"{prompt}\nChoice [y/n/abort/quit]: ", end="", file=sys.stderr, flush=True)
        response = input().lower()
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
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        """Get user response to elicitation request using stdlib input."""
        try:
            print(f"\n{params.message}", file=sys.stderr)

            # URL mode: prompt user to open external URL
            if isinstance(params, types.ElicitRequestURLParams):
                print(f"URL: {params.url}", file=sys.stderr)
                print("Open this URL? [y/n]: ", end="", file=sys.stderr, flush=True)
                response = input().strip().lower()
                action = (
                    "accept"
                    if response in ("y", "yes")
                    else ("decline" if response in ("n", "no") else "cancel")
                )
                return types.ElicitResult(action=action)

            # Form mode: collect structured JSON input
            print("Please provide response as JSON:", file=sys.stderr)
            if params.requestedSchema:
                schema_json = anyenv.dump_json(params.requestedSchema, indent=True)
                print(f"Expected schema:\n{schema_json}", file=sys.stderr)
            print("> ", end="", file=sys.stderr, flush=True)
            response = input()
            try:
                content = anyenv.load_json(response, return_type=dict)
                return types.ElicitResult(action="accept", content=content)
            except anyenv.JsonLoadError as e:
                return types.ErrorData(code=types.INVALID_REQUEST, message=f"Invalid JSON: {e}")

        except KeyboardInterrupt:
            return types.ElicitResult(action="cancel")
        except Exception as e:  # noqa: BLE001
            return types.ErrorData(code=types.INVALID_REQUEST, message=f"Elicitation failed: {e}")


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
        logger.warning("Promptantic failed", error=e)
        return None
