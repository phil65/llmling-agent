"""Slash command handling for chat sessions."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import typing
from typing import TYPE_CHECKING, Any, TypedDict

from llmling_agent.chat_session.base import AgentChatSession  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llmling.prompts import BasePrompt


class CommandResult(TypedDict):
    """Result of a command execution."""

    content: str
    metadata: dict[str, Any]


@dataclass
class CommandParameter:
    """Parameter information for a command."""

    name: str
    type: type
    required: bool
    default: Any
    description: str | None = None


@dataclass
class Command:
    """A registered command."""

    name: str
    description: str
    handler: Callable[..., Awaitable[CommandResult]]
    parameters: dict[str, CommandParameter]

    @classmethod
    def from_callable(
        cls,
        name: str,
        func: Callable[..., Awaitable[Any]],
        description: str | None = None,
    ) -> Command:
        """Create command from a callable."""
        sig = inspect.signature(func)
        hints = typing.get_type_hints(func)
        doc = inspect.getdoc(func)

        # Get parameter info
        params: dict[str, CommandParameter] = {}
        for param_name, param in sig.parameters.items():
            # Skip self/cls and session parameters
            if param_name in ("self", "cls", "session"):
                continue

            params[param_name] = CommandParameter(
                name=param_name,
                type=hints.get(param_name, str),
                required=param.default is param.empty,
                default=param.default if param.default is not param.empty else None,
                description=None,  # TODO: Parse from docstring
            )

        return cls(
            name=name,
            description=description or doc or "",
            handler=func,
            parameters=params,
        )

    @classmethod
    def from_tool(cls, name: str, tool: Any) -> Command:
        """Create command from a tool."""
        # TODO: Proper tool type
        return cls(
            name=name,
            description=tool.description,
            handler=cls._create_tool_handler(tool),
            parameters=cls._get_tool_parameters(tool),
        )

    @classmethod
    def from_prompt(cls, name: str, prompt: BasePrompt) -> Command:
        """Create command from any prompt type."""
        params = {}
        for arg in prompt.arguments:
            params[arg.name] = CommandParameter(
                name=arg.name,
                type=arg.type_hint,  # Use LLMling's type hint
                required=arg.required,
                default=arg.default,
                description=arg.description,
            )

        return cls(
            name=name,
            description=prompt.description,
            handler=cls._create_prompt_handler(prompt),
            parameters=params,
        )

    @staticmethod
    def _create_tool_handler(tool: Any) -> Callable[..., Awaitable[CommandResult]]:
        async def handler(session: AgentChatSession, **kwargs: Any) -> CommandResult:
            result = await session.runtime.execute_tool(tool.name, **kwargs)
            return CommandResult(
                content=str(result),
                metadata={"type": "tool", "name": tool.name},
            )

        return handler

    @staticmethod
    def _create_prompt_handler(
        prompt: BasePrompt,
    ) -> Callable[..., Awaitable[CommandResult]]:
        async def handler(session: AgentChatSession, **kwargs: Any) -> CommandResult:
            try:
                # Format prompt with arguments
                messages = await prompt.format(kwargs)

                # Send each message to the agent
                responses = []
                for msg in messages:
                    text = msg.get_text_content()
                    if text:  # Only send non-empty messages
                        response = await session.send_message(text)
                        responses.append(response)

                # Use last response or default message
                content = str(responses[-1]) if responses else "No response"

                return CommandResult(
                    content=content,
                    metadata={
                        "type": "prompt",
                        "name": prompt.name,
                        "prompt_type": prompt.type,  # type: ignore  # from discriminator
                    },
                )
            except ValueError as e:
                return CommandResult(
                    content=f"Error formatting prompt: {e}",
                    metadata={"type": "error"},
                )

        return handler

    @staticmethod
    def _get_tool_parameters(tool: Any) -> dict[str, CommandParameter]:
        """Extract parameters from tool schema."""
        # TODO: Implement based on tool schema
        return {}


class CommandRegistry:
    """Registry for available commands."""

    def __init__(self, session: AgentChatSession) -> None:
        """Initialize command registry."""
        self._session = session
        self._commands: dict[str, Command] = {}
        self._register_commands()

    def _register_commands(self) -> None:
        """Register all available commands."""
        # Register all prompts as commands
        for prompt in self._session.runtime.get_prompts():
            if prompt.name:
                self._commands[prompt.name] = Command.from_prompt(
                    prompt.name,
                    prompt,
                )
        # Register all tools as commands
        for name, tool in self._session.runtime.tools.items():
            self._commands[name] = Command.from_tool(name, tool)

        # Register built-in commands
        for name, func in BUILTIN_COMMANDS.items():
            self._commands[name] = Command.from_callable(name, func)

    def _parse_args(
        self,
        arg_str: str,
        parameters: dict[str, CommandParameter],
    ) -> dict[str, Any]:
        """Parse command arguments."""
        if not arg_str and not any(p.required for p in parameters.values()):
            return {}

        # Split into words, respecting quotes
        import shlex

        try:
            args = shlex.split(arg_str)
        except ValueError as e:
            msg = f"Invalid argument syntax: {e}"
            raise ValueError(msg) from e

        result = {}
        errors = []

        # Process positional arguments
        for (name, param), value in zip(parameters.items(), args):
            try:
                result[name] = param.type(value)
            except (ValueError, TypeError) as e:
                errors.append(f"Invalid value for {name}: {e}")

        if errors:
            msg = "\n".join(errors)
            raise ValueError(msg)

        # Add defaults for missing optional parameters
        for name, param in parameters.items():
            if name not in result and not param.required:
                result[name] = param.default

        # Check for missing required parameters
        missing = [
            name
            for name, param in parameters.items()
            if param.required and name not in result
        ]
        if missing:
            msg = f"Missing required arguments: {', '.join(missing)}"
            raise ValueError(msg)

        return result

    async def execute(self, text: str) -> CommandResult:
        """Execute a command from text."""
        parts = text.split(maxsplit=1)
        name = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        if cmd := self._commands.get(name):
            try:
                kwargs = self._parse_args(args, cmd.parameters)
                return await cmd.handler(self._session, **kwargs)
            except ValueError as e:
                return CommandResult(
                    content=f"Invalid arguments: {e}",
                    metadata={"type": "error", "command": name},
                )
            except Exception as e:  # noqa: BLE001
                return CommandResult(
                    content=f"Command failed: {e}",
                    metadata={"type": "error", "command": name},
                )

        return CommandResult(
            content=f"Unknown command: {name}",
            metadata={"type": "error"},
        )


# Built-in commands
async def help_command(
    session: AgentChatSession,
    topic: str | None = None,
) -> CommandResult:
    """Show help about available commands."""
    registry = session._command_registry
    if not topic:
        lines = ["Available commands:"]
        for name, cmd in sorted(registry._commands.items()):
            lines.append(f"/{name}: {cmd.description}")
        return CommandResult(
            content="\n".join(lines),
            metadata={"type": "system", "command": "help"},
        )

    if cmd_obj := registry._commands.get(topic):
        lines = [
            f"/{topic}: {cmd_obj.description}",
            "",
            "Parameters:",
        ]
        for param in cmd_obj.parameters.values():
            req = "required" if param.required else f"optional, default: {param.default}"
            desc = f" - {param.description}" if param.description else ""
            lines.append(f"  {param.name} ({req}){desc}")

        return CommandResult(
            content="\n".join(lines),
            metadata={"type": "system", "command": "help"},
        )

    return CommandResult(
        content=f"No help available for '{topic}'",
        metadata={"type": "error", "command": "help"},
    )


async def status_command(session: AgentChatSession) -> CommandResult:
    """Show current session status."""
    lines = [
        "Session Status:",
        f"Agent: {session._agent.name}",
        f"Model: {session._model or 'default'}",
        "",
        "Active Tools:",
    ]
    for name, enabled in session.get_tool_states().items():
        status = "enabled" if enabled else "disabled"
        lines.append(f"  - {name}: {status}")

    return CommandResult(
        content="\n".join(lines),
        metadata={"type": "system", "command": "status"},
    )


BUILTIN_COMMANDS: dict[str, Callable[..., Awaitable[CommandResult]]] = {
    "help": help_command,
    "status": status_command,
}
