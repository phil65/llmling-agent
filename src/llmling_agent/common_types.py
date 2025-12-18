"""Type definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    get_args,
    get_origin,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_ai import AgentStreamEvent, RunContext
from pydantic_ai.models import Model
from tokonomics import ModelName
from toprompt.to_prompt import AnyPromptType
from upathtools import JoinablePathLike


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID

    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.messaging import ChatMessage
    from llmling_agent.messaging.messagenode import MessageNode
    from llmling_agent.tools.base import Tool

    # Type alias for team streaming return type (node + event tuples)
    type TeamStreamEvent = tuple[MessageNode[Any, Any], RichAgentStreamEvent[Any]]
    type AnyTransformFn[T] = Callable[[T], T | Awaitable[T]]
    type OptionalAwaitable[T] = T | Awaitable[T]
    type ToolType = str | AnyCallable | Tool
    # Define what we consider JSON-serializable
    type JsonPrimitive = None | bool | int | float | str
    type SessionIdType = str | UUID | None
    type ProcessorCallback[TResult] = Callable[..., TResult | Awaitable[TResult]]

# In reflex for example, the complex ones create issues..
SimpleJsonType = dict[
    str, bool | int | float | str | list[str] | dict[str, bool | int | float | str]
]
type JsonValue = JsonPrimitive | JsonArray | JsonObject
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]


NodeName = str
TeamName = str
AgentName = str
MessageRole = Literal["user", "assistant"]
PartType = Literal["text", "image", "audio", "video"]
ModelType = Model | ModelName | str | None
EnvironmentType = Literal["file", "inline"]
ToolSource = Literal["agent", "builtin", "dynamic", "task", "mcp", "toolset"]
AnyCallable = Callable[..., Any]
AsyncFilterFn = Callable[..., Awaitable[bool]]
SyncFilterFn = Callable[..., bool]
AnyFilterFn = Callable[..., bool | Awaitable[bool]]
# Event handler types for composable event processing
# Individual event handler for composability - takes single events
IndividualEventHandler = Callable[[RunContext, AgentStreamEvent], Awaitable[None]]
BuiltinEventHandlerType = Literal["simple", "detailed"]
PromptCompatible = AnyPromptType | JoinablePathLike
# P = ParamSpec("P")
# SyncAsync = Callable[P, OptionalAwaitable[T]]
EndStrategy = Literal["early", "exhaustive"]
QueueStrategy = Literal["concat", "latest", "buffer"]
"""The strategy for handling multiple tool calls when a final result is found.

- `'early'`: Stop processing other tool calls once a final result is found
- `'exhaustive'`: Process all tool calls even after finding a final result
"""


class SupportsStructuredOutput(Protocol):
    """Protocol for nodes that support structured output via run().

    This protocol is used for components that need to call run() with
    an output_type parameter (e.g., picker agents, Interactions).
    """

    async def run(self, *prompts: Any, output_type: Any = ...) -> ChatMessage[Any]: ...


@runtime_checkable
class SupportsRunStream[TResult](Protocol):
    """Protocol for nodes that support streaming via run_stream().

    Used by Team and TeamRun to check if a node can be streamed.

    Return type is a union because:
    - Agent returns `AsyncIterator[RichAgentStreamEvent[TResult]]`
    - Team/TeamRun return `AsyncIterator[tuple[MessageNode, RichAgentStreamEvent]]`
    """

    def run_stream(
        self, *prompts: Any, **kwargs: Any
    ) -> AsyncIterator[RichAgentStreamEvent[TResult] | TeamStreamEvent]: ...


class BaseCode(BaseModel):
    """Base class for syntax-validated code."""

    code: str
    """The source code."""

    @field_validator("code")
    @classmethod
    def validate_syntax(cls, code: str) -> str:
        """Override in subclasses."""
        return code

    model_config = ConfigDict(use_attribute_docstrings=True)


def _validate_type_args(data: Any, args: tuple[Any, ...]) -> None:
    """Validate data against type arguments."""
    match data:
        case dict() if len(args) == 2:  # noqa: PLR2004
            key_type, value_type = args
            for k, v in data.items():
                if not isinstance(k, key_type):
                    msg = f"Invalid key type: {type(k)}, expected {key_type}"
                    raise ValueError(msg)  # noqa: TRY004
                if not isinstance(v, value_type):
                    msg = f"Invalid value type: {type(v)}, expected {value_type}"
                    raise ValueError(msg)  # noqa: TRY004
        case list() if len(args) == 1:
            item_type = args[0]
            for item in data:
                if not isinstance(item, item_type):
                    msg = f"Invalid item type: {type(item)}, expected {item_type}"
                    raise ValueError(msg)  # noqa: TRY004


class ConfigCode[T](BaseCode):
    """Base class for configuration code that validates against a specific type.

    Generic type T specifies the type to validate against.
    """

    validator_type: ClassVar[type]

    @field_validator("code")
    @classmethod
    def validate_syntax(cls, code: str) -> str:
        """Validate both YAML syntax and type constraints."""
        import yamling

        try:
            # First validate YAML syntax
            data = yamling.load(code, mode="yaml")

            # Then validate against target type
            match cls.validator_type:
                case type() as model_cls if issubclass(model_cls, BaseModel):
                    model_cls.model_validate(data)
                case _ if origin := get_origin(cls.validator_type):
                    # Handle generics like dict[str, int]
                    if not isinstance(data, origin):
                        msg = f"Expected {origin.__name__}, got {type(data).__name__}"
                        raise ValueError(msg)  # noqa: TRY004, TRY301
                    # Validate type arguments if present
                    if args := get_args(cls.validator_type):
                        _validate_type_args(data, args)
                case _:
                    msg = f"Unsupported validation type: {cls.validator_type}"
                    raise TypeError(msg)  # noqa: TRY301

        except Exception as e:
            msg = f"Invalid YAML for {cls.validator_type.__name__}: {e}"
            raise ValueError(msg) from e

        return code

    @classmethod
    def for_config[TConfig](
        cls,
        base_type: type[TConfig],
        *,
        name: str | None = None,
        error_msg: str | None = None,
    ) -> type[ConfigCode[TConfig]]:
        """Create a new ConfigCode class for a specific type.

        Args:
            base_type: The type to validate against
            name: Optional name for the new class
            error_msg: Optional custom error message

        Returns:
            New ConfigCode subclass with type-specific validation
        """

        class TypedConfigCode(ConfigCode[TConfig]):
            validator_type = base_type

            @field_validator("code")
            @classmethod
            def validate_syntax(cls, code: str) -> str:
                try:
                    return super().validate_syntax(code)
                except ValueError as e:
                    msg = error_msg or str(e)
                    raise ValueError(msg) from e

        if name:
            TypedConfigCode.__name__ = name

        return TypedConfigCode


if __name__ == "__main__":
    from llmling_agent.models.manifest import AgentsManifest

    AgentsManifestCode = ConfigCode.for_config(
        AgentsManifest,
        name="AgentsManifestCode",
        error_msg="Invalid agents manifest YAML",
    )
