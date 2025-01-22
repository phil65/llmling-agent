from __future__ import annotations

from collections.abc import Awaitable, Callable
import os
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

from llmling import LLMCallableTool
from typing_extensions import TypeVar


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for model objects."""

    def name(self) -> str: ...


# Define what we consider JSON-serializable
type JsonPrimitive = None | bool | int | float | str
type JsonValue = JsonPrimitive | JsonArray | JsonObject
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]

type StrPath = str | os.PathLike[str]
type SessionIdType = str | UUID | None


AgentName = str
MessageRole = Literal["user", "assistant", "system"]
PartType = Literal["text", "image", "audio", "video"]
ModelType = ModelProtocol | str | None
EnvironmentType = Literal["file", "inline"]
ToolSource = Literal["runtime", "agent", "builtin", "dynamic", "task", "mcp"]
AnyCallable = Callable[..., Any]
AsyncFilterFn = Callable[..., Awaitable[bool]]
SyncFilterFn = Callable[..., bool]
AnyFilterFn = Callable[..., bool | Awaitable[bool]]
AnyTransformFn = Callable[[Any], Any | Awaitable[Any]]
T = TypeVar("T")
type OptionalAwaitable[T] = T | Awaitable[T]

type ToolType = str | AnyCallable | LLMCallableTool

# P = ParamSpec("P")
# SyncAsync = Callable[P, OptionalAwaitable[T]]
