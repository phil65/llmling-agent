"""Models for OpenAI responses endpoint."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal, TypedDict
from uuid import uuid4

from pydantic import Field
from schemez import Schema


class InputText(Schema):
    """Text input part."""

    type: Literal["input_text"] = "input_text"
    text: str


class InputImage(Schema):
    """Image input part."""

    type: Literal["input_image"] = "input_image"
    image_url: str


class ResponseOutputText(Schema):
    """Text output part."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ResponseToolCall(Schema):
    """Tool call in response."""

    type: str  # web_search_call etc
    id: str
    status: Literal["completed", "error"] = "completed"


class ResponseMessage(Schema):
    """ResponseMessage in response."""

    type: Literal["message"] = "message"
    id: str
    status: Literal["completed", "error"] = "completed"
    role: Literal["user", "assistant", "system"]
    content: list[ResponseOutputText]


class ResponseUsage(TypedDict):
    """Token usage information."""

    input_tokens: int
    input_tokens_details: dict[str, int]
    output_tokens: int
    output_tokens_details: dict[str, int]
    total_tokens: int


class ResponseRequest(Schema):
    """Request for /v1/responses endpoint."""

    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    stream: bool = False
    temperature: float = 1.0
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str = "auto"
    max_output_tokens: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Response(Schema):
    """Response from /v1/responses endpoint."""

    id: str = Field(default_factory=lambda: f"resp_{uuid4().hex}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    status: Literal["completed", "error"] = "completed"
    error: str | None = None
    model: str
    output: Sequence[ResponseMessage | ResponseToolCall]

    # Include all the request parameters
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float = 1.0
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str = "auto"
    usage: ResponseUsage | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
