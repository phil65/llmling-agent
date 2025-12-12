"""Database models for llmling_agent storage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import ConfigDict
from schemez import Schema
from sqlalchemy import Column, DateTime, Text
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.types import TypeDecorator
from sqlmodel import JSON, Field, SQLModel
from sqlmodel.main import SQLModelConfig  # type: ignore[attr-defined]

from llmling_agent.common_types import JsonValue
from llmling_agent.utils.now import get_now


if TYPE_CHECKING:
    from sqlalchemy import Dialect


class UTCDateTime(TypeDecorator[datetime]):
    """Stores DateTime as UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        if value is not None:
            value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return value

    def process_result_value(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        if value is not None:
            value = value.replace(tzinfo=UTC)
        return value


class CommandHistory(AsyncAttrs, SQLModel, table=True):
    """Database model for command history."""

    id: int = Field(default=None, primary_key=True)
    """Primary key for command history entry"""

    session_id: str = Field(index=True)
    """ID of the chat session"""

    agent_name: str = Field(index=True)
    """Name of the agent that executed the command"""

    command: str
    """The command that was executed"""

    context_type: str | None = Field(default=None, index=True)
    """Type of the command context (e.g. 'AgentContext', 'PoolSupervisor', etc.)"""

    context_metadata: dict[str, JsonValue] = Field(default_factory=dict, sa_column=Column(JSON))
    """Additional context information about command execution"""

    timestamp: datetime = Field(
        sa_column=Column(UTCDateTime, default=get_now), default_factory=get_now
    )
    """When the command was executed"""

    model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]


class MessageLog(Schema):
    """Raw message log entry."""

    timestamp: datetime
    """When the message was sent"""

    role: str
    """Role of the message sender (user/assistant/system)"""

    content: str
    """Content of the message"""

    token_usage: dict[str, int] | None = None
    """Token usage statistics as provided by model"""

    cost: float | None = None
    """Cost of generating this message in USD"""

    model: str | None = None
    """Name of the model that generated this message"""

    model_config = ConfigDict(frozen=True)


class ConversationLog(Schema):
    """Collection of messages forming a conversation."""

    id: str
    """Unique identifier for the conversation"""

    agent_name: str
    """Name of the agent handling the conversation"""

    start_time: datetime
    """When the conversation started"""

    messages: list[MessageLog]
    """List of messages in the conversation"""

    model_config = ConfigDict(frozen=True)


class Message(AsyncAttrs, SQLModel, table=True):
    """Database model for message logs."""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    """Unique identifier for the message"""

    conversation_id: str = Field(index=True)
    """ID of the conversation this message belongs to"""

    timestamp: datetime = Field(
        sa_column=Column(UTCDateTime, default=get_now), default_factory=get_now
    )
    """When the message was sent"""

    role: str
    """Role of the message sender (user/assistant/system)"""

    name: str | None = Field(default=None, index=True)
    """Display name of the sender"""

    content: str
    """Content of the message"""

    model: str | None = None
    """Full model identifier (including provider)"""

    model_name: str | None = Field(default=None, index=True)
    """Name of the model (e.g., "gpt-5")"""

    model_provider: str | None = Field(default=None, index=True)
    """Provider of the model (e.g., "openai")"""

    forwarded_from: list[str] | None = Field(default=None, sa_column=Column(JSON))
    """List of agent names that forwarded this message"""

    total_tokens: int | None = Field(default=None, index=True)
    """Total number of tokens used"""

    input_tokens: int | None = None
    """Number of tokens in the prompt"""

    output_tokens: int | None = None
    """Number of tokens in the completion"""

    cost: float | None = Field(default=None, index=True)
    """Cost of generating this message in USD"""

    response_time: float | None = None
    """Time taken to generate the response in seconds"""

    provider_name: str | None = Field(default=None, index=True)
    """Name of the LLM provider that generated the response"""

    provider_response_id: str | None = None
    """Request ID as specified by the model provider"""

    messages: str | None = None
    """Serialized pydantic-ai model messages"""

    finish_reason: str | None = None  # Literal pydantic_ai.FinishReason
    """Reason the model finished generating the response"""

    checkpoint_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    """A dictionary of checkpoints (name -> metadata)."""

    model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]


class Session(AsyncAttrs, SQLModel, table=True):
    """Database model for session persistence."""

    session_id: str = Field(primary_key=True)
    """Unique session identifier."""

    agent_name: str = Field(index=True)
    """Name of the currently active agent."""

    conversation_id: str = Field(index=True)
    """Links to conversation in message storage."""

    pool_id: str | None = Field(default=None, index=True)
    """Optional pool/manifest identifier for multi-pool setups."""

    title: str | None = Field(default=None, index=True)
    """AI-generated or user-provided title for the conversation."""

    cwd: str | None = Field(default=None, sa_column=Column(Text))
    """Working directory for the session."""

    created_at: datetime = Field(
        sa_column=Column(UTCDateTime, index=True),
        default_factory=get_now,
    )
    """When the session was created."""

    last_active: datetime = Field(
        sa_column=Column(UTCDateTime, index=True),
        default_factory=get_now,
    )
    """Last activity timestamp."""

    metadata_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    """Protocol-specific or custom metadata stored as JSON."""

    model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]


class Conversation(AsyncAttrs, SQLModel, table=True):
    """Database model for conversations."""

    id: str = Field(primary_key=True)
    """Unique identifier for the conversation"""

    agent_name: str = Field(index=True)
    """Name of the agent handling the conversation"""

    title: str | None = Field(default=None, index=True)
    """Generated title for the conversation"""

    start_time: datetime = Field(sa_column=Column(UTCDateTime, index=True), default_factory=get_now)
    """When the conversation started"""

    total_tokens: int = 0
    """Total number of tokens used in this conversation"""

    total_cost: float = 0.0
    """Total cost of this conversation in USD"""

    model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]
