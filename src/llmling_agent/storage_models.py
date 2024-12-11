"""Logging configuration for llmling_agent."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy import JSON, Column, DateTime
from sqlmodel import Field, SQLModel


class MessageLog(BaseModel):
    """Raw message log entry."""

    timestamp: datetime
    role: str
    content: str
    token_usage: dict[str, int] | None = None  # as provided by model
    cost: float | None = None
    model: str | None = None

    model_config = ConfigDict(frozen=True)


class ConversationLog(BaseModel):
    """Collection of messages forming a conversation."""

    id: str
    agent_name: str
    start_time: datetime
    messages: list[MessageLog]

    model_config = ConfigDict(frozen=True)


class Message(SQLModel, table=True):  # type: ignore[call-arg]
    """Database model for message logs."""

    id: str | None = Field(default=None, primary_key=True)
    conversation_id: str = Field(index=True)
    timestamp: datetime = Field(default_factory=datetime.now, sa_column=Column(DateTime))
    role: str
    content: str
    token_usage: dict[str, int] | None = Field(default=None, sa_column=Column(JSON))
    cost: float | None = Field(default=None)
    model: str | None = Field(default=None)


class Conversation(SQLModel, table=True):  # type: ignore[call-arg]
    """Database model for conversations."""

    id: str = Field(primary_key=True)
    agent_name: str = Field(index=True)
    start_time: datetime = Field(
        default_factory=datetime.now, sa_column=Column(DateTime, index=True)
    )