"""Question models for OpenCode compatibility."""

from __future__ import annotations

from pydantic import Field

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel


class QuestionOption(OpenCodeBaseModel):
    """A single option for a question."""

    label: str
    """Display text (1-5 words, concise)."""

    description: str
    """Explanation of choice."""


class QuestionInfo(OpenCodeBaseModel):
    """Information about a single question."""

    question: str
    """Complete question."""

    header: str = Field(max_length=12)
    """Very short label (max 12 chars)."""

    options: list[QuestionOption]
    """Available choices."""

    multiple: bool | None = None
    """Allow selecting multiple choices."""


class QuestionRequest(OpenCodeBaseModel):
    """A pending question request."""

    id: str
    """Unique question identifier."""

    session_id: str
    """Session identifier."""

    questions: list[QuestionInfo]
    """List of questions to ask."""

    tool: dict[str, str] | None = None
    """Optional tool context: {message_id, call_id}."""


class QuestionReply(OpenCodeBaseModel):
    """Reply to a question request."""

    answers: list[list[str]]
    """User answers in order of questions (each answer is an array of selected labels)."""
