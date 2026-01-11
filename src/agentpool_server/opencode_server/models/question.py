"""Question models for OpenCode compatibility."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QuestionOption(BaseModel):
    """A single option for a question."""

    label: str = Field(description="Display text (1-5 words, concise)")
    description: str = Field(description="Explanation of choice")


class QuestionInfo(BaseModel):
    """Information about a single question."""

    question: str = Field(description="Complete question")
    header: str = Field(max_length=12, description="Very short label (max 12 chars)")
    options: list[QuestionOption] = Field(description="Available choices")
    multiple: bool | None = Field(None, description="Allow selecting multiple choices")


class QuestionRequest(BaseModel):
    """A pending question request."""

    id: str
    sessionID: str
    questions: list[QuestionInfo]
    tool: dict[str, str] | None = None  # {messageID, callID}


class QuestionReply(BaseModel):
    """Reply to a question request."""

    answers: list[list[str]] = Field(
        description="User answers in order of questions (each answer is an array of selected labels)"
    )
