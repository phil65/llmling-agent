"""Task-related exceptions."""

from __future__ import annotations

from llmling_agent.utils.baseregistry import LLMLingError


class JobError(LLMLingError):
    """General task-related exception."""


class ToolSkippedError(JobError):
    """Tool execution was skipped by user."""


class RunAbortedError(JobError):
    """Run was aborted by user."""


class ChainAbortedError(JobError):
    """Agent chain was aborted by user."""


class JobRegistrationError(JobError):
    """Task could not get registered."""
