"""TUI routes for external control of the TUI.

These endpoints allow external integrations (e.g., VSCode extension) to control
the TUI by broadcasting events that the TUI listens for via SSE.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.models.events import (
    TuiCommandExecuteEvent,
    TuiPromptAppendEvent,
    TuiToastShowEvent,
)


router = APIRouter(prefix="/tui", tags=["tui"])


class AppendPromptRequest(BaseModel):
    """Request body for appending text to the prompt."""

    text: str = Field(..., description="Text to append to the prompt")


class ExecuteCommandRequest(BaseModel):
    """Request body for executing a TUI command."""

    command: str = Field(
        ...,
        description="Command to execute (e.g., 'prompt.submit', 'prompt.clear', 'session.new')",
    )


class ShowToastRequest(BaseModel):
    """Request body for showing a toast notification."""

    title: str | None = Field(None, description="Optional toast title")
    message: str = Field(..., description="Toast message")
    variant: Literal["info", "success", "warning", "error"] = Field(
        "info", description="Toast variant"
    )
    duration: int = Field(5000, description="Duration in milliseconds")


@router.post("/append-prompt")
async def append_prompt(request: AppendPromptRequest, state: StateDep) -> bool:
    """Append text to the TUI prompt.

    Used by external integrations (e.g., VSCode) to insert text like file
    references into the prompt input.
    """
    await state.broadcast_event(TuiPromptAppendEvent.create(request.text))
    return True


@router.post("/submit-prompt")
async def submit_prompt(state: StateDep) -> bool:
    """Submit the current prompt.

    Triggers the TUI to submit whatever is currently in the prompt input.
    """
    await state.broadcast_event(TuiCommandExecuteEvent.create("prompt.submit"))
    return True


@router.post("/clear-prompt")
async def clear_prompt(state: StateDep) -> bool:
    """Clear the TUI prompt.

    Clears any text currently in the prompt input.
    """
    await state.broadcast_event(TuiCommandExecuteEvent.create("prompt.clear"))
    return True


@router.post("/execute-command")
async def execute_command(request: ExecuteCommandRequest, state: StateDep) -> bool:
    """Execute a TUI command.

    Available commands:
    - session.list, session.new, session.share, session.interrupt, session.compact
    - session.page.up, session.page.down, session.half.page.up, session.half.page.down
    - session.first, session.last
    - prompt.clear, prompt.submit
    - agent.cycle
    """
    await state.broadcast_event(TuiCommandExecuteEvent.create(request.command))
    return True


@router.post("/show-toast")
async def show_toast(request: ShowToastRequest, state: StateDep) -> bool:
    """Show a toast notification in the TUI."""
    await state.broadcast_event(
        TuiToastShowEvent.create(
            message=request.message,
            variant=request.variant,
            title=request.title,
            duration=request.duration,
        )
    )
    return True


# Additional convenience endpoints matching OpenCode's API


@router.post("/open-help")
async def open_help(state: StateDep) -> bool:
    """Open the help dialog."""
    await state.broadcast_event(TuiCommandExecuteEvent.create("help.open"))
    return True


@router.post("/open-sessions")
async def open_sessions(state: StateDep) -> bool:
    """Open the session selector."""
    await state.broadcast_event(TuiCommandExecuteEvent.create("session.list"))
    return True


@router.post("/open-themes")
async def open_themes(state: StateDep) -> bool:
    """Open the theme selector."""
    await state.broadcast_event(TuiCommandExecuteEvent.create("theme.list"))
    return True


@router.post("/open-models")
async def open_models(state: StateDep) -> bool:
    """Open the model selector."""
    await state.broadcast_event(TuiCommandExecuteEvent.create("model.list"))
    return True
