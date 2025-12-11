"""Client ACP Connection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from acp.schema import (
        AuthenticateRequest,
        AuthenticateResponse,
        CancelNotification,
        ForkSessionRequest,
        ForkSessionResponse,
        InitializeRequest,
        InitializeResponse,
        ListSessionsRequest,
        ListSessionsResponse,
        LoadSessionRequest,
        LoadSessionResponse,
        NewSessionRequest,
        NewSessionResponse,
        PromptRequest,
        PromptResponse,
        ResumeSessionRequest,
        ResumeSessionResponse,
        SetSessionModelRequest,
        SetSessionModelResponse,
        SetSessionModeRequest,
        SetSessionModeResponse,
    )


class Agent(Protocol):
    """Base agent interface for ACP."""

    async def initialize(self, params: InitializeRequest) -> InitializeResponse: ...

    async def new_session(self, params: NewSessionRequest) -> NewSessionResponse: ...

    async def prompt(self, params: PromptRequest) -> PromptResponse: ...

    async def cancel(self, params: CancelNotification) -> None: ...

    async def load_session(self, params: LoadSessionRequest) -> LoadSessionResponse: ...

    async def list_sessions(self, params: ListSessionsRequest) -> ListSessionsResponse: ...

    async def fork_session(self, params: ForkSessionRequest) -> ForkSessionResponse: ...

    async def resume_session(self, params: ResumeSessionRequest) -> ResumeSessionResponse: ...

    async def authenticate(self, params: AuthenticateRequest) -> AuthenticateResponse | None: ...

    async def set_session_mode(
        self, params: SetSessionModeRequest
    ) -> SetSessionModeResponse | None: ...

    async def set_session_model(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse | None: ...

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None: ...
