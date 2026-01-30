"""Question routes for OpenCode compatibility."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agentpool_server.opencode_server.dependencies import StateDep
from agentpool_server.opencode_server.input_provider import OpenCodeInputProvider
from agentpool_server.opencode_server.models.events import (
    QuestionRejectedEvent,
    QuestionRepliedEvent,
)
from agentpool_server.opencode_server.models.question import QuestionReply, QuestionRequest


router = APIRouter(prefix="/question", tags=["question"])


@router.get("/", response_model=list[QuestionRequest])
async def list_questions(state: StateDep) -> list[QuestionRequest]:
    """List all pending question requests.

    Returns a list of all pending questions awaiting user response.
    """
    return [
        QuestionRequest(id=question_id, session_id=i.session_id, questions=i.questions, tool=i.tool)
        for question_id, i in state.pending_questions.items()
    ]


@router.post("/{requestID}/reply")
async def reply_to_question(
    requestID: str,  # noqa: N803
    reply: QuestionReply,
    state: StateDep,
) -> bool:
    """Reply to a question request.

    The user provides answers to the questions. Answers must be provided
    as an array of arrays, where each inner array contains the selected
    label(s) for that question.

    Args:
        requestID: The question request ID
        reply: The user's answers
        state: Server state

    Returns:
        True if the question was resolved successfully

    Raises:
        HTTPException: If question not found or invalid provider
    """
    pending = state.pending_questions.get(requestID)
    if not pending:
        raise HTTPException(status_code=404, detail="Question request not found")

    session_id = pending.session_id
    provider = state.input_providers.get(session_id)

    if not isinstance(provider, OpenCodeInputProvider):
        raise HTTPException(status_code=500, detail="Invalid provider for session")

    # Resolve via provider
    if not provider.resolve_question(requestID, reply.answers):
        raise HTTPException(status_code=404, detail="Question already resolved")

    # Broadcast replied event
    event = QuestionRepliedEvent.create(
        session_id=session_id,
        request_id=requestID,
        answers=reply.answers,
    )
    await state.broadcast_event(event)

    return True


@router.post("/{requestID}/reject")
async def reject_question(requestID: str, state: StateDep) -> bool:  # noqa: N803
    """Reject a question request.

    Called when the user dismisses the question without providing an answer.

    Args:
        requestID: The question request ID
        state: Server state

    Returns:
        True if the question was rejected successfully

    Raises:
        HTTPException: If question not found
    """
    pending = state.pending_questions.get(requestID)
    if not pending:
        raise HTTPException(status_code=404, detail="Question request not found")

    session_id = pending.session_id
    future = pending.future
    # Cancel the future
    if not future.done():
        future.cancel()
    # Remove from pending
    del state.pending_questions[requestID]
    # Broadcast rejected event
    event = QuestionRejectedEvent.create(session_id=session_id, request_id=requestID)
    await state.broadcast_event(event)
    return True
