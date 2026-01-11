"""Integration tests for OpenCode question system."""

from __future__ import annotations

import asyncio

from mcp import types
import pytest

from agentpool_server.opencode_server.input_provider import OpenCodeInputProvider


@pytest.mark.asyncio
async def test_question_elicitation_single_select():
    """Test single-select question via elicitation."""
    # This is a basic unit test without full server
    from agentpool_server.opencode_server.state import ServerState

    # Create minimal state
    state = ServerState(
        working_dir="/tmp",
        pool=None,  # type: ignore
        agent=None,  # type: ignore
    )

    # Create provider
    provider = OpenCodeInputProvider(state=state, session_id="test_session")

    # Create elicitation params with enum
    params = types.ElicitRequestFormParams(
        message="Which database?",
        requestedSchema={
            "type": "string",
            "enum": ["PostgreSQL", "MySQL", "SQLite"],
        },
    )

    # Start elicitation in background
    async def get_answer():
        return await provider.get_elicitation(params)

    task = asyncio.create_task(get_answer())

    # Wait a bit for question to be created
    await asyncio.sleep(0.1)

    # Verify question was created
    assert len(state.pending_questions) == 1
    question_id = next(iter(state.pending_questions.keys()))
    pending = state.pending_questions[question_id]

    # Verify question structure
    assert pending.session_id == "test_session"
    assert len(pending.questions) == 1
    question_info = pending.questions[0]
    assert question_info.question == "Which database?"
    assert question_info.multiple is None
    assert len(question_info.options) == 3

    # Simulate user reply
    success = provider.resolve_question(question_id, [["PostgreSQL"]])
    assert success

    # Wait for result
    result = await task

    # Verify result
    assert isinstance(result, types.ElicitResult)
    assert result.action == "accept"
    assert result.content == {"value": "PostgreSQL"}

    # Verify cleanup
    assert question_id not in state.pending_questions


@pytest.mark.asyncio
async def test_question_elicitation_multi_select():
    """Test multi-select question via elicitation."""
    from agentpool_server.opencode_server.state import ServerState

    state = ServerState(
        working_dir="/tmp",
        pool=None,  # type: ignore
        agent=None,  # type: ignore
    )

    provider = OpenCodeInputProvider(state=state, session_id="test_session")

    # Multi-select schema
    params = types.ElicitRequestFormParams(
        message="Which features?",
        requestedSchema={
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["Auth", "API", "Admin"],
            },
        },
    )

    task = asyncio.create_task(provider.get_elicitation(params))
    await asyncio.sleep(0.1)

    # Get question
    question_id = next(iter(state.pending_questions.keys()))
    pending = state.pending_questions[question_id]
    question_info = pending.questions[0]

    # Verify multi-select flag
    assert question_info.multiple is True

    # Reply with multiple selections
    provider.resolve_question(question_id, [["Auth", "Admin"]])

    result = await task

    # Multi-select returns list in dict
    assert isinstance(result, types.ElicitResult)
    assert result.action == "accept"
    assert result.content == {"value": ["Auth", "Admin"]}


@pytest.mark.asyncio
async def test_question_cancellation():
    """Test question cancellation."""
    from agentpool_server.opencode_server.state import ServerState

    state = ServerState(
        working_dir="/tmp",
        pool=None,  # type: ignore
        agent=None,  # type: ignore
    )

    provider = OpenCodeInputProvider(state=state, session_id="test_session")

    params = types.ElicitRequestFormParams(
        message="Which database?",
        requestedSchema={
            "type": "string",
            "enum": ["PostgreSQL", "MySQL"],
        },
    )

    task = asyncio.create_task(provider.get_elicitation(params))
    await asyncio.sleep(0.1)

    # Get question and cancel it
    question_id = next(iter(state.pending_questions.keys()))
    future = state.pending_questions[question_id].future
    future.cancel()

    result = await task

    # Should return cancel action
    assert isinstance(result, types.ElicitResult)
    assert result.action == "cancel"


@pytest.mark.asyncio
async def test_question_with_descriptions():
    """Test question with option descriptions."""
    from agentpool_server.opencode_server.state import ServerState

    state = ServerState(
        working_dir="/tmp",
        pool=None,  # type: ignore
        agent=None,  # type: ignore
    )

    provider = OpenCodeInputProvider(state=state, session_id="test_session")

    # Schema with custom descriptions
    params = types.ElicitRequestFormParams(
        message="Which database?",
        requestedSchema={
            "type": "string",
            "enum": ["PostgreSQL", "MySQL", "SQLite"],
            "x-option-descriptions": {
                "PostgreSQL": "Best for production",
                "MySQL": "Compatible with many tools",
                "SQLite": "Lightweight, file-based",
            },
        },
    )

    task = asyncio.create_task(provider.get_elicitation(params))
    await asyncio.sleep(0.1)

    # Verify descriptions were included
    question_id = next(iter(state.pending_questions.keys()))
    question_info = state.pending_questions[question_id].questions[0]
    options = question_info.options

    assert options[0].label == "PostgreSQL"
    assert options[0].description == "Best for production"
    assert options[1].description == "Compatible with many tools"

    # Clean up
    future = state.pending_questions[question_id].future
    future.cancel()
    await task


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
