from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlmodel import Session, select

from llmling_agent.history.filters import parse_time_period
from llmling_agent.history.models import QueryFilters, StatsFilters
from llmling_agent.history.queries import get_conversations, get_stats_data
from llmling_agent.storage import Conversation, Message, engine


@pytest.fixture(autouse=True)
def cleanup_database() -> None:
    """Clean up the database before each test."""
    with Session(engine) as session:
        # Delete all messages first (due to foreign key)
        stmt = select(Message)
        messages = session.exec(stmt).all()
        for msg in messages:
            session.delete(msg)

        stmt = select(Conversation)
        conversations = session.exec(stmt).all()
        for conv in conversations:
            session.delete(conv)

        session.commit()


@pytest.fixture
def sample_data(cleanup_database: None) -> None:
    """Create sample conversation data."""
    with Session(engine) as session:
        # Create two conversations
        conv1 = Conversation(
            id="conv1",
            agent_name="test_agent",
            start_time=datetime.now() - timedelta(hours=1),
        )
        conv2 = Conversation(
            id="conv2",
            agent_name="other_agent",
            start_time=datetime.now() - timedelta(hours=2),
        )
        session.add(conv1)
        session.add(conv2)
        session.commit()

        # Add messages to conversations
        messages = [
            Message(
                id="msg1",
                conversation_id="conv1",
                timestamp=datetime.now() - timedelta(hours=1),
                role="user",
                content="Hello",
                model="gpt-4",
                token_usage={"total": 10, "prompt": 5, "completion": 5},
            ),
            Message(
                id="msg2",
                conversation_id="conv1",
                timestamp=datetime.now() - timedelta(minutes=55),
                role="assistant",
                content="Hi there!",
                model="gpt-4",
                token_usage={"total": 20, "prompt": 10, "completion": 10},
            ),
            Message(
                id="msg3",
                conversation_id="conv2",
                timestamp=datetime.now() - timedelta(hours=2),
                role="user",
                content="Testing",
                model="gpt-3.5-turbo",
                token_usage={"total": 15, "prompt": 7, "completion": 8},
            ),
        ]
        for msg in messages:
            session.add(msg)
        session.commit()


def test_parse_time_period() -> None:
    """Test time period parsing."""
    assert parse_time_period("1h") == timedelta(hours=1)
    assert parse_time_period("2d") == timedelta(days=2)
    assert parse_time_period("1w") == timedelta(weeks=1)


def test_get_conversations(sample_data: None) -> None:
    """Test conversation retrieval with filters."""
    # Get all conversations
    filters = QueryFilters()
    results = get_conversations(filters)
    assert len(results) == 2  # noqa: PLR2004

    # Filter by agent
    filters = QueryFilters(agent_name="test_agent")
    results = get_conversations(filters)
    assert len(results) == 1
    conv, msgs = results[0]
    assert conv.agent_name == "test_agent"
    assert len(msgs) == 2  # noqa: PLR2004

    # Filter by time
    filters = QueryFilters(since=datetime.now() - timedelta(hours=1.5))
    results = get_conversations(filters)
    assert len(results) == 1


def test_get_stats_data(sample_data: None) -> None:
    """Test statistics data retrieval."""
    filters = StatsFilters(
        cutoff=datetime.now() - timedelta(hours=3),
        group_by="model",
    )
    rows = get_stats_data(filters)

    # Should get all messages
    assert len(rows) == 3  # noqa: PLR2004

    # Check structure of returned data
    model, agent, timestamp, usage = rows[0]
    assert isinstance(model, str | None)
    assert isinstance(agent, str | None)
    assert isinstance(timestamp, datetime)
    assert isinstance(usage, dict | None)


def test_complex_filtering(sample_data: None) -> None:
    """Test combined filtering capabilities."""
    # Filter by multiple criteria
    filters = QueryFilters(
        agent_name="test_agent",
        model="gpt-4",
        since=datetime.now() - timedelta(hours=1.5),
        query="Hello",  # Content search
    )
    results = get_conversations(filters)
    assert len(results) == 1
    conv, msgs = results[0]
    assert conv.agent_name == "test_agent"
    assert any(msg.content == "Hello" for msg in msgs)
    assert all(msg.model == "gpt-4" for msg in msgs)


def test_stats_aggregation(sample_data: None) -> None:
    """Test statistics aggregation by different criteria."""
    from llmling_agent.history.stats import aggregate_stats

    # Get raw data
    filters = StatsFilters(
        cutoff=datetime.now() - timedelta(hours=3),
        group_by="model",
    )
    rows = get_stats_data(filters)

    # Test model grouping
    model_stats = aggregate_stats(rows, "model")
    assert "gpt-4" in model_stats
    assert model_stats["gpt-4"]["messages"] == 2  # Two GPT-4 messages  # noqa: PLR2004
    assert model_stats["gpt-4"]["total_tokens"] == 30  # 10 + 20 tokens  # noqa: PLR2004

    # Test hour grouping
    hour_stats = aggregate_stats(rows, "hour")
    # Should have two time slots (one hour ago and two hours ago)
    assert len(hour_stats) == 2  # noqa: PLR2004

    # Test aggregation totals
    total_messages = sum(s["messages"] for s in model_stats.values())
    total_tokens = sum(s["total_tokens"] for s in model_stats.values())
    assert total_messages == 3  # Total messages across all models  # noqa: PLR2004
    assert total_tokens == 45  # Total tokens (10 + 20 + 15)  # noqa: PLR2004