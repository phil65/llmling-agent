from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel, delete, select

from llmling_agent.messaging.messages import TokenCost
from llmling_agent.utils.now import get_now
from llmling_agent.utils.parse_time import parse_time_period
from llmling_agent_config.storage import SQLStorageConfig
from llmling_agent_storage.models import QueryFilters, StatsFilters
from llmling_agent_storage.sql_provider import Conversation, Message, SQLModelProvider


# Reference time for all tests
BASE_TIME = datetime(2024, 1, 1, 12, 0)  # noon on Jan 1, 2024

# Create async engine for testing
async_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    connect_args={"check_same_thread": False},
)


async def create_async_tables():
    """Create tables for async engine."""
    from sqlalchemy.ext.asyncio import create_async_engine

    temp_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with temp_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await temp_engine.dispose()


@pytest.fixture(autouse=True)
async def cleanup_database():
    """Clean up the database before each test."""
    # Create tables for async engine
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Clean database
    async with AsyncSession(async_engine) as session:
        await session.execute(delete(Message))
        await session.execute(delete(Conversation))
        await session.commit()


@pytest.fixture
async def sample_data(cleanup_database: None):
    """Create sample conversation data."""
    async with AsyncSession(async_engine) as session:
        # Create two conversations
        start = BASE_TIME - timedelta(hours=1)  # 11:00
        conv1 = Conversation(id="conv1", agent_name="test_agent", start_time=start)
        start = BASE_TIME - timedelta(hours=2)  # 10:00
        conv2 = Conversation(id="conv2", agent_name="other_agent", start_time=start)
        session.add(conv1)
        session.add(conv2)
        await session.commit()

        # Create provider for async operations
        config = SQLStorageConfig(url="sqlite+aiosqlite:///:memory:")
        provider = SQLModelProvider(config, async_engine)

        # Add messages
        test_data = [
            (
                "conv1",  # conversation_id
                "Hello",  # content
                "user",  # role
                "user",  # name
                "gpt-5",  # model
                TokenCost(
                    token_usage={"total": 10, "prompt": 5, "completion": 5},
                    total_cost=Decimal("0.001"),
                ),  # cost_info
            ),
            (
                "conv1",
                "Hi there!",
                "assistant",
                "test_agent",
                "gpt-5",
                TokenCost(
                    token_usage={"total": 20, "prompt": 10, "completion": 10},
                    total_cost=Decimal("0.002"),
                ),
            ),
            (
                "conv2",
                "Testing",
                "user",
                "user",
                "gpt-3.5-turbo",
                TokenCost(
                    token_usage={"total": 15, "prompt": 7, "completion": 8},
                    total_cost=Decimal("0.0015"),
                ),
            ),
        ]

        # Add messages using the provider's method signature
        for conv_id, content, role, name, model, cost_info in test_data:
            await provider.log_message(
                conversation_id=conv_id,
                message_id=str(uuid4()),
                content=content,
                role=role,
                name=name,
                model=model,
                cost_info=cost_info,
                response_time=None,
                forwarded_from=None,
            )


@pytest.fixture
async def provider():
    """Create SQLModelProvider instance."""
    config = SQLStorageConfig(url="sqlite+aiosqlite:///:memory:")
    return SQLModelProvider(config, async_engine)


def test_parse_time_period():
    """Test time period parsing."""
    assert parse_time_period("1h") == timedelta(hours=1)
    assert parse_time_period("2d") == timedelta(days=2)
    assert parse_time_period("1w") == timedelta(weeks=1)


async def test_get_conversations(provider: SQLModelProvider, sample_data: None):
    """Test conversation retrieval with filters."""
    # Get all conversations
    filters = QueryFilters()
    results = await provider.get_conversations(filters)
    assert len(results) == 2  # noqa: PLR2004

    # Filter by agent
    filters = QueryFilters(agent_name="test_agent")
    results = await provider.get_conversations(filters)
    assert len(results) == 1
    conv, msgs = results[0]
    assert conv["agent"] == "test_agent"
    assert len(msgs) == 2  # noqa: PLR2004

    # Filter by time
    filters = QueryFilters(since=BASE_TIME - timedelta(hours=1.5))
    results = await provider.get_conversations(filters)
    assert len(results) == 1


async def test_get_conversation_stats(provider: SQLModelProvider, sample_data: None):
    """Test statistics retrieval and aggregation."""
    cutoff = BASE_TIME - timedelta(hours=3)
    filters = StatsFilters(cutoff=cutoff, group_by="model")
    stats = await provider.get_conversation_stats(filters)

    # Check model grouping
    assert "gpt-5" in stats
    assert stats["gpt-5"]["messages"] == 2  # noqa: PLR2004
    assert stats["gpt-5"]["total_tokens"] == 30  # noqa: PLR2004
    assert "gpt-3.5-turbo" in stats
    assert stats["gpt-3.5-turbo"]["messages"] == 1


async def test_complex_filtering(provider: SQLModelProvider, sample_data: None):
    """Test combined filtering capabilities."""
    since = BASE_TIME - timedelta(hours=1.5)
    filters = QueryFilters(
        agent_name="test_agent", model="gpt-5", since=since, query="Hello"
    )
    results = await provider.get_conversations(filters)
    assert len(results) == 1
    conv, msgs = results[0]
    assert conv["agent"] == "test_agent"
    assert any(msg.content == "Hello" for msg in msgs)
    assert all(msg.model_name == "gpt-5" for msg in msgs)


async def test_basic_filters(provider: SQLModelProvider, sample_data: None):
    """Test basic filtering by agent and model."""
    # Get all conversations
    filters = QueryFilters()
    results = await provider.get_conversations(filters)
    assert len(results) == 2  # noqa: PLR2004

    # Filter by agent
    filters = QueryFilters(agent_name="test_agent")
    results = await provider.get_conversations(filters)
    assert len(results) == 1
    conv, msgs = results[0]
    assert conv["agent"] == "test_agent"
    assert len(msgs) == 2  # noqa: PLR2004

    # Filter by model
    filters = QueryFilters(model="gpt-5")
    results = await provider.get_conversations(filters)
    assert len(results) == 1
    conv, msgs = results[0]
    assert all(msg.model_name == "gpt-5" for msg in msgs)


async def test_time_filters(provider: SQLModelProvider, sample_data: None):
    """Test time-based filtering."""
    # First find conversation start times
    async with AsyncSession(async_engine) as session:
        result = await session.execute(
            select(Conversation).order_by(Conversation.start_time.desc())
        )
        conv = result.scalars().first()
        latest_conv_time = conv.start_time  # type:ignore

    # Get all conversations (no time filter)
    filters = QueryFilters()
    results = await provider.get_conversations(filters)
    assert len(results) == 2  # All conversations  # noqa: PLR2004

    # Filter with cutoff after latest conversation - should get nothing
    filters = QueryFilters(since=latest_conv_time + timedelta(seconds=1))
    results = await provider.get_conversations(filters)
    assert len(results) == 0  # No conversations after cutoff

    # Filter with cutoff before latest conversation - should get conversations
    filters = QueryFilters(since=latest_conv_time - timedelta(hours=1))
    results = await provider.get_conversations(filters)
    assert len(results) > 0


async def test_filtered_conversations(provider: SQLModelProvider, sample_data: None):
    """Test high-level filtered conversation helper."""
    # Filter by agent (should get all messages for test_agent)
    results = await provider.get_filtered_conversations(
        agent_name="test_agent", include_tokens=True
    )
    assert len(results) == 1
    conv = results[0]
    assert conv["agent"] == "test_agent"
    assert len(conv["messages"]) == 2  # noqa: PLR2004
    assert conv["token_usage"] is not None
    assert conv["token_usage"]["total"] == 30  # 10 + 20 tokens  # noqa: PLR2004


async def test_period_filtering(provider: SQLModelProvider, sample_data: None):
    """Test just the period filtering to isolate the issue."""
    # Test direct period parsing
    period = "2h"
    since = get_now() - parse_time_period(period)
    print(f"\nPeriod '2h' parsed to: {since}")

    # Test with QueryFilters
    filters = QueryFilters(since=since)
    results = await provider.get_conversations(filters)
    print(f"\nGot {len(results)} conversations with since={since}")

    # Print all conversations and their times
    async with AsyncSession(async_engine) as session:
        result = await session.execute(select(Conversation))
        convs = result.scalars().all()
        for conv in convs:
            print(f"Conv {conv.id}: start_time={conv.start_time}, since={since}")
            print(
                f"Comparison: {conv.start_time} >= {since} = {conv.start_time >= since}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
