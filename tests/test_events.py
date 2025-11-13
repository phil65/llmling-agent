from __future__ import annotations

from evented.configs import TimeEventConfig
from evented.event_data import EventData
import pytest

from llmling_agent.messaging import ChatMessage, EventManager
from llmling_agent.utils.now import get_now


@pytest.fixture
def agent():
    """Mock agent for testing."""
    from llmling_agent import Agent

    return Agent(name="test_agent")


@pytest.fixture
def event_manager(agent):
    """Create event manager for testing."""
    return EventManager(agent, enable_events=True)


class _TestEvent(EventData):
    """Simple event type for testing."""

    message: str = ""

    def to_prompt(self) -> str:
        """Convert event to prompt format."""
        return self.message


async def test_event_manager_basic_callback(event_manager: EventManager):
    """Test basic callback registration and event emission."""
    received_events = []

    async def test_callback(event):
        received_events.append(event)

    event_manager.add_callback(test_callback)
    event = _TestEvent(source="test", message="test message")
    await event_manager.emit_event(event)

    assert len(received_events) == 1
    assert received_events[0].source == "test"
    assert received_events[0].message == "test message"


async def test_event_manager_multiple_callbacks(event_manager: EventManager):
    """Test multiple callbacks can receive same event."""
    counter1 = 0
    counter2 = 0

    async def callback1(event):
        nonlocal counter1
        counter1 += 1

    async def callback2(event):
        nonlocal counter2
        counter2 += 1

    event_manager.add_callback(callback1)
    event_manager.add_callback(callback2)

    event = _TestEvent(source="test")
    await event_manager.emit_event(event)

    assert counter1 == 1
    assert counter2 == 1


async def test_event_manager_disabled(agent):
    """Test that disabled event manager doesn't emit events."""
    manager = EventManager(agent, enable_events=False)
    counter = 0

    async def callback(event):
        nonlocal counter
        counter += 1

    manager.add_callback(callback)
    event = _TestEvent(source="test")
    await manager.emit_event(event)

    assert counter == 0


async def test_event_manager_remove_callback(event_manager: EventManager):
    """Test callback removal."""
    counter = 0

    async def callback(event):
        nonlocal counter
        counter += 1

    event_manager.add_callback(callback)
    event = _TestEvent(source="test")
    await event_manager.emit_event(event)
    assert counter == 1

    event_manager.remove_callback(callback)
    await event_manager.emit_event(event)
    assert counter == 1  # Shouldn't have increased


async def test_timed_event_basic(event_manager: EventManager):
    """Test basic timed event setup."""
    events_received = []

    async def callback(event):
        events_received.append(event)

    event_manager.add_callback(callback)

    # Add timed event through public API
    source = await event_manager.add_timed_event(
        schedule="* * * * *",  # every minute
        prompt="Test prompt",
        name="test_timer",
    )

    # Verify source was created and configured correctly
    assert source.config.name == "test_timer"
    assert source.config.schedule == "* * * * *"
    assert source.config.prompt == "Test prompt"
    assert "test_timer" in event_manager._sources
    await event_manager.cleanup()


async def test_auto_run_handling(event_manager, agent):
    """Test auto_run feature with message handling."""
    received_messages = []

    # Mock agent's run method
    async def mock_run(*args, **kwargs):
        received_messages.append(args[0])
        return ChatMessage(content="response", role="assistant")

    agent.run = mock_run

    # Create and emit event with a prompt
    event = _TestEvent(
        source="test",
        timestamp=get_now(),
        message="Test prompt",
    )
    await event_manager.emit_event(event)

    assert len(received_messages) == 1
    assert received_messages[0] == "Test prompt"


async def test_event_manager_cleanup(event_manager: EventManager):
    """Test cleanup of event manager."""
    # Add a simple event source
    config = TimeEventConfig(
        name="test_timer", schedule="* * * * *", prompt="Test prompt"
    )
    await event_manager.add_source(config)

    assert len(event_manager._sources) == 1
    await event_manager.cleanup()
    assert len(event_manager._sources) == 0


async def test_event_manager_async_context(agent):
    """Test async context management."""
    async with EventManager(agent) as manager:
        assert manager.enabled
        assert manager.node == agent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
