from __future__ import annotations

from evented.configs import TimeEventConfig
from evented.event_data import EventData
import pytest

from agentpool.messaging import EventManager
from agentpool.utils.now import get_now


@pytest.fixture
def event_manager():
    """Create event manager for testing."""
    return EventManager(enable_events=True)


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


async def test_event_manager_disabled():
    """Test that disabled event manager doesn't emit events."""
    manager = EventManager(enable_events=False)
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
        schedule="* * * * *", prompt="Test", name="test_timer"
    )

    # Verify source was created and configured correctly
    assert source.config.name == "test_timer"
    assert source.config.schedule == "* * * * *"
    assert source.config.prompt == "Test"
    assert "test_timer" in event_manager._sources
    await event_manager.__aexit__(None, None, None)


async def test_event_callback_receives_prompt():
    """Test that event callbacks can handle prompts like auto_run did."""
    received_prompts: list[str] = []

    async def prompt_handler(event: EventData) -> None:
        if prompt := event.to_prompt():
            received_prompts.append(prompt)

    manager = EventManager(event_callbacks=[prompt_handler], enable_events=True)
    event = _TestEvent(source="test", timestamp=get_now(), message="Test")
    await manager.emit_event(event)
    assert len(received_prompts) == 1
    assert received_prompts[0] == "Test"


async def test_event_manager_cleanup(event_manager: EventManager):
    """Test cleanup of event manager."""
    # Add a simple event source
    config = TimeEventConfig(name="test_timer", schedule="* * * * *", prompt="Test")
    await event_manager.add_source(config)
    assert len(event_manager._sources) == 1
    await event_manager.__aexit__(None, None, None)
    assert len(event_manager._sources) == 0


async def test_event_manager_async_context():
    """Test async context management."""
    async with EventManager() as manager:
        assert manager.enabled


async def test_event_manager_with_initial_callbacks():
    """Test creating event manager with callbacks passed at init."""
    events: list[EventData] = []

    async def callback(event: EventData) -> None:
        events.append(event)

    manager = EventManager(event_callbacks=[callback], enable_events=True)
    event = _TestEvent(source="test", message="hello")
    await manager.emit_event(event)
    assert len(events) == 1
    assert isinstance(events[0], _TestEvent)
    assert events[0].message == "hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
