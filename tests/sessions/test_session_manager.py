"""Tests for the session management infrastructure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from llmling_agent.delegation.pool import AgentPool
from llmling_agent.models.agents import NativeAgentConfig
from llmling_agent.models.manifest import AgentsManifest
from llmling_agent.sessions import ClientSession, SessionData, SessionManager
from llmling_agent.sessions.store import MemorySessionStore
from llmling_agent_config.storage import SQLStorageConfig
from llmling_agent_storage.session_store import SQLSessionStore


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def memory_store() -> MemorySessionStore:
    """Create a memory session store for testing."""
    return MemorySessionStore()


@pytest.fixture
async def agent_pool():
    """Create a real agent pool for testing."""
    manifest = AgentsManifest(
        agents={
            "test_agent": NativeAgentConfig(
                name="test_agent",
                model="test",
                system_prompts=["You are a test agent"],
            ),
            "other_agent": NativeAgentConfig(
                name="other_agent",
                model="test",
                system_prompts=["You are another test agent"],
            ),
        }
    )
    pool = AgentPool(manifest=manifest)
    async with pool:
        yield pool


class TestSessionData:
    """Tests for SessionData model."""

    def test_session_data_creation(self) -> None:
        """Test basic session data creation."""
        data = SessionData(
            session_id="test_session",
            agent_name="test_agent",
            conversation_id="conv_123",
        )
        assert data.session_id == "test_session"
        assert data.agent_name == "test_agent"
        assert data.conversation_id == "conv_123"
        assert data.pool_id is None
        assert data.cwd is None
        assert data.metadata == {}

    def test_session_data_with_metadata(self) -> None:
        """Test session data with metadata."""
        data = SessionData(
            session_id="test_session",
            agent_name="test_agent",
            conversation_id="conv_123",
            metadata={"protocol": "acp", "version": "1.0"},
        )
        assert data.metadata["protocol"] == "acp"
        assert data.metadata["version"] == "1.0"

    def test_with_agent(self) -> None:
        """Test creating a copy with different agent."""
        original = SessionData(
            session_id="test_session",
            agent_name="agent1",
            conversation_id="conv_123",
        )
        updated = original.with_agent("agent2")

        assert updated.agent_name == "agent2"
        assert updated.session_id == original.session_id
        assert updated.conversation_id == original.conversation_id
        # Original should be unchanged
        assert original.agent_name == "agent1"

    def test_with_metadata(self) -> None:
        """Test creating a copy with updated metadata."""
        original = SessionData(
            session_id="test_session",
            agent_name="test_agent",
            conversation_id="conv_123",
            metadata={"key1": "value1"},
        )
        updated = original.with_metadata(key2="value2")

        assert updated.metadata["key1"] == "value1"
        assert updated.metadata["key2"] == "value2"
        # Original should be unchanged
        assert "key2" not in original.metadata


class TestMemorySessionStore:
    """Tests for MemorySessionStore."""

    async def test_save_and_load(self, memory_store: MemorySessionStore) -> None:
        """Test saving and loading a session."""
        data = SessionData(
            session_id="test_session",
            agent_name="test_agent",
            conversation_id="conv_123",
        )

        async with memory_store:
            await memory_store.save(data)
            loaded = await memory_store.load("test_session")

        assert loaded is not None
        assert loaded.session_id == data.session_id
        assert loaded.agent_name == data.agent_name

    async def test_load_nonexistent(self, memory_store: MemorySessionStore) -> None:
        """Test loading a nonexistent session returns None."""
        async with memory_store:
            loaded = await memory_store.load("nonexistent")

        assert loaded is None

    async def test_delete(self, memory_store: MemorySessionStore) -> None:
        """Test deleting a session."""
        data = SessionData(
            session_id="test_session",
            agent_name="test_agent",
            conversation_id="conv_123",
        )

        async with memory_store:
            await memory_store.save(data)
            deleted = await memory_store.delete("test_session")
            assert deleted is True

            loaded = await memory_store.load("test_session")
            assert loaded is None

    async def test_delete_nonexistent(self, memory_store: MemorySessionStore) -> None:
        """Test deleting a nonexistent session returns False."""
        async with memory_store:
            deleted = await memory_store.delete("nonexistent")

        assert deleted is False

    async def test_list_sessions(self, memory_store: MemorySessionStore) -> None:
        """Test listing sessions."""
        data1 = SessionData(
            session_id="session1",
            agent_name="agent1",
            conversation_id="conv_1",
            pool_id="pool1",
        )
        data2 = SessionData(
            session_id="session2",
            agent_name="agent2",
            conversation_id="conv_2",
            pool_id="pool1",
        )
        data3 = SessionData(
            session_id="session3",
            agent_name="agent1",
            conversation_id="conv_3",
            pool_id="pool2",
        )

        async with memory_store:
            await memory_store.save(data1)
            await memory_store.save(data2)
            await memory_store.save(data3)

            # List all
            all_sessions = await memory_store.list_sessions()
            expected_total = 3
            assert len(all_sessions) == expected_total

            # Filter by pool_id
            pool1_sessions = await memory_store.list_sessions(pool_id="pool1")
            expected_pool1 = 2
            assert len(pool1_sessions) == expected_pool1
            assert "session1" in pool1_sessions
            assert "session2" in pool1_sessions

            # Filter by agent_name
            agent1_sessions = await memory_store.list_sessions(agent_name="agent1")
            expected_agent1 = 2
            assert len(agent1_sessions) == expected_agent1
            assert "session1" in agent1_sessions
            assert "session3" in agent1_sessions

    async def test_update_existing(self, memory_store: MemorySessionStore) -> None:
        """Test updating an existing session."""
        original = SessionData(
            session_id="test_session",
            agent_name="agent1",
            conversation_id="conv_123",
        )

        async with memory_store:
            await memory_store.save(original)

            updated = original.with_agent("agent2")
            await memory_store.save(updated)

            loaded = await memory_store.load("test_session")

        assert loaded is not None
        assert loaded.agent_name == "agent2"


class TestClientSession:
    """Tests for ClientSession base class."""

    async def test_session_properties(self, agent_pool: AgentPool) -> None:
        """Test basic session properties."""
        data = SessionData(
            session_id="test_session",
            agent_name="test_agent",
            conversation_id="conv_123",
            cwd="/tmp/test",
        )

        session = ClientSession(data=data, pool=agent_pool)

        assert session.session_id == "test_session"
        assert session.agent_name == "test_agent"
        assert session.conversation_id == "conv_123"
        assert not session.is_closed

    async def test_session_close(self, agent_pool: AgentPool) -> None:
        """Test closing a session."""
        data = SessionData(
            session_id="test_session",
            agent_name="test_agent",
            conversation_id="conv_123",
        )

        session = ClientSession(data=data, pool=agent_pool)
        assert not session.is_closed

        await session.close()
        assert session.is_closed

        # Closing again should be idempotent
        await session.close()
        assert session.is_closed

    async def test_session_context_manager(self, agent_pool: AgentPool) -> None:
        """Test session as async context manager."""
        data = SessionData(
            session_id="test_session",
            agent_name="test_agent",
            conversation_id="conv_123",
        )

        async with ClientSession(data=data, pool=agent_pool) as session:
            assert not session.is_closed

        assert session.is_closed

    async def test_update_metadata(self, agent_pool: AgentPool) -> None:
        """Test updating session metadata."""
        data = SessionData(
            session_id="test_session",
            agent_name="test_agent",
            conversation_id="conv_123",
            metadata={"key1": "value1"},
        )

        session = ClientSession(data=data, pool=agent_pool)
        session.update_metadata(key2="value2")

        assert session.data.metadata["key1"] == "value1"
        assert session.data.metadata["key2"] == "value2"


class TestSQLSessionStore:
    """Tests for SQLSessionStore."""

    @pytest.fixture
    def sql_store(self, tmp_path: Path) -> SQLSessionStore:
        """Create a SQL session store with temp database."""
        db_path = tmp_path / "test_sessions.db"
        config = SQLStorageConfig(url=f"sqlite:///{db_path}")
        return SQLSessionStore(config)

    async def test_save_and_load(self, sql_store: SQLSessionStore) -> None:
        """Test saving and loading a session."""
        data = SessionData(
            session_id="sql_test_session",
            agent_name="test_agent",
            conversation_id="conv_123",
            cwd="/tmp/test",
            metadata={"protocol": "acp"},
        )

        async with sql_store:
            await sql_store.save(data)
            loaded = await sql_store.load("sql_test_session")

        assert loaded is not None
        assert loaded.session_id == data.session_id
        assert loaded.agent_name == data.agent_name
        assert loaded.cwd == data.cwd
        assert loaded.metadata["protocol"] == "acp"

    async def test_load_nonexistent(self, sql_store: SQLSessionStore) -> None:
        """Test loading a nonexistent session returns None."""
        async with sql_store:
            loaded = await sql_store.load("nonexistent")

        assert loaded is None

    async def test_delete(self, sql_store: SQLSessionStore) -> None:
        """Test deleting a session."""
        data = SessionData(
            session_id="delete_test",
            agent_name="test_agent",
            conversation_id="conv_123",
        )

        async with sql_store:
            await sql_store.save(data)
            deleted = await sql_store.delete("delete_test")
            assert deleted is True

            loaded = await sql_store.load("delete_test")
            assert loaded is None

    async def test_update_existing(self, sql_store: SQLSessionStore) -> None:
        """Test updating an existing session (upsert)."""
        original = SessionData(
            session_id="update_test",
            agent_name="agent1",
            conversation_id="conv_123",
        )

        async with sql_store:
            await sql_store.save(original)

            updated = original.with_agent("agent2")
            await sql_store.save(updated)

            loaded = await sql_store.load("update_test")

        assert loaded is not None
        assert loaded.agent_name == "agent2"

    async def test_list_sessions(self, sql_store: SQLSessionStore) -> None:
        """Test listing sessions with filters."""
        data1 = SessionData(
            session_id="sql_session1",
            agent_name="agent1",
            conversation_id="conv_1",
            pool_id="pool1",
        )
        data2 = SessionData(
            session_id="sql_session2",
            agent_name="agent2",
            conversation_id="conv_2",
            pool_id="pool1",
        )

        async with sql_store:
            await sql_store.save(data1)
            await sql_store.save(data2)

            all_sessions = await sql_store.list_sessions()
            expected_total = 2
            assert len(all_sessions) == expected_total

            pool1_sessions = await sql_store.list_sessions(pool_id="pool1")
            assert len(pool1_sessions) == expected_total

            agent1_sessions = await sql_store.list_sessions(agent_name="agent1")
            assert len(agent1_sessions) == 1
            assert "sql_session1" in agent1_sessions

    async def test_get_all(self, sql_store: SQLSessionStore) -> None:
        """Test getting all session data objects."""
        data1 = SessionData(
            session_id="all_test_1",
            agent_name="agent1",
            conversation_id="conv_1",
        )
        data2 = SessionData(
            session_id="all_test_2",
            agent_name="agent2",
            conversation_id="conv_2",
        )

        async with sql_store:
            await sql_store.save(data1)
            await sql_store.save(data2)

            all_data = await sql_store.get_all()

        expected_count = 2
        assert len(all_data) == expected_count
        session_ids = {d.session_id for d in all_data}
        assert "all_test_1" in session_ids
        assert "all_test_2" in session_ids


class TestSessionManager:
    """Tests for SessionManager."""

    async def test_create_session(self, agent_pool: AgentPool) -> None:
        """Test creating a session through the manager."""
        manager = SessionManager(pool=agent_pool)

        async with manager:
            session = await manager.create(agent_name="test_agent")

            assert session.agent_name == "test_agent"
            assert session.session_id.startswith("sess_")
            assert not session.is_closed

    async def test_create_session_with_custom_id(self, agent_pool: AgentPool) -> None:
        """Test creating a session with a specific ID."""
        manager = SessionManager(pool=agent_pool)

        async with manager:
            session = await manager.create(
                agent_name="test_agent",
                session_id="custom_id_123",
            )

            assert session.session_id == "custom_id_123"

    async def test_create_duplicate_session_raises(self, agent_pool: AgentPool) -> None:
        """Test that creating a session with existing ID raises."""
        manager = SessionManager(pool=agent_pool)

        async with manager:
            await manager.create(agent_name="test_agent", session_id="duplicate_test")

            with pytest.raises(ValueError, match="already exists"):
                await manager.create(agent_name="test_agent", session_id="duplicate_test")

    async def test_create_session_unknown_agent_raises(self, agent_pool: AgentPool) -> None:
        """Test that creating a session with unknown agent raises."""
        manager = SessionManager(pool=agent_pool)

        async with manager:
            with pytest.raises(KeyError):
                await manager.create(agent_name="unknown_agent")

    async def test_get_session(self, agent_pool: AgentPool) -> None:
        """Test getting an active session."""
        manager = SessionManager(pool=agent_pool)

        async with manager:
            created = await manager.create(agent_name="test_agent")
            retrieved = await manager.get(created.session_id)

            assert retrieved is not None
            assert retrieved.session_id == created.session_id

    async def test_get_nonexistent_session(self, agent_pool: AgentPool) -> None:
        """Test getting a nonexistent session returns None."""
        manager = SessionManager(pool=agent_pool)

        async with manager:
            retrieved = await manager.get("nonexistent")

        assert retrieved is None

    async def test_close_session(self, agent_pool: AgentPool) -> None:
        """Test closing a session."""
        manager = SessionManager(pool=agent_pool)

        async with manager:
            session = await manager.create(agent_name="test_agent")
            session_id = session.session_id

            closed = await manager.close(session_id)
            assert closed is True

            # Session should no longer be active
            retrieved = await manager.get(session_id)
            assert retrieved is None

    async def test_list_active_sessions(self, agent_pool: AgentPool) -> None:
        """Test listing active sessions."""
        manager = SessionManager(pool=agent_pool)

        async with manager:
            await manager.create(agent_name="test_agent", session_id="list_test_1")
            await manager.create(agent_name="other_agent", session_id="list_test_2")

            active = await manager.list_sessions(active_only=True)
            expected_active = 2
            assert len(active) == expected_active
            assert "list_test_1" in active
            assert "list_test_2" in active

    async def test_list_sessions_by_agent(self, agent_pool: AgentPool) -> None:
        """Test listing sessions filtered by agent name."""
        manager = SessionManager(pool=agent_pool)

        async with manager:
            await manager.create(agent_name="test_agent", session_id="agent_filter_1")
            await manager.create(agent_name="other_agent", session_id="agent_filter_2")
            await manager.create(agent_name="test_agent", session_id="agent_filter_3")

            test_agent_sessions = await manager.list_sessions(
                active_only=True, agent_name="test_agent"
            )
            expected_test_agent = 2
            assert len(test_agent_sessions) == expected_test_agent
            assert "agent_filter_1" in test_agent_sessions
            assert "agent_filter_3" in test_agent_sessions

    async def test_resume_session(self, agent_pool: AgentPool) -> None:
        """Test resuming a session from storage."""
        manager = SessionManager(pool=agent_pool)

        async with manager:
            # Create and close a session
            await manager.create(agent_name="test_agent", session_id="resume_test")
            await manager.close("resume_test")

            # Session should not be active
            assert await manager.get("resume_test") is None

            # Resume it
            resumed = await manager.resume("resume_test")

            assert resumed is not None
            assert resumed.session_id == "resume_test"
            assert resumed.agent_name == "test_agent"

    async def test_context_manager_cleanup(self, agent_pool: AgentPool) -> None:
        """Test that context manager closes all sessions."""
        manager = SessionManager(pool=agent_pool)

        async with manager:
            await manager.create(agent_name="test_agent", session_id="cleanup_1")
            await manager.create(agent_name="other_agent", session_id="cleanup_2")

        # After exiting context, sessions should be cleaned up
        # This is verified by the manager closing properly without errors
