"""Integration tests for conversation title generation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import AgentPool
from llmling_agent.models.agents import AgentConfig
from llmling_agent.models.manifest import AgentsManifest
from llmling_agent.sessions import SessionManager
from llmling_agent.storage.manager import StorageManager
from llmling_agent_config.storage import MemoryStorageConfig, StorageConfig


if TYPE_CHECKING:
    from llmling_agent_config.storage import SQLStorageConfig
    from llmling_agent_storage.models import ConversationData


GENERATED_TITLE = "Test Conversation Title"


@pytest.fixture
def title_model() -> TestModel:
    """Model for title generation that returns a fixed title."""
    return TestModel(custom_output_text=GENERATED_TITLE)


@pytest.fixture
def storage_config() -> StorageConfig:
    """Storage config with memory provider and title generation enabled."""
    return StorageConfig(
        providers=[MemoryStorageConfig()],
        title_generation_model="test",  # Will be overridden in tests
    )


@pytest.fixture
async def pool_with_storage(storage_config: StorageConfig):
    """Create agent pool with storage configured."""
    manifest = AgentsManifest(
        agents={
            "test_agent": AgentConfig(
                name="test_agent",
                model="test",
                system_prompts=["You are a test agent"],
            ),
        },
        storage=storage_config,
    )
    pool = AgentPool(manifest=manifest)
    async with pool:
        yield pool


class TestStorageManagerTitleGeneration:
    """Tests for StorageManager title generation methods."""

    async def test_generate_title_stores_title(self) -> None:
        """Test that generate_conversation_title generates and stores a title."""
        from llmling_agent.messaging import ChatMessage

        config = StorageConfig(
            providers=[MemoryStorageConfig()],
            title_generation_model="test",
        )
        async with StorageManager(config) as manager:
            # Create a conversation first
            conv_id = "test_conv_123"
            await manager.log_conversation(
                conversation_id=conv_id,
                node_name="test_agent",
            )

            # Create test messages
            messages = [
                ChatMessage.user_prompt("What is the weather today?"),
                ChatMessage(
                    content="The weather is sunny with a high of 75°F.",
                    role="assistant",
                ),
            ]

            # Generate title (using TestModel via pydantic-ai)
            title = await manager.generate_conversation_title(conv_id, messages)

            # Title should be generated
            assert title is not None
            assert len(title) > 0

            # Title should be stored
            stored_title = await manager.get_conversation_title(conv_id)
            assert stored_title == title

    async def test_generate_title_disabled(self) -> None:
        """Test that title generation is skipped when model is None."""
        from llmling_agent.messaging import ChatMessage

        config = StorageConfig(
            providers=[MemoryStorageConfig()],
            title_generation_model=None,
        )
        async with StorageManager(config) as manager:
            conv_id = "test_conv_456"
            await manager.log_conversation(
                conversation_id=conv_id,
                node_name="test_agent",
            )

            messages = [
                ChatMessage.user_prompt("Hello"),
                ChatMessage(content="Hi there!", role="assistant"),
            ]

            title = await manager.generate_conversation_title(conv_id, messages)
            assert title is None

    async def test_generate_title_already_exists(self) -> None:
        """Test that existing title is returned without regenerating."""
        from llmling_agent.messaging import ChatMessage

        config = StorageConfig(
            providers=[MemoryStorageConfig()],
            title_generation_model="test",
        )
        async with StorageManager(config) as manager:
            conv_id = "test_conv_789"
            await manager.log_conversation(
                conversation_id=conv_id,
                node_name="test_agent",
            )

            # Set existing title
            existing_title = "Existing Title"
            await manager.update_conversation_title(conv_id, existing_title)

            messages = [
                ChatMessage.user_prompt("New message"),
                ChatMessage(content="New response", role="assistant"),
            ]

            # Should return existing title without calling model
            title = await manager.generate_conversation_title(conv_id, messages)
            assert title == existing_title

    async def test_update_and_get_title(self) -> None:
        """Test updating and retrieving conversation title."""
        config = StorageConfig(providers=[MemoryStorageConfig()])
        async with StorageManager(config) as manager:
            conv_id = "test_conv_update"
            await manager.log_conversation(
                conversation_id=conv_id,
                node_name="test_agent",
            )

            # Initially no title
            title = await manager.get_conversation_title(conv_id)
            assert title is None

            # Update title
            await manager.update_conversation_title(conv_id, "My Title")

            # Verify title was stored
            title = await manager.get_conversation_title(conv_id)
            assert title == "My Title"


class TestClientSessionTitleGeneration:
    """Tests for ClientSession title generation integration."""

    async def test_title_generation_flag_initially_false(
        self,
        pool_with_storage: AgentPool,
    ) -> None:
        """Test that title generation flag is initially False."""
        manager = SessionManager(pool_with_storage)

        async with manager:
            session = await manager.create(
                agent_name="test_agent",
                conversation_id="conv_flag_test",
            )
            # Flag should be False before any run
            assert not session._title_generation_triggered

    async def test_title_generation_flag_set_after_run(
        self,
        pool_with_storage: AgentPool,
    ) -> None:
        """Test that title generation flag is set after first run."""
        manager = SessionManager(pool_with_storage)

        async with manager:
            session = await manager.create(
                agent_name="test_agent",
                conversation_id="conv_title_flag",
            )

            # Log the conversation to storage first
            if pool_with_storage.storage:
                await pool_with_storage.storage.log_conversation(
                    conversation_id=session.conversation_id,
                    node_name="test_agent",
                )

            # Run the agent - this should trigger title generation
            await session.run("Hello, how are you?")

            # Flag should be set after run
            assert session._title_generation_triggered

            # Wait briefly for background task
            await asyncio.sleep(0.1)

    async def test_title_generation_flag_stays_true(
        self,
        pool_with_storage: AgentPool,
    ) -> None:
        """Test that title generation flag remains True after multiple runs."""
        manager = SessionManager(pool_with_storage)

        async with manager:
            session = await manager.create(
                agent_name="test_agent",
                conversation_id="conv_flag_stays",
            )

            if pool_with_storage.storage:
                await pool_with_storage.storage.log_conversation(
                    conversation_id=session.conversation_id,
                    node_name="test_agent",
                )

            # First run sets the flag
            await session.run("First message")
            assert session._title_generation_triggered

            # Second run should keep flag True
            await session.run("Second message")
            assert session._title_generation_triggered

    async def test_generate_title_method_directly(self) -> None:
        """Test the _generate_title method directly via StorageManager."""
        from llmling_agent.messaging import ChatMessage

        config = StorageConfig(
            providers=[MemoryStorageConfig()],
            title_generation_model="test",
        )

        async with StorageManager(config) as storage:
            conv_id = "direct_title_test"
            await storage.log_conversation(
                conversation_id=conv_id,
                node_name="test_agent",
            )

            # Create test messages
            messages = [
                ChatMessage.user_prompt("What is Python?"),
                ChatMessage(
                    content="Python is a programming language.",
                    role="assistant",
                ),
            ]

            # Generate title
            title = await storage.generate_conversation_title(conv_id, messages)
            assert title is not None

            # Verify it was stored
            stored = await storage.get_conversation_title(conv_id)
            assert stored == title


class TestMemoryProviderTitleSupport:
    """Tests for MemoryStorageProvider title methods."""

    async def test_memory_provider_title_operations(self) -> None:
        """Test title update and get on memory provider."""
        from llmling_agent_config.storage import MemoryStorageConfig
        from llmling_agent_storage.memory_provider import MemoryStorageProvider

        config = MemoryStorageConfig()
        provider = MemoryStorageProvider(config)

        conv_id = "mem_conv_123"
        await provider.log_conversation(
            conversation_id=conv_id,
            node_name="test_agent",
        )

        # Initially no title
        title = await provider.get_conversation_title(conv_id)
        assert title is None

        # Update title
        await provider.update_conversation_title(conv_id, "Memory Title")

        # Get title
        title = await provider.get_conversation_title(conv_id)
        assert title == "Memory Title"

    async def test_memory_provider_title_nonexistent_conv(self) -> None:
        """Test getting title for non-existent conversation."""
        from llmling_agent_config.storage import MemoryStorageConfig
        from llmling_agent_storage.memory_provider import MemoryStorageProvider

        config = MemoryStorageConfig()
        provider = MemoryStorageProvider(config)

        title = await provider.get_conversation_title("nonexistent")
        assert title is None


class TestSQLProviderTitleSupport:
    """Tests for SQLModelProvider title methods."""

    @pytest.fixture
    def sql_config(self, tmp_path) -> SQLStorageConfig:
        """Create SQL config with temp database."""
        from llmling_agent_config.storage import SQLStorageConfig

        db_path = tmp_path / "test_titles.db"
        return SQLStorageConfig(url=f"sqlite:///{db_path}")

    async def test_sql_provider_title_operations(self, sql_config) -> None:
        """Test title update and get on SQL provider."""
        from llmling_agent_storage.sql_provider import SQLModelProvider

        async with SQLModelProvider(sql_config) as provider:
            conv_id = "sql_conv_123"
            await provider.log_conversation(
                conversation_id=conv_id,
                node_name="test_agent",
            )

            # Initially no title
            title = await provider.get_conversation_title(conv_id)
            assert title is None

            # Update title
            await provider.update_conversation_title(conv_id, "SQL Title")

            # Get title
            title = await provider.get_conversation_title(conv_id)
            assert title == "SQL Title"

    async def test_sql_provider_title_nonexistent_conv(self, sql_config) -> None:
        """Test getting title for non-existent conversation."""
        from llmling_agent_storage.sql_provider import SQLModelProvider

        async with SQLModelProvider(sql_config) as provider:
            title = await provider.get_conversation_title("nonexistent")
            assert title is None

    async def test_sql_provider_title_update_overwrites(self, sql_config) -> None:
        """Test that updating title overwrites previous value."""
        from llmling_agent_storage.sql_provider import SQLModelProvider

        async with SQLModelProvider(sql_config) as provider:
            conv_id = "sql_conv_overwrite"
            await provider.log_conversation(
                conversation_id=conv_id,
                node_name="test_agent",
            )

            await provider.update_conversation_title(conv_id, "First Title")
            await provider.update_conversation_title(conv_id, "Second Title")

            title = await provider.get_conversation_title(conv_id)
            assert title == "Second Title"


class TestFileProviderTitleSupport:
    """Tests for FileProvider title methods."""

    async def test_file_provider_title_operations(self, tmp_path) -> None:
        """Test title update and get on file provider."""
        from llmling_agent_config.storage import FileStorageConfig
        from llmling_agent_storage.file_provider import FileProvider

        storage_file = tmp_path / "storage.json"
        config = FileStorageConfig(path=str(storage_file))
        provider = FileProvider(config)

        conv_id = "file_conv_123"
        await provider.log_conversation(
            conversation_id=conv_id,
            node_name="test_agent",
        )

        # Initially no title
        title = await provider.get_conversation_title(conv_id)
        assert title is None

        # Update title
        await provider.update_conversation_title(conv_id, "File Title")

        # Get title
        title = await provider.get_conversation_title(conv_id)
        assert title == "File Title"

        # Verify file was written
        assert storage_file.exists()


class TestConversationDataTitle:
    """Tests for ConversationData title field."""

    def test_conversation_data_has_title(self) -> None:
        """Test that ConversationData TypedDict includes title field."""
        # Create ConversationData with title
        data: ConversationData = {
            "id": "conv_123",
            "agent": "test_agent",
            "title": "My Conversation",
            "start_time": "2024-01-01T00:00:00",
            "messages": [],
            "token_usage": None,
        }
        assert data["title"] == "My Conversation"

    def test_conversation_data_title_none(self) -> None:
        """Test that ConversationData can have None title."""
        data: ConversationData = {
            "id": "conv_456",
            "agent": "test_agent",
            "title": None,
            "start_time": "2024-01-01T00:00:00",
            "messages": [],
            "token_usage": None,
        }
        assert data["title"] is None
