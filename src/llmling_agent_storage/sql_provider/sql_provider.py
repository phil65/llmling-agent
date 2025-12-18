"""SQLModel-based storage provider."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Self

from pydantic_ai import RunUsage
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, desc, select

from llmling_agent.log import get_logger
from llmling_agent.messaging import TokenCost
from llmling_agent.utils.now import get_now
from llmling_agent.utils.parse_time import parse_time_period
from llmling_agent_storage.base import StorageProvider
from llmling_agent_storage.models import QueryFilters
from llmling_agent_storage.sql_provider.models import CommandHistory, Conversation, Message
from llmling_agent_storage.sql_provider.utils import (
    build_message_query,
    format_conversation,
    parse_model_info,
    to_chat_message,
)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from types import TracebackType

    from sqlalchemy import Connection

    from llmling_agent.common_types import JsonValue
    from llmling_agent.messaging import ChatMessage
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.storage import SQLStorageConfig
    from llmling_agent_storage.models import ConversationData, StatsFilters


logger = get_logger(__name__)


class SQLModelProvider(StorageProvider):
    """Storage provider using SQLModel.

    Can work with any database supported by SQLAlchemy/SQLModel.
    Provides efficient SQL-based filtering and storage.
    """

    can_load_history = True

    def __init__(self, config: SQLStorageConfig) -> None:
        """Initialize provider with async database engine.

        Args:
            config: Configuration for provider
        """
        super().__init__(config)
        self.engine = config.get_engine()
        self.auto_migrate = config.auto_migration
        self.session: AsyncSession | None = None

    async def _init_database(self, auto_migrate: bool = True) -> None:
        """Initialize database tables and optionally migrate columns.

        Args:
            auto_migrate: Whether to automatically add missing columns
        """
        from sqlmodel import SQLModel

        from llmling_agent_storage.sql_provider.utils import auto_migrate_columns

        # Create tables if they don't exist
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        # Optionally add missing columns
        if auto_migrate:
            async with self.engine.begin() as conn:

                def sync_migrate(sync_conn: Connection) -> None:
                    auto_migrate_columns(sync_conn, self.engine.dialect)

                await conn.run_sync(sync_migrate)

    async def __aenter__(self) -> Self:
        """Initialize async database resources."""
        await self._init_database(auto_migrate=self.auto_migrate)
        await super().__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up async database resources properly."""
        await self.engine.dispose()
        return await super().__aexit__(exc_type, exc_val, exc_tb)

    def cleanup(self) -> None:
        """Clean up database resources."""
        # For sync cleanup, just pass - proper cleanup happens in __aexit__

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages using SQL queries."""
        async with AsyncSession(self.engine) as session:
            stmt = build_message_query(query)
            result = await session.execute(stmt)
            messages = result.scalars().all()
            return [to_chat_message(msg) for msg in messages]

    async def log_message(
        self,
        *,
        conversation_id: str,
        message_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: TokenCost | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
        provider_name: str | None = None,
        provider_response_id: str | None = None,
        messages: str | None = None,
        finish_reason: str | None = None,
    ) -> None:
        """Log message to database."""
        from llmling_agent_storage.sql_provider.models import Message

        provider, model_name = parse_model_info(model)

        async with AsyncSession(self.engine) as session:
            msg = Message(
                conversation_id=conversation_id,
                id=message_id,
                content=content,
                role=role,
                name=name,
                model=model,
                model_provider=provider,
                model_name=model_name,
                response_time=response_time,
                total_tokens=cost_info.token_usage.total_tokens if cost_info else None,
                input_tokens=cost_info.token_usage.input_tokens if cost_info else None,
                output_tokens=cost_info.token_usage.output_tokens if cost_info else None,
                cost=float(cost_info.total_cost) if cost_info else None,
                forwarded_from=forwarded_from,
                provider_name=provider_name,
                provider_response_id=provider_response_id,
                messages=messages,
                finish_reason=finish_reason,
                timestamp=get_now(),
            )
            session.add(msg)
            await session.commit()

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Log conversation to database."""
        from llmling_agent_storage.sql_provider.models import Conversation

        async with AsyncSession(self.engine) as session:
            now = start_time or get_now()
            convo = Conversation(id=conversation_id, agent_name=node_name, start_time=now)
            session.add(convo)
            await session.commit()

    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str,
    ) -> None:
        """Update the title of a conversation."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conversation = result.scalar_one_or_none()
            if conversation:
                conversation.title = title
                session.add(conversation)
                await session.commit()

    async def get_conversation_title(
        self,
        conversation_id: str,
    ) -> str | None:
        """Get the title of a conversation."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                select(Conversation.title).where(Conversation.id == conversation_id)
            )
            return result.scalar_one_or_none()

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        """Log command to database."""
        async with AsyncSession(self.engine) as session:
            history = CommandHistory(
                session_id=session_id,
                agent_name=agent_name,
                command=command,
                context_type=context_type.__name__ if context_type else None,
                context_metadata=metadata or {},
            )
            session.add(history)
            await session.commit()

    async def get_filtered_conversations(
        self,
        agent_name: str | None = None,
        period: str | None = None,
        since: datetime | None = None,
        query: str | None = None,
        model: str | None = None,
        limit: int | None = None,
        *,
        compact: bool = False,
        include_tokens: bool = False,
    ) -> list[ConversationData]:
        """Get filtered conversations with formatted output."""
        # Convert period to since if provided
        if period:
            since = get_now() - parse_time_period(period)

        # Create filters
        filters = QueryFilters(
            agent_name=agent_name,
            since=since,
            query=query,
            model=model,
            limit=limit,
        )

        # Use existing get_conversations method
        conversations = await self.get_conversations(filters)
        return [
            format_conversation(conv, msgs, compact=compact, include_tokens=include_tokens)
            for conv, msgs in conversations
        ]

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Get command history from database."""
        async with AsyncSession(self.engine) as session:
            query = select(CommandHistory)
            if current_session_only:
                query = query.where(CommandHistory.session_id == str(session_id))
            else:
                query = query.where(CommandHistory.agent_name == agent_name)

            query = query.order_by(desc(CommandHistory.timestamp))
            if limit:
                query = query.limit(limit)

            result = await session.execute(query)
            return [h.command for h in result.scalars()]

    async def get_conversations(
        self,
        filters: QueryFilters,
    ) -> list[tuple[ConversationData, Sequence[ChatMessage[str]]]]:
        """Get filtered conversations using SQL queries."""
        async with AsyncSession(self.engine) as session:
            results: list[tuple[ConversationData, Sequence[ChatMessage[str]]]] = []

            # Base conversation query
            conv_query = select(Conversation)

            if filters.agent_name:
                conv_query = conv_query.where(Conversation.agent_name == filters.agent_name)

            # Apply time filters if provided
            if filters.since:
                conv_query = conv_query.where(Conversation.start_time >= filters.since)

            if filters.limit:
                conv_query = conv_query.limit(filters.limit)

            conv_result = await session.execute(conv_query)
            conversations = conv_result.scalars().all()

            for conv in conversations:
                # Get messages for this conversation
                msg_query = select(Message).where(Message.conversation_id == conv.id)

                if filters.query:
                    msg_query = msg_query.where(Message.content.contains(filters.query))  # type: ignore
                if filters.model:
                    msg_query = msg_query.where(Message.model_name == filters.model)

                msg_query = msg_query.order_by(Message.timestamp.asc())  # type: ignore
                msg_result = await session.execute(msg_query)
                messages = msg_result.scalars().all()

                if not messages:
                    continue

                chat_messages = [to_chat_message(msg) for msg in messages]
                conv_data = format_conversation(conv, messages)
                results.append((conv_data, chat_messages))

            return results

    async def get_conversation_stats(
        self,
        filters: StatsFilters,
    ) -> dict[str, dict[str, Any]]:
        """Get statistics using SQL aggregations."""
        from llmling_agent_storage.sql_provider.models import Conversation, Message

        async with AsyncSession(self.engine) as session:
            # Base query for stats
            query = (
                select(  # type: ignore[call-overload]
                    Message.model,
                    Conversation.agent_name,
                    Message.timestamp,
                    Message.total_tokens,
                    Message.input_tokens,
                    Message.output_tokens,
                )
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(Message.timestamp > filters.cutoff)
            )

            if filters.agent_name:
                query = query.where(Conversation.agent_name == filters.agent_name)

            # Execute query and get raw data
            result = await session.execute(query)
            rows = [
                (
                    model,
                    agent,
                    timestamp,
                    TokenCost(
                        token_usage=RunUsage(
                            input_tokens=prompt or 0,
                            output_tokens=completion or 0,
                        ),
                        total_cost=Decimal(0),  # We don't store this in DB
                    )
                    if total or prompt or completion
                    else None,
                )
                for model, agent, timestamp, total, prompt, completion in result.all()
            ]

        # Use base class aggregation
        return self.aggregate_stats(rows, filters.group_by)

    async def reset(
        self,
        *,
        agent_name: str | None = None,
        hard: bool = False,
    ) -> tuple[int, int]:
        """Reset database storage."""
        from sqlalchemy import text

        from llmling_agent_storage.sql_provider.queries import (
            DELETE_AGENT_CONVERSATIONS,
            DELETE_AGENT_MESSAGES,
            DELETE_ALL_CONVERSATIONS,
            DELETE_ALL_MESSAGES,
        )

        async with AsyncSession(self.engine) as session:
            if hard:
                if agent_name:
                    msg = "Hard reset cannot be used with agent_name"
                    raise ValueError(msg)
                # Drop and recreate all tables
                async with self.engine.begin() as conn:
                    await conn.run_sync(SQLModel.metadata.drop_all)
                await session.commit()
                # Recreate schema
                await self._init_database()
                return 0, 0

            # Get counts first
            conv_count, msg_count = await self.get_conversation_counts(agent_name=agent_name)

            # Delete data
            if agent_name:
                await session.execute(text(DELETE_AGENT_MESSAGES), {"agent": agent_name})
                await session.execute(text(DELETE_AGENT_CONVERSATIONS), {"agent": agent_name})
            else:
                await session.execute(text(DELETE_ALL_MESSAGES))
                await session.execute(text(DELETE_ALL_CONVERSATIONS))

            await session.commit()
        return conv_count, msg_count

    async def get_conversation_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get conversation and message counts."""
        from llmling_agent_storage.sql_provider import Conversation, Message

        if not self.session:
            msg = "Session not initialized. Use provider as async context manager."
            raise RuntimeError(msg)
        if agent_name:
            conv_query = select(Conversation).where(Conversation.agent_name == agent_name)
            msg_query = (
                select(Message).join(Conversation).where(Conversation.agent_name == agent_name)
            )
        else:
            conv_query = select(Conversation)
            msg_query = select(Message)

        conv_result = await self.session.execute(conv_query)
        msg_result = await self.session.execute(msg_query)
        conv_count = len(conv_result.scalars().all())
        msg_count = len(msg_result.scalars().all())

        return conv_count, msg_count
