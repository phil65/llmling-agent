"""SQLModel-based storage provider."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlmodel import Session, SQLModel, desc, select

from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import TokenCost
from llmling_agent.utils.now import get_now
from llmling_agent.utils.parse_time import parse_time_period
from llmling_agent_storage.base import StorageProvider
from llmling_agent_storage.models import QueryFilters
from llmling_agent_storage.sql_provider.models import (
    CommandHistory,
    Conversation,
    Message,
)
from llmling_agent_storage.sql_provider.utils import (
    build_message_query,
    format_conversation,
    get_column_default,
    parse_model_info,
    to_chat_message,
)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy import Engine

    from llmling_agent.common_types import JsonValue
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.tools import ToolCallInfo
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.storage import SQLStorageConfig
    from llmling_agent_storage.models import ConversationData, StatsFilters


logger = get_logger(__name__)


class SQLModelProvider(StorageProvider[Message]):
    """Storage provider using SQLModel.

    Can work with any database supported by SQLAlchemy/SQLModel.
    Provides efficient SQL-based filtering and storage.
    """

    can_load_history = True

    def __init__(
        self,
        config: SQLStorageConfig,
        engine: Engine,
    ):
        """Initialize provider with database engine.

        Args:
            config: Configuration for provider
            engine: SQLModel engine instance
        """
        from llmling_agent_storage.sql_provider.models import SQLModel

        super().__init__(config)
        self.engine = engine
        SQLModel.metadata.create_all(self.engine)
        self._init_database(auto_migrate=config.auto_migration)

    def _init_database(self, auto_migrate: bool = True):
        """Initialize database tables and optionally migrate columns.

        Args:
            auto_migrate: Whether to automatically add missing columns
        """
        from sqlalchemy import inspect
        from sqlalchemy.sql import text

        from llmling_agent_storage.sql_provider.models import SQLModel

        # Create tables if they don't exist
        SQLModel.metadata.create_all(self.engine)

        # Optionally add missing columns
        if auto_migrate:
            with self.engine.connect() as conn:
                inspector = inspect(self.engine)

                # For each table in our models
                for table_name, table in SQLModel.metadata.tables.items():
                    existing = {col["name"] for col in inspector.get_columns(table_name)}

                    # For each column in model that doesn't exist in DB
                    for col in table.columns:
                        if col.name not in existing:
                            # Create ALTER TABLE statement based on column type
                            type_sql = col.type.compile(self.engine.dialect)
                            nullable = "" if col.nullable else " NOT NULL"
                            default = get_column_default(col)
                            sql = f"ALTER TABLE {table_name} ADD COLUMN {col.name} {type_sql}{nullable}{default}"  # noqa: E501
                            conn.execute(text(sql))

                conn.commit()

    def cleanup(self):
        """Clean up database resources."""
        self.engine.dispose()

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Filter messages using SQL queries."""
        with Session(self.engine) as session:
            stmt = build_message_query(query)
            messages = session.exec(stmt).all()
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
    ):
        """Log message to database."""
        from llmling_agent_storage.sql_provider.models import Message

        provider, model_name = parse_model_info(model)

        with Session(self.engine) as session:
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
                total_tokens=cost_info.token_usage.get("total") if cost_info else None,
                input_tokens=cost_info.token_usage.get("prompt") if cost_info else None,
                output_tokens=cost_info.token_usage.get("completion")
                if cost_info
                else None,
                cost=float(cost_info.total_cost) if cost_info else None,
                forwarded_from=forwarded_from,
                timestamp=get_now(),
            )
            session.add(msg)
            session.commit()

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ):
        """Log conversation to database."""
        from llmling_agent_storage.sql_provider.models import Conversation

        with Session(self.engine) as session:
            now = start_time or get_now()
            convo = Conversation(id=conversation_id, agent_name=node_name, start_time=now)
            session.add(convo)
            session.commit()

    async def log_tool_call(
        self,
        *,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ):
        """Log tool call to database."""
        from llmling_agent_storage.sql_provider.models import ToolCall

        with Session(self.engine) as session:
            call = ToolCall(
                conversation_id=conversation_id,
                message_id=message_id,
                tool_call_id=tool_call.tool_call_id,
                timestamp=tool_call.timestamp,
                tool_name=tool_call.tool_name,
                args=tool_call.args,
                result=str(tool_call.result),
            )
            session.add(call)
            session.commit()

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ):
        """Log command to database."""
        with Session(self.engine) as session:
            history = CommandHistory(
                session_id=session_id,
                agent_name=agent_name,
                command=command,
                context_type=context_type.__name__ if context_type else None,
                context_metadata=metadata or {},
            )
            session.add(history)
            session.commit()

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
            format_conversation(
                conv, msgs, compact=compact, include_tokens=include_tokens
            )
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
        with Session(self.engine) as session:
            query = select(CommandHistory)
            if current_session_only:
                query = query.where(CommandHistory.session_id == str(session_id))
            else:
                query = query.where(CommandHistory.agent_name == agent_name)

            query = query.order_by(desc(CommandHistory.timestamp))  # type: ignore
            if limit:
                query = query.limit(limit)

            return [h.command for h in session.exec(query)]

    async def get_conversations(
        self,
        filters: QueryFilters,
    ) -> list[tuple[ConversationData, Sequence[ChatMessage[str]]]]:
        """Get filtered conversations using SQL queries."""
        with Session(self.engine) as session:
            results: list[tuple[ConversationData, Sequence[ChatMessage[str]]]] = []

            # Base conversation query
            conv_query = select(Conversation)

            if filters.agent_name:
                conv_query = conv_query.where(
                    Conversation.agent_name == filters.agent_name
                )

            if filters.since:
                # Changed: Gets conversations that STARTED after the cutoff time
                conv_query = conv_query.where(Conversation.start_time >= filters.since)

            conv_query = conv_query.order_by(desc(Conversation.start_time))  # type: ignore
            if filters.limit:
                conv_query = conv_query.limit(filters.limit)

            conversations = session.exec(conv_query).all()

            for conv in conversations:
                msg_query = select(Message).where(Message.conversation_id == conv.id)

                if filters.query:
                    msg_query = msg_query.where(Message.content.contains(filters.query))  # type: ignore
                if filters.model:
                    msg_query = msg_query.where(Message.model_name == filters.model)

                msg_query = msg_query.order_by(Message.timestamp.asc())  # type: ignore
                messages = session.exec(msg_query).all()

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

        with Session(self.engine) as session:
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
            rows = [
                (
                    model,
                    agent,
                    timestamp,
                    TokenCost(
                        token_usage={
                            "total": total or 0,
                            "prompt": prompt or 0,
                            "completion": completion or 0,
                        },
                        total_cost=Decimal(0),  # We don't store this in DB
                    )
                    if total or prompt or completion
                    else None,
                )
                for model, agent, timestamp, total, prompt, completion in session.exec(
                    query
                )
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

        with Session(self.engine) as session:
            if hard:
                if agent_name:
                    msg = "Hard reset cannot be used with agent_name"
                    raise ValueError(msg)
                # Drop and recreate all tables
                SQLModel.metadata.drop_all(self.engine)
                session.commit()
                # Recreate schema
                self._init_database()
                return 0, 0

            # Get counts first
            conv_count, msg_count = await self.get_conversation_counts(
                agent_name=agent_name
            )

            # Delete data
            if agent_name:
                session.execute(text(DELETE_AGENT_MESSAGES), {"agent": agent_name})
                session.execute(text(DELETE_AGENT_CONVERSATIONS), {"agent": agent_name})
            else:
                session.execute(text(DELETE_ALL_MESSAGES))
                session.execute(text(DELETE_ALL_CONVERSATIONS))

            session.commit()
            return conv_count, msg_count

    async def get_conversation_counts(
        self,
        *,
        agent_name: str | None = None,
    ) -> tuple[int, int]:
        """Get conversation and message counts."""
        from llmling_agent_storage.sql_provider import Conversation, Message

        with Session(self.engine) as session:
            if agent_name:
                conv_query = select(Conversation).where(
                    Conversation.agent_name == agent_name
                )
                msg_query = (
                    select(Message)
                    .join(Conversation)
                    .where(Conversation.agent_name == agent_name)
                )
            else:
                conv_query = select(Conversation)
                msg_query = select(Message)

            conv_count = len(session.exec(conv_query).all())
            msg_count = len(session.exec(msg_query).all())

            return conv_count, msg_count
