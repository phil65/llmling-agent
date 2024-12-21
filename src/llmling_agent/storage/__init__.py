"""Database configuration and initialization for LLMling agent."""

from llmling_agent.storage.engine import engine, init_database
from llmling_agent.storage.models import (
    Conversation,
    Message,
    MessageLog,
    ConversationLog,
)

# Initialize database on import
init_database()

__all__ = [
    "DELETE_AGENT_CONVERSATIONS",
    "DELETE_AGENT_MESSAGES",
    "DELETE_ALL_CONVERSATIONS",
    "DELETE_ALL_MESSAGES",
    "Conversation",
    "ConversationLog",
    "Message",
    "MessageLog",
    "engine",
    "init_database",
]
