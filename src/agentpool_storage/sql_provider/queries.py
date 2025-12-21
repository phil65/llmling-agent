"""Some pre-defined Queries."""

from __future__ import annotations


DELETE_AGENT_MESSAGES = """\
DELETE FROM message
WHERE conversation_id IN (
    SELECT id FROM conversation WHERE agent_name = :agent
)
"""
DELETE_AGENT_CONVERSATIONS = "DELETE FROM conversation WHERE agent_name = :agent"

DELETE_ALL_MESSAGES = "DELETE FROM message"
DELETE_ALL_CONVERSATIONS = "DELETE FROM conversation"
