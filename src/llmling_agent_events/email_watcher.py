"""Email event source."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llmling_agent.log import get_logger
from llmling_agent.messaging.events import EmailEventData
from llmling_agent_events.base import EventSource


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from email.message import Message

    import aioimaplib

    from llmling_agent.messaging.events import EventData
    from llmling_agent_config.events import EmailConfig

logger = get_logger(__name__)


class EmailEventSource(EventSource):
    """Monitors email inbox for events."""

    def __init__(self, config: EmailConfig):
        self.config = config
        self._client: aioimaplib.IMAP4_SSL | aioimaplib.IMAP4 | None = None
        self._stop_event = asyncio.Event()

    async def connect(self):
        """Connect to email server with configured protocol."""
        import ssl

        import aioimaplib

        if self.config.ssl:
            ssl_context = ssl.create_default_context()
            self._client = aioimaplib.IMAP4_SSL(
                self.config.host, self.config.port, ssl_context=ssl_context
            )
        else:
            self._client = aioimaplib.IMAP4(self.config.host, self.config.port)
        await self._client.login(
            self.config.username, self.config.password.get_secret_value()
        )
        await self._client.select(self.config.folder)

    async def disconnect(self):
        """Close connection and cleanup."""
        self._stop_event.set()
        if self._client:
            try:
                await self._client.close()
                await self._client.logout()
            except Exception:
                logger.exception("Error during email client cleanup")
        self._client = None

    def _build_search_criteria(self) -> str:
        """Build IMAP search string from filters."""
        criteria = ["UNSEEN"] if not self.config.mark_seen else []

        # Add configured filters
        for key, value in self.config.filters.items():
            # Convert filter keys to IMAP criteria
            match key.upper():
                case "FROM":
                    criteria.append(f'FROM "{value}"')
                case "SUBJECT":
                    criteria.append(f'SUBJECT "{value}"')
                case "TO":
                    criteria.append(f'TO "{value}"')
                case _:
                    logger.warning("Unsupported email filter", key=key)

        return " ".join(criteria)

    def _process_email(self, email_bytes: bytes) -> EventData:
        from email.parser import BytesParser

        parser = BytesParser()
        email_msg: Message = parser.parsebytes(email_bytes)

        # Check size limit if configured
        if self.config.max_size and len(email_bytes) > self.config.max_size:
            msg = f"Email exceeds size limit of {self.config.max_size} bytes"
            raise ValueError(msg)

        # Extract content (prefer text/plain)
        content = ""
        for part in email_msg.walk():  # walk() is a method of Message
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    content = payload.decode()
                break
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    content = payload.decode()

        # Create event with email metadata
        meta = {"date": email_msg["date"], "message_id": email_msg["message-id"]}
        return EmailEventData(
            source=self.config.name,
            subject=email_msg["subject"],
            sender=email_msg["from"],
            body=content,
            metadata=meta,
        )

    async def events(self) -> AsyncGenerator[EventData]:
        """Monitor inbox and yield new email events."""
        if not self._client:
            msg = "Not connected to email server"
            raise RuntimeError(msg)

        while not self._stop_event.is_set():
            try:
                # Search for messages matching criteria
                search_criteria = self._build_search_criteria()
                _, messages = await self._client.search(search_criteria)

                # Process each message
                for num in messages[0].split():
                    try:
                        # Fetch full message
                        _, msg_data = await self._client.fetch(num, "(RFC822)")
                        if not msg_data:
                            continue

                        email_bytes = msg_data[0][1]
                        event = self._process_email(email_bytes)

                        # Mark as seen if configured
                        if self.config.mark_seen:
                            await self._client.store(num, "+FLAGS", "\\Seen")

                        yield event

                    except Exception:
                        logger.exception("Error processing email")

                # Wait before next check
                await asyncio.sleep(self.config.check_interval)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error checking emails")
                await asyncio.sleep(self.config.check_interval)
