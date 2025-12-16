"""Semantic memory toolset using TypeAgent's KnowPro for knowledge processing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import os
import re
import time
from typing import TYPE_CHECKING, Any, Literal, Self

from llmling_agent.resource_providers import ResourceProvider


if TYPE_CHECKING:
    from types import TracebackType

    from typeagent.knowpro import answers, query, searchlang
    from typeagent.knowpro.answer_response_schema import AnswerResponse
    from typeagent.knowpro.conversation_base import ConversationBase
    from typeagent.knowpro.search_query_schema import SearchQuery
    from typeagent.podcasts import podcast
    from typeagent.storage.memory.semrefindex import TermToSemanticRefIndex
    import typechat

    from llmling_agent.agents import Agent
    from llmling_agent.agents.acp_agent import ACPAgent
    from llmling_agent.common_types import ModelType
    from llmling_agent.tools.base import Tool


class AgentTypeChatModel:
    """TypeChat language model backed by llmling-agent Agent.

    Implements the typechat.TypeChatLanguageModel protocol using an Agent
    for LLM completions instead of direct API calls.
    """

    def __init__(self, agent: Agent[Any, str] | ACPAgent) -> None:
        """Initialize with an Agent for completions.

        Args:
            agent: The llmling-agent Agent to use for LLM calls
        """
        self.agent = agent

    async def complete(
        self, prompt: str | list[typechat.PromptSection]
    ) -> typechat.Success[str] | typechat.Failure:
        """Request completion from the Agent.

        Args:
            prompt: Either a string prompt or list of PromptSection dicts

        Returns:
            Success with response text or Failure with error message
        """
        import typechat

        try:
            # Convert prompt sections to a single string if needed
            if isinstance(prompt, list):
                # Combine sections into a conversation-style prompt
                parts: list[str] = []
                for section in prompt:
                    role = section["role"]
                    content = section["content"]
                    match role:
                        case "system":
                            parts.append(f"[System]: {content}")
                        case "user":
                            parts.append(f"[User]: {content}")
                        case "assistant":
                            parts.append(f"[Assistant]: {content}")
                prompt_text = "\n\n".join(parts)
            else:
                prompt_text = prompt

            # Run the agent and get response
            result = await self.agent.run(prompt_text)
            return typechat.Success(result.data)

        except Exception as e:  # noqa: BLE001
            return typechat.Failure(f"Agent completion failed: {e!r}")


@dataclass
class ProcessingContext:
    """Context for TypeAgent knowledge processing."""

    lang_search_options: searchlang.LanguageSearchOptions
    answer_context_options: answers.AnswerContextOptions
    query_context: query.QueryEvalContext[podcast.PodcastMessage, TermToSemanticRefIndex]
    query_translator: typechat.TypeChatJsonTranslator[SearchQuery]
    answer_translator: typechat.TypeChatJsonTranslator[AnswerResponse]


@dataclass
class QueryResponse:
    """Response from a knowledge query."""

    success: bool
    answer: str
    time_ms: int


@dataclass
class IngestResponse:
    """Response from ingesting content into the knowledge base."""

    success: bool
    messages_added: int
    semantic_refs_added: int
    error: str | None = None


class SemanticMemoryTools(ResourceProvider):
    """Provider for semantic memory / knowledge processing tools.

    Uses TypeAgent's KnowPro for:
    - Semantic indexing of conversations/transcripts
    - Natural language search queries
    - Structured answer generation
    """

    def __init__(
        self,
        model: ModelType = None,
        dbname: str | None = None,
        name: str = "semantic_memory",
    ) -> None:
        """Initialize semantic memory tools.

        Args:
            model: Model to use for LLM sampling (query translation, answers)
            dbname: SQLite database path, or None for in-memory storage
            name: Provider name
        """
        super().__init__(name=name)
        self.model = model
        self.dbname = dbname
        self._agent: Agent[Any, str] | None = None
        self._context: ProcessingContext | None = None
        self._tools: list[Tool] | None = None

    async def __aenter__(self) -> Self:
        """Initialize the agent and TypeAgent context."""
        from llmling_agent import Agent

        # Create minimal agent for LLM sampling
        self._agent = Agent(model=self.model, name=f"{self.name}-sampler")
        await self._agent.__aenter__()
        self._context = await self._make_context()  # Build TypeAgent processing context
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Cleanup agent resources."""
        if self._agent:
            await self._agent.__aexit__(exc_type, exc_val, exc_tb)
            self._agent = None
        self._context = None

    async def _make_context(self) -> ProcessingContext:
        """Create TypeAgent processing context with our Agent-backed model."""
        from typeagent.aitools import utils
        from typeagent.knowpro import answers, searchlang
        from typeagent.knowpro.answer_response_schema import AnswerResponse
        from typeagent.knowpro.convsettings import ConversationSettings
        from typeagent.knowpro.search_query_schema import SearchQuery
        from typeagent.podcasts import podcast
        from typeagent.storage.utils import create_storage_provider

        if self._agent is None:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        settings = ConversationSettings()
        # Set up storage provider (SQLite or memory)
        settings.storage_provider = await create_storage_provider(
            settings.message_text_index_settings,
            settings.related_term_index_settings,
            self.dbname,
            podcast.PodcastMessage,
        )
        lang_search_options = searchlang.LanguageSearchOptions(
            compile_options=searchlang.LanguageQueryCompileOptions(),
            max_message_matches=25,
        )
        answer_context_options = answers.AnswerContextOptions(entities_top_k=50, topics_top_k=50)

        query_context = await self._load_conversation_index(settings)  # Load / create conv index
        # Create Agent-backed TypeChat model
        model = AgentTypeChatModel(self._agent)
        # Create translators for structured extraction
        query_translator = utils.create_translator(model, SearchQuery)
        answer_translator = utils.create_translator(model, AnswerResponse)
        return ProcessingContext(
            lang_search_options=lang_search_options,
            answer_context_options=answer_context_options,
            query_context=query_context,
            query_translator=query_translator,
            answer_translator=answer_translator,
        )

    async def _load_conversation_index(
        self,
        settings: Any,
    ) -> query.QueryEvalContext[podcast.PodcastMessage, Any]:
        """Load conversation index from database or create empty one."""
        from typeagent.knowpro import query
        from typeagent.podcasts import podcast

        if self.dbname is None:
            # Try loading from default test data, or create empty
            try:
                conversation = await podcast.Podcast.read_from_file(
                    "testdata/Episode_53_AdrianTchaikovsky_index",
                    settings,
                )
            except FileNotFoundError:
                conversation = await podcast.Podcast.create(settings)
        else:
            conversation = await podcast.Podcast.create(settings)

        self._conversation = conversation
        return query.QueryEvalContext(conversation)

    @property
    def conversation(self) -> ConversationBase[Any] | None:
        """Get the current conversation/knowledge base."""
        return getattr(self, "_conversation", None)

    async def get_tools(self) -> list[Tool]:
        """Get available semantic memory tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            self.create_tool(self.query_knowledge, read_only=True, idempotent=True),
            self.create_tool(self.ingest_transcript),
            self.create_tool(self.ingest_text),
        ]
        return self._tools

    async def query_knowledge(self, question: str) -> QueryResponse:
        """Query the knowledge base with a natural language question.

        Returns an answer synthesized from indexed conversations and documents.

        Args:
            question: Natural language question to answer

        Returns:
            QueryResponse with success status, answer text, and timing
        """
        from typeagent.knowpro import answers, searchlang
        import typechat

        if self._context is None:
            return QueryResponse(success=False, answer="Semantic memory not initialized", time_ms=0)
        t0 = time.time()
        question = question.strip()
        if not question:
            dt = int((time.time() - t0) * 1000)
            return QueryResponse(success=False, answer="No question provided", time_ms=dt)

        # Stage 1-3: LLM -> proto-query, compile, execute
        result = await searchlang.search_conversation_with_language(
            self._context.query_context.conversation,
            self._context.query_translator,
            question,
            self._context.lang_search_options,
        )

        if isinstance(result, typechat.Failure):
            dt = int((time.time() - t0) * 1000)
            return QueryResponse(success=False, answer=result.message, time_ms=dt)

        # Stage 3a-4: ordinals -> messages/semrefs, LLM -> answer
        _, combined_answer = await answers.generate_answers(
            self._context.answer_translator,
            result.value,
            self._context.query_context.conversation,
            question,
            options=self._context.answer_context_options,
        )

        dt = int((time.time() - t0) * 1000)

        match combined_answer.type:
            case "NoAnswer":
                answer = combined_answer.whyNoAnswer or "No answer found"
                return QueryResponse(success=False, answer=answer, time_ms=dt)
            case "Answered":
                answer = combined_answer.answer or ""
                return QueryResponse(success=True, answer=answer, time_ms=dt)
            case _:
                return QueryResponse(success=False, answer="Unexpected response type", time_ms=dt)

    async def ingest_transcript(
        self,
        file_path: str,
        name: str | None = None,
        fmt: Literal["auto", "txt", "vtt"] = "auto",
    ) -> IngestResponse:
        """Ingest a transcript file into the knowledge base.

        Supports plain text (.txt) and WebVTT (.vtt) formats.
        The content will be indexed for semantic search.

        Args:
            file_path: Path to the transcript file
            name: Optional name for the transcript (defaults to filename)
            fmt: File format - "auto" detects from extension, or specify "txt"/"vtt"

        Returns:
            IngestResponse with counts of added messages and semantic refs
        """
        if self._context is None or self.conversation is None:
            return IngestResponse(
                success=False,
                messages_added=0,
                semantic_refs_added=0,
                error="Semantic memory not initialized",
            )

        # Detect format
        if fmt == "auto":
            ext = os.path.splitext(file_path)[1].lower()  # noqa: PTH122
            fmt = "vtt" if ext == ".vtt" else "txt"

        try:
            if fmt == "vtt":
                result = await self._ingest_vtt_file(file_path, name)
            else:
                result = await self._ingest_text_file(file_path, name)
        except Exception as e:  # noqa: BLE001
            return IngestResponse(
                success=False,
                messages_added=0,
                semantic_refs_added=0,
                error=str(e),
            )
        else:
            return result

    async def _ingest_vtt_file(self, file_path: str, name: str | None) -> IngestResponse:
        """Ingest a WebVTT file."""
        from datetime import timedelta

        from typeagent.knowpro.universal_message import (
            UNIX_EPOCH,
            ConversationMessage,
            ConversationMessageMeta,
            format_timestamp_utc,
        )
        import webvtt  # type: ignore[import-untyped]

        assert self.conversation
        vtt = webvtt.read(file_path)
        messages: list[ConversationMessage] = []

        for caption in vtt:
            if not caption.text.strip():
                continue

            # Parse voice tags for speaker detection
            from typeagent.transcripts.transcript_ingest import (
                parse_voice_tags,
                webvtt_timestamp_to_seconds,
            )

            raw_text = getattr(caption, "raw_text", caption.text)
            voice_segments = parse_voice_tags(raw_text)

            for speaker, text in voice_segments:
                if not text.strip():
                    continue

                offset_seconds = webvtt_timestamp_to_seconds(caption.start)
                ts = format_timestamp_utc(UNIX_EPOCH + timedelta(seconds=offset_seconds))
                metadata = ConversationMessageMeta(speaker=speaker, recipients=[])
                message = ConversationMessage(text_chunks=[text], metadata=metadata, timestamp=ts)
                messages.append(message)
        if not messages:
            return IngestResponse(
                success=False,
                messages_added=0,
                semantic_refs_added=0,
                error="No messages found in VTT file",
            )

        result = await self.conversation.add_messages_with_indexing(messages)
        return IngestResponse(
            success=True,
            messages_added=result.messages_added,
            semantic_refs_added=result.semrefs_added,
        )

    async def _ingest_text_file(self, file_path: str, name: str | None) -> IngestResponse:
        """Ingest a plain text transcript file."""
        from typeagent.knowpro.universal_message import (
            UNIX_EPOCH,
            ConversationMessage,
            ConversationMessageMeta,
            format_timestamp_utc,
        )

        with open(file_path, encoding="utf-8") as f:  # noqa: PTH123
            lines = f.readlines()

        # Parse transcript lines with speaker detection
        speaker_pattern = re.compile(r"^\s*(?P<speaker>[A-Z][A-Z\s]*?):\s*(?P<text>.*)$")

        messages: list[ConversationMessage] = []
        current_speaker: str | None = None
        current_chunks: list[str] = []
        assert self.conversation
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            match = speaker_pattern.match(stripped)
            if match:
                # Save previous message if exists
                if current_chunks:
                    meta = ConversationMessageMeta(speaker=current_speaker, recipients=[])
                    ts = format_timestamp_utc(UNIX_EPOCH)
                    chunks = [" ".join(current_chunks)]
                    message = ConversationMessage(text_chunks=chunks, metadata=meta, timestamp=ts)
                    messages.append(message)
                current_speaker = match.group("speaker").strip()
                current_chunks = [match.group("text").strip()]
            elif current_chunks:
                current_chunks.append(stripped)
            else:
                # No speaker detected, use None
                current_chunks = [stripped]

        # Don't forget last message
        if current_chunks:
            metadata = ConversationMessageMeta(speaker=current_speaker, recipients=[])
            chunks = [" ".join(current_chunks)]
            ts = format_timestamp_utc(UNIX_EPOCH)
            message = ConversationMessage(text_chunks=chunks, metadata=metadata, timestamp=ts)
            messages.append(message)

        if not messages:
            err = "No messages found in text file"
            return IngestResponse(success=False, messages_added=0, semantic_refs_added=0, error=err)
        result = await self.conversation.add_messages_with_indexing(messages)
        return IngestResponse(
            success=True,
            messages_added=result.messages_added,
            semantic_refs_added=result.semrefs_added,
        )

    async def ingest_text(
        self,
        content: str,
        speaker: str | None = None,
        timestamp: str | None = None,
    ) -> IngestResponse:
        """Ingest raw text content into the knowledge base.

        Useful for adding content from memory, APIs, or other sources.
        Optionally specify a speaker name for attribution.

        Args:
            content: The text content to ingest
            speaker: Optional speaker/source attribution
            timestamp: Optional ISO timestamp (defaults to now)

        Returns:
            IngestResponse with counts of added messages and semantic refs
        """
        from typeagent.knowpro.universal_message import (
            ConversationMessage,
            ConversationMessageMeta,
            format_timestamp_utc,
        )

        if self._context is None or self.conversation is None:
            return IngestResponse(
                success=False,
                messages_added=0,
                semantic_refs_added=0,
                error="Semantic memory not initialized",
            )

        content = content.strip()
        if not content:
            return IngestResponse(
                success=False,
                messages_added=0,
                semantic_refs_added=0,
                error="No content provided",
            )

        # Use provided timestamp or current time
        if timestamp is None:
            timestamp = format_timestamp_utc(datetime.now(UTC))

        meta = ConversationMessageMeta(speaker=speaker, recipients=[])
        message = ConversationMessage(text_chunks=[content], metadata=meta, timestamp=timestamp)
        try:
            result = await self.conversation.add_messages_with_indexing([message])
            return IngestResponse(
                success=True,
                messages_added=result.messages_added,
                semantic_refs_added=result.semrefs_added,
            )
        except Exception as e:  # noqa: BLE001
            return IngestResponse(
                success=False,
                messages_added=0,
                semantic_refs_added=0,
                error=str(e),
            )


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        async with SemanticMemoryTools(model="openai:gpt-4o-mini") as tools:
            fns = await tools.get_tools()
            print(f"Available tools: {[t.name for t in fns]}")

    asyncio.run(main())
