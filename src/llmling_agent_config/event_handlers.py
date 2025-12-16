"""Event handler configuration models for LLMling agent."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field
from pydantic.types import SecretStr
from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from schemez import Schema


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import RunContext

    from llmling_agent.agent.events import RichAgentStreamEvent
    from llmling_agent.common_types import IndividualEventHandler


StdOutStyle = Literal["simple", "detailed"]
TTSModel = Literal["tts-1", "tts-1-hd"]
TTSVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class BaseEventHandlerConfig(Schema):
    """Base configuration for event handlers."""

    type: str = Field(init=False)
    """Event handler type discriminator."""

    enabled: bool = Field(default=True)
    """Whether this handler is enabled."""

    def get_handler(self) -> IndividualEventHandler:
        """Create and return the configured event handler.

        Returns:
            Configured event handler callable.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class StdoutEventHandlerConfig(BaseEventHandlerConfig):
    """Configuration for built-in event handlers (simple, detailed)."""

    model_config = ConfigDict(title="Stdout Event Handler")

    type: Literal["builtin"] = Field("builtin", init=False)
    """Builtin event handler."""

    handler: StdOutStyle = Field(default="simple", examples=["simple", "detailed"])
    """Which builtin handler to use.

    - simple: Basic text and tool notifications
    - detailed: Comprehensive execution visibility
    """

    def get_handler(self) -> IndividualEventHandler:
        """Get the builtin event handler."""
        from llmling_agent.agent.events import detailed_print_handler, simple_print_handler

        handlers = {"simple": simple_print_handler, "detailed": detailed_print_handler}
        return handlers[self.handler]


class CallbackEventHandlerConfig(BaseEventHandlerConfig):
    """Configuration for custom callback event handlers via import path."""

    model_config = ConfigDict(title="Callback Event Handler")

    type: Literal["callback"] = Field("callback", init=False)
    """Callback event handler."""

    import_path: str = Field(
        examples=[
            "mymodule:my_handler",
            "mypackage.handlers:custom_event_handler",
        ],
    )
    """Import path to the handler function (module:function format)."""

    def get_handler(self) -> IndividualEventHandler:
        """Import and return the callback handler."""
        from llmling_agent.utils.importing import import_callable

        return import_callable(self.import_path)


class TTSEventHandler:
    """Text-to-Speech event handler with optional non-blocking synthesis.

    When blocking=False (default), sentences are queued and synthesized sequentially
    in the background, avoiding blocking the event stream while ensuring audio
    plays in correct order.

    When blocking=True, awaits each TTS synthesis call before processing the next
    event (original behavior, useful for testing).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModel = "tts-1",
        voice: TTSVoice = "alloy",
        speed: float = 1.0,
        chunk_size: int = 1024,
        sample_rate: int = 24000,
        min_text_length: int = 20,
        blocking: bool = False,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._voice = voice
        self._speed = speed
        self._chunk_size = chunk_size
        self._sample_rate = sample_rate
        self._min_text_length = min_text_length
        self._blocking = blocking

        # State
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._playback_task: asyncio.Task[None] | None = None
        self._synthesis_task: asyncio.Task[None] | None = None
        self._text_buffer = ""
        self._sentence_terminators = frozenset({".", "!", "?", "\n"})

    async def _play_audio(self) -> None:
        """Async audio playback using sounddevice."""
        import sounddevice as sd  # type: ignore[import-untyped]

        try:
            stream = sd.RawOutputStream(samplerate=self._sample_rate, channels=1, dtype="int16")
            stream.start()

            while True:
                chunk = await self._audio_queue.get()
                if chunk is None:
                    break
                if chunk:
                    stream.write(chunk)

            stream.stop()
            stream.close()
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ Audio playback error: {e}", file=sys.stderr)

    async def _synthesize_text(self, text: str) -> None:
        """Synthesize text and queue audio chunks."""
        if not text.strip():
            return

        # Ensure playback task is running
        if self._playback_task is None or self._playback_task.done():
            self._playback_task = asyncio.create_task(self._play_audio())

        try:
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=self._voice,
                input=text,
                response_format="pcm",
                speed=self._speed,
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=self._chunk_size):
                    await self._audio_queue.put(chunk)
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ TTS error: {e}", file=sys.stderr)

    async def _synthesis_worker(self) -> None:
        """Worker that processes sentences sequentially from the queue."""
        while True:
            sentence = await self._sentence_queue.get()
            if sentence is None:  # Shutdown signal
                break
            await self._synthesize_text(sentence)

    def _schedule_synthesis(self, text: str) -> None:
        """Queue text for sequential synthesis (non-blocking to caller)."""
        # Start worker if not running
        if self._synthesis_task is None or self._synthesis_task.done():
            self._synthesis_task = asyncio.create_task(self._synthesis_worker())
        # Queue the sentence - doesn't block
        self._sentence_queue.put_nowait(text)

    async def __call__(self, ctx: RunContext[Any], event: RichAgentStreamEvent[Any]) -> None:
        """Handle stream events and trigger TTS synthesis."""
        from llmling_agent.agent.events import StreamCompleteEvent

        match event:
            case (
                PartStartEvent(part=TextPart(content=delta))
                | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
            ):
                self._text_buffer += delta

                # Check for sentence boundaries
                if any(term in self._text_buffer for term in self._sentence_terminators):
                    last_term = max(
                        (self._text_buffer.rfind(term) for term in self._sentence_terminators),
                        default=-1,
                    )

                    if last_term > 0 and last_term >= self._min_text_length:
                        sentence = self._text_buffer[: last_term + 1].strip()
                        self._text_buffer = self._text_buffer[last_term + 1 :]

                        if sentence:
                            if self._blocking:
                                await self._synthesize_text(sentence)
                            else:
                                self._schedule_synthesis(sentence)

            case StreamCompleteEvent():
                # Process remaining text
                if self._text_buffer.strip():
                    if self._blocking:
                        await self._synthesize_text(self._text_buffer.strip())
                    else:
                        self._schedule_synthesis(self._text_buffer.strip())
                    self._text_buffer = ""

                # Wait for synthesis worker to finish (non-blocking mode)
                if self._synthesis_task and not self._synthesis_task.done():
                    # Signal worker to stop and wait for it
                    await self._sentence_queue.put(None)
                    await self._synthesis_task

                # Signal playback to stop and wait for it
                await self._audio_queue.put(None)
                if self._playback_task and not self._playback_task.done():
                    await self._playback_task


class TTSEventHandlerConfig(BaseEventHandlerConfig):
    """Configuration for Text-to-Speech event handler with OpenAI streaming."""

    model_config = ConfigDict(title="Text-to-Speech Event Handler")

    type: Literal["tts"] = Field("tts", init=False)
    """TTS event handler."""

    api_key: SecretStr | None = Field(default=None, examples=["sk-..."], title="OpenAI API Key")
    """OpenAI API key. If not provided, uses OPENAI_API_KEY env var."""

    model: TTSModel = Field(default="tts-1", examples=["tts-1", "tts-1-hd"], title="TTS Model")
    """TTS model to use.

    - tts-1: Fast, optimized for real-time streaming
    - tts-1-hd: Higher quality, slightly higher latency
    """

    voice: TTSVoice = Field(
        default="alloy",
        examples=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        title="Voice type",
    )
    """Voice to use for synthesis."""

    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        examples=[0.5, 1.0, 1.5, 2.0],
        title="Speed of speech",
    )
    """Speed of speech (0.25 to 4.0, default 1.0)."""

    chunk_size: int = Field(default=1024, ge=256, examples=[512, 1024, 2048], title="Chunk Size")
    """Size of audio chunks to process (in bytes)."""

    sample_rate: int = Field(default=24000, examples=[16000, 24000, 44100], title="Sample Rate")
    """Audio sample rate in Hz (for PCM format)."""

    min_text_length: int = Field(
        default=20,
        ge=5,
        examples=[10, 20, 50],
        title="Minimum Text Length",
    )
    """Minimum text length before synthesizing (in characters)."""

    blocking: bool = Field(default=False, title="Blocking Mode")
    """If True, wait for each TTS synthesis to complete before processing next event.

    Default (False) uses non-blocking fire-and-forget synthesis for better streaming
    performance. Set to True for testing or when you need sequential processing.
    """

    def get_handler(self) -> IndividualEventHandler:
        """Get the TTS event handler."""
        key = self.api_key.get_secret_value() if self.api_key else None
        return TTSEventHandler(
            api_key=key,
            model=self.model,
            voice=self.voice,
            speed=self.speed,
            chunk_size=self.chunk_size,
            sample_rate=self.sample_rate,
            blocking=self.blocking,
            min_text_length=self.min_text_length,
        )


EventHandlerConfig = Annotated[
    StdoutEventHandlerConfig | CallbackEventHandlerConfig | TTSEventHandlerConfig,
    Field(discriminator="type"),
]


def resolve_handler_configs(
    configs: Sequence[EventHandlerConfig] | None,
) -> list[IndividualEventHandler]:
    """Resolve event handler configs to actual handler callables.

    Args:
        configs: List of event handler configurations.

    Returns:
        List of resolved event handler callables.
    """
    if not configs:
        return []
    return [cfg.get_handler() for cfg in configs]
