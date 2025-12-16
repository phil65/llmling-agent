"""Event handler configuration models for LLMling agent."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field
from pydantic.types import SecretStr
from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from schemez import Schema


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import RunContext

    from llmling_agent.agents.events import RichAgentStreamEvent
    from llmling_agent.common_types import IndividualEventHandler


StdOutStyle = Literal["simple", "detailed"]
TTSModel = Literal["tts-1", "tts-1-hd"]
TTSVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
TTSMode = Literal["sync_sentence", "sync_run", "async_queue", "async_cancel"]
"""TTS synchronization modes.

- sync_sentence: Wait for each sentence's audio before continuing (slowest, most synchronized)
- sync_run: Stream fast, wait for all audio at run end (default)
- async_queue: Stream fast, audio plays in background, multiple runs queue up
- async_cancel: Stream fast, audio plays in background, new run cancels previous audio
"""


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
        from llmling_agent.agents.events import detailed_print_handler, simple_print_handler

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
    """Text-to-Speech event handler with configurable synchronization modes.

    Modes:
    - sync_sentence: Wait for each sentence's audio before continuing (most synchronized)
    - sync_run: Stream fast, wait for all audio at run end (default)
    - async_queue: Stream fast, audio in background, multiple runs queue up
    - async_cancel: Stream fast, audio in background, new run cancels previous
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
        mode: TTSMode = "sync_run",
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._voice = voice
        self._speed = speed
        self._chunk_size = chunk_size
        self._sample_rate = sample_rate
        self._min_text_length = min_text_length
        self._mode = mode

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

    async def _cancel_pending(self) -> None:
        """Cancel all pending synthesis and playback."""
        # Cancel synthesis worker
        if self._synthesis_task and not self._synthesis_task.done():
            self._synthesis_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._synthesis_task
            self._synthesis_task = None

        # Clear sentence queue
        while not self._sentence_queue.empty():
            try:
                self._sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Cancel playback
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._playback_task
            self._playback_task = None

        # Clear audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Reset text buffer
        self._text_buffer = ""

    async def __call__(self, ctx: RunContext[Any], event: RichAgentStreamEvent[Any]) -> None:
        """Handle stream events and trigger TTS synthesis."""
        from llmling_agent.agents.events import RunStartedEvent, StreamCompleteEvent

        match event:
            case RunStartedEvent():
                # For async_cancel mode, cancel any pending audio from previous run
                if self._mode == "async_cancel":
                    await self._cancel_pending()

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
                            if self._mode == "sync_sentence":
                                await self._synthesize_text(sentence)
                            else:
                                self._schedule_synthesis(sentence)

            case StreamCompleteEvent():
                # Process remaining text
                if self._text_buffer.strip():
                    if self._mode == "sync_sentence":
                        await self._synthesize_text(self._text_buffer.strip())
                    else:
                        self._schedule_synthesis(self._text_buffer.strip())
                    self._text_buffer = ""

                # For sync modes, wait for everything to finish
                if self._mode in ("sync_sentence", "sync_run"):
                    # Wait for synthesis worker to finish
                    if self._synthesis_task and not self._synthesis_task.done():
                        await self._sentence_queue.put(None)
                        await self._synthesis_task

                    # Signal playback to stop and wait for it
                    await self._audio_queue.put(None)
                    if self._playback_task and not self._playback_task.done():
                        await self._playback_task
                # For async modes, don't wait - let audio continue in background


class EdgeTTSEventHandler:
    """Text-to-Speech event handler using Edge TTS (free, no API key required).

    Uses Microsoft Edge's TTS service via edge-tts library.
    Outputs MP3 which is decoded to PCM via miniaudio for playback.

    Modes:
    - sync_sentence: Wait for each sentence's audio before continuing (most synchronized)
    - sync_run: Stream fast, wait for all audio at run end (default)
    - async_queue: Stream fast, audio in background, multiple runs queue up
    - async_cancel: Stream fast, audio in background, new run cancels previous
    """

    def __init__(
        self,
        *,
        voice: str = "en-US-AriaNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        sample_rate: int = 24000,
        min_text_length: int = 20,
        mode: TTSMode = "sync_run",
    ) -> None:
        self._voice = voice
        self._rate = rate
        self._volume = volume
        self._pitch = pitch
        self._sample_rate = sample_rate
        self._min_text_length = min_text_length
        self._mode = mode

        # State
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._playback_task: asyncio.Task[None] | None = None
        self._synthesis_task: asyncio.Task[None] | None = None
        self._text_buffer = ""
        self._sentence_terminators = frozenset({".", "!", "?", "\n"})

    async def _play_audio(self) -> None:
        """Async audio playback using sounddevice."""
        import sounddevice as sd

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
        """Synthesize text using edge-tts and queue decoded PCM audio."""
        import edge_tts
        import miniaudio  # type: ignore[import-untyped]

        if not text.strip():
            return

        # Ensure playback task is running
        if self._playback_task is None or self._playback_task.done():
            self._playback_task = asyncio.create_task(self._play_audio())

        try:
            communicate = edge_tts.Communicate(
                text,
                voice=self._voice,
                rate=self._rate,
                volume=self._volume,
                pitch=self._pitch,
            )

            # Collect MP3 chunks and decode to PCM
            mp3_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_data.extend(chunk["data"])

            if mp3_data:
                # Decode MP3 to mono 16-bit PCM at our target sample rate
                decoded = miniaudio.decode(
                    bytes(mp3_data),
                    output_format=miniaudio.SampleFormat.SIGNED16,
                    nchannels=1,
                    sample_rate=self._sample_rate,
                )
                await self._audio_queue.put(decoded.samples.tobytes())

        except Exception as e:  # noqa: BLE001
            print(f"\n❌ Edge TTS error: {e}", file=sys.stderr)

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

    async def _cancel_pending(self) -> None:
        """Cancel all pending synthesis and playback."""
        # Cancel synthesis worker
        if self._synthesis_task and not self._synthesis_task.done():
            self._synthesis_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._synthesis_task
            self._synthesis_task = None

        # Clear sentence queue
        while not self._sentence_queue.empty():
            try:
                self._sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Cancel playback
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._playback_task
            self._playback_task = None

        # Clear audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Reset text buffer
        self._text_buffer = ""

    async def __call__(self, ctx: RunContext[Any], event: RichAgentStreamEvent[Any]) -> None:
        """Handle stream events and trigger TTS synthesis."""
        from llmling_agent.agents.events import RunStartedEvent, StreamCompleteEvent

        match event:
            case RunStartedEvent():
                # For async_cancel mode, cancel any pending audio from previous run
                if self._mode == "async_cancel":
                    await self._cancel_pending()

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
                            if self._mode == "sync_sentence":
                                await self._synthesize_text(sentence)
                            else:
                                self._schedule_synthesis(sentence)

            case StreamCompleteEvent():
                # Process remaining text
                if self._text_buffer.strip():
                    if self._mode == "sync_sentence":
                        await self._synthesize_text(self._text_buffer.strip())
                    else:
                        self._schedule_synthesis(self._text_buffer.strip())
                    self._text_buffer = ""

                # For sync modes, wait for everything to finish
                if self._mode in ("sync_sentence", "sync_run"):
                    # Wait for synthesis worker to finish
                    if self._synthesis_task and not self._synthesis_task.done():
                        await self._sentence_queue.put(None)
                        await self._synthesis_task

                    # Signal playback to stop and wait for it
                    await self._audio_queue.put(None)
                    if self._playback_task and not self._playback_task.done():
                        await self._playback_task
                # For async modes, don't wait - let audio continue in background


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

    mode: TTSMode = Field(
        default="sync_run",
        examples=["sync_sentence", "sync_run", "async_queue", "async_cancel"],
        title="Synchronization Mode",
    )
    """How TTS synthesis synchronizes with the event stream.

    - sync_sentence: Wait for each sentence's audio before continuing (slowest, most synchronized)
    - sync_run: Stream fast, wait for all audio at run end (default, recommended)
    - async_queue: Stream fast, audio plays in background, multiple runs queue up
    - async_cancel: Stream fast, audio plays in background, new run cancels previous audio
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
            mode=self.mode,
            min_text_length=self.min_text_length,
        )


class EdgeTTSEventHandlerConfig(BaseEventHandlerConfig):
    """Configuration for Edge TTS event handler (free, no API key required).

    Uses Microsoft Edge's TTS service via edge-tts library.
    Supports many voices and languages without requiring an API key.
    """

    model_config = ConfigDict(title="Edge TTS Event Handler")

    type: Literal["edge-tts"] = Field("edge-tts", init=False)
    """Edge TTS event handler."""

    voice: str = Field(
        default="en-US-AriaNeural",
        examples=[
            "en-US-AriaNeural",
            "en-US-GuyNeural",
            "en-GB-SoniaNeural",
            "de-DE-KatjaNeural",
            "fr-FR-DeniseNeural",
        ],
        title="Voice name",
    )
    """Voice to use for synthesis.

    Use `edge-tts --list-voices` to see all available voices.
    Format: {locale}-{Name}Neural (e.g., en-US-AriaNeural)
    """

    rate: str = Field(
        default="+0%",
        examples=["-50%", "+0%", "+25%", "+50%"],
        title="Speech rate",
    )
    """Speaking rate adjustment (e.g., '+25%', '-10%')."""

    volume: str = Field(
        default="+0%",
        examples=["-50%", "+0%", "+25%", "+50%"],
        title="Volume",
    )
    """Volume adjustment (e.g., '+10%', '-20%')."""

    pitch: str = Field(
        default="+0Hz",
        examples=["-50Hz", "+0Hz", "+25Hz", "+50Hz"],
        title="Pitch",
    )
    """Pitch adjustment in Hz (e.g., '+10Hz', '-5Hz')."""

    sample_rate: int = Field(default=24000, examples=[16000, 24000, 44100], title="Sample Rate")
    """Audio sample rate in Hz for playback."""

    min_text_length: int = Field(
        default=20,
        ge=5,
        examples=[10, 20, 50],
        title="Minimum Text Length",
    )
    """Minimum text length before synthesizing (in characters)."""

    mode: TTSMode = Field(
        default="sync_run",
        examples=["sync_sentence", "sync_run", "async_queue", "async_cancel"],
        title="Synchronization Mode",
    )
    """How TTS synthesis synchronizes with the event stream.

    - sync_sentence: Wait for each sentence's audio before continuing (slowest, most synchronized)
    - sync_run: Stream fast, wait for all audio at run end (default, recommended)
    - async_queue: Stream fast, audio plays in background, multiple runs queue up
    - async_cancel: Stream fast, audio plays in background, new run cancels previous audio
    """

    def get_handler(self) -> IndividualEventHandler:
        """Get the Edge TTS event handler."""
        return EdgeTTSEventHandler(
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
            sample_rate=self.sample_rate,
            mode=self.mode,
            min_text_length=self.min_text_length,
        )


EventHandlerConfig = Annotated[
    StdoutEventHandlerConfig
    | CallbackEventHandlerConfig
    | TTSEventHandlerConfig
    | EdgeTTSEventHandlerConfig,
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
