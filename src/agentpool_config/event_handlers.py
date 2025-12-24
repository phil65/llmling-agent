"""Event handler configuration models for AgentPool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field
from pydantic.types import SecretStr
from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from schemez import Schema


if TYPE_CHECKING:
    from collections.abc import Sequence

    from anyvoice import TTSMode, TTSStream
    from pydantic_ai import RunContext

    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.common_types import IndividualEventHandler

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
        from agentpool.agents.events import detailed_print_handler, simple_print_handler

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
        from agentpool.utils.importing import import_callable

        return import_callable(self.import_path)


class TTSEventHandler:
    """Event handler adapter that bridges pydantic-ai events to anyvoice.

    This is a thin adapter that translates stream events to TTSStream.feed() calls.
    All TTS logic is delegated to the anyvoice library.
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
        self._api_key = api_key
        self._model: TTSModel = model
        self._voice: TTSVoice = voice
        self._speed = speed
        self._chunk_size = chunk_size
        self._sample_rate = sample_rate
        self._min_text_length = min_text_length
        self._mode: TTSMode = mode
        self._tts_stream: TTSStream | None = None

    async def _ensure_stream(self) -> TTSStream:
        """Get or create the TTS stream."""
        if self._tts_stream is None:
            from anyvoice import OpenAITTSProvider, SoundDeviceSink, TTSStream

            provider = OpenAITTSProvider(api_key=self._api_key)
            session = provider.session(
                model=self._model,
                voice=self._voice,
                speed=self._speed,
                chunk_size=self._chunk_size,
            )
            sink = SoundDeviceSink(sample_rate=self._sample_rate)
            self._tts_stream = TTSStream(
                session,
                sink=sink,
                mode=self._mode,
                min_text_length=self._min_text_length,
            )
            await self._tts_stream.__aenter__()
        return self._tts_stream

    async def _close_stream(self) -> None:
        """Close the TTS stream if open."""
        if self._tts_stream is not None:
            await self._tts_stream.__aexit__(None, None, None)
            self._tts_stream = None

    async def __call__(self, ctx: RunContext[Any], event: RichAgentStreamEvent[Any]) -> None:
        """Handle stream events and trigger TTS synthesis."""
        from agentpool.agents.events import RunStartedEvent, StreamCompleteEvent

        match event:
            case RunStartedEvent():
                # For async_cancel mode, cancel any pending audio from previous run
                if self._mode == "async_cancel" and self._tts_stream is not None:
                    await self._tts_stream.cancel()

            case (
                PartStartEvent(part=TextPart(content=delta))
                | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
            ):
                stream = await self._ensure_stream()
                await stream.feed(delta)

            case StreamCompleteEvent():
                await self._close_stream()


class EdgeTTSEventHandler:
    """Event handler adapter that bridges pydantic-ai events to anyvoice with Edge TTS.

    This is a thin adapter that translates stream events to TTSStream.feed() calls.
    Uses the free Microsoft Edge TTS service (no API key required).
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
        self._mode: TTSMode = mode
        self._tts_stream: TTSStream | None = None

    async def _ensure_stream(self) -> TTSStream:
        """Get or create the TTS stream."""
        if self._tts_stream is None:
            from anyvoice import EdgeTTSProvider, SoundDeviceSink, TTSStream

            provider = EdgeTTSProvider(sample_rate=self._sample_rate)
            session = provider.session(
                voice=self._voice,
                rate=self._rate,
                volume=self._volume,
                pitch=self._pitch,
            )
            sink = SoundDeviceSink(sample_rate=self._sample_rate)
            self._tts_stream = TTSStream(
                session,
                sink=sink,
                mode=self._mode,
                min_text_length=self._min_text_length,
            )
            await self._tts_stream.__aenter__()
        return self._tts_stream

    async def _close_stream(self) -> None:
        """Close the TTS stream if open."""
        if self._tts_stream is not None:
            await self._tts_stream.__aexit__(None, None, None)
            self._tts_stream = None

    async def __call__(self, ctx: RunContext[Any], event: RichAgentStreamEvent[Any]) -> None:
        """Handle stream events and trigger TTS synthesis."""
        from agentpool.agents.events import RunStartedEvent, StreamCompleteEvent

        match event:
            case RunStartedEvent():
                # For async_cancel mode, cancel any pending audio from previous run
                if self._mode == "async_cancel" and self._tts_stream is not None:
                    await self._tts_stream.cancel()

            case (
                PartStartEvent(part=TextPart(content=delta))
                | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
            ):
                stream = await self._ensure_stream()
                await stream.feed(delta)

            case StreamCompleteEvent():
                await self._close_stream()


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

    mode: Literal["sync_sentence", "sync_run", "async_queue", "async_cancel"] = Field(
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

    mode: Literal["sync_sentence", "sync_run", "async_queue", "async_cancel"] = Field(
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
