"""Event handler configuration models for AgentPool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field
from pydantic.types import SecretStr
from schemez import Schema


if TYPE_CHECKING:
    from collections.abc import Sequence

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


class TTSEventHandlerConfig(BaseEventHandlerConfig):
    """Configuration for Text-to-Speech event handler with OpenAI streaming."""

    model_config = ConfigDict(title="Text-to-Speech Event Handler")

    type: Literal["tts-openai"] = Field("tts-openai", init=False)
    """OpenAI TTS event handler."""

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
        from agentpool.agents.events.tts_handlers import OpenAITTSEventHandler

        key = self.api_key.get_secret_value() if self.api_key else None
        return OpenAITTSEventHandler(
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

    type: Literal["tts-edge"] = Field("tts-edge", init=False)
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

    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        examples=[0.5, 1.0, 1.5, 2.0],
        title="Speed of speech",
    )
    """Speed of speech (0.25 to 4.0, default 1.0)."""

    volume: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        examples=[0.5, 1.0, 1.5, 2.0],
        title="Volume",
    )
    """Volume level (0.0 to 2.0, default 1.0 = normal)."""

    pitch: float = Field(
        default=0.0,
        ge=-100.0,
        le=100.0,
        examples=[-50.0, 0.0, 25.0, 50.0],
        title="Pitch",
    )
    """Pitch adjustment in Hz (default 0.0 = no change)."""

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
        from agentpool.agents.events.tts_handlers import EdgeTTSEventHandler

        # Convert to Edge TTS string formats
        # speed: 1.0 -> "+0%", 1.5 -> "+50%", 0.5 -> "-50%"
        # volume: 1.0 -> "+0%", 1.5 -> "+50%", 0.5 -> "-50%"
        # pitch: 0.0 -> "+0Hz", 50.0 -> "+50Hz", -25.0 -> "-25Hz"
        rate = f"{round((self.speed - 1.0) * 100):+d}%"
        volume = f"{round((self.volume - 1.0) * 100):+d}%"
        pitch = f"{round(self.pitch):+d}Hz"

        return EdgeTTSEventHandler(
            voice=self.voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
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
