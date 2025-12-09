"""Event handler configuration models for LLMling agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field
from schemez import Schema


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import RunContext

    from llmling_agent.agent.events import RichAgentStreamEvent
    from llmling_agent.common_types import IndividualEventHandler


class BaseEventHandlerConfig(Schema):
    """Base configuration for event handlers."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

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


class BuiltinEventHandlerConfig(BaseEventHandlerConfig):
    """Configuration for built-in event handlers (simple, detailed)."""

    type: Literal["builtin"] = Field("builtin", init=False)
    """Builtin event handler."""

    handler: Literal["simple", "detailed"] = Field(
        default="simple",
        examples=["simple", "detailed"],
    )
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


class TTSEventHandlerConfig(BaseEventHandlerConfig):
    """Configuration for Text-to-Speech event handler with OpenAI streaming."""

    type: Literal["tts"] = Field("tts", init=False)
    """TTS event handler."""

    api_key: str | None = Field(
        default=None,
        examples=["sk-..."],
    )
    """OpenAI API key. If not provided, uses OPENAI_API_KEY env var."""

    model: Literal["tts-1", "tts-1-hd"] = Field(
        default="tts-1",
        examples=["tts-1", "tts-1-hd"],
    )
    """TTS model to use.

    - tts-1: Fast, optimized for real-time streaming
    - tts-1-hd: Higher quality, slightly higher latency
    """

    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = Field(
        default="alloy",
        examples=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    )
    """Voice to use for synthesis."""

    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        examples=[0.5, 1.0, 1.5, 2.0],
    )
    """Speed of speech (0.25 to 4.0, default 1.0)."""

    chunk_size: int = Field(
        default=1024,
        ge=256,
        examples=[512, 1024, 2048],
    )
    """Size of audio chunks to process (in bytes)."""

    sample_rate: int = Field(
        default=24000,
        examples=[16000, 22050, 24000, 44100],
    )
    """Audio sample rate in Hz (for PCM format)."""

    min_text_length: int = Field(
        default=20,
        ge=5,
        examples=[10, 20, 50],
    )
    """Minimum text length before synthesizing (in characters)."""

    def get_handler(self) -> IndividualEventHandler:  # noqa: PLR0915
        """Get the TTS event handler."""
        import asyncio
        import sys

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            msg = "openai package required for TTS. Install with: uv pip install llmling-agent[tts]"
            raise ImportError(msg) from e

        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as e:
            msg = "sounddevice package required for TTS. Install extra 'tts'"
            raise ImportError(msg) from e

        from pydantic_ai import PartDeltaEvent, PartStartEvent
        from pydantic_ai.messages import TextPart, TextPartDelta

        from llmling_agent.agent.events import StreamCompleteEvent

        client = AsyncOpenAI(api_key=self.api_key)

        # Shared state for handler closure
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        playback_task: asyncio.Task[None] | None = None
        text_buffer = ""
        sentence_terminators = {".", "!", "?", "\n"}

        async def play_audio() -> None:
            """Async audio playback using sounddevice."""
            try:
                stream = sd.RawOutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                )
                stream.start()

                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break
                    if chunk:
                        stream.write(chunk)

                stream.stop()
                stream.close()
            except Exception as e:  # noqa: BLE001
                print(f"\n❌ Audio playback error: {e}", file=sys.stderr)

        async def synthesize_text(text: str) -> None:
            """Synthesize text and queue audio chunks."""
            nonlocal playback_task

            if not text.strip():
                return

            # Start playback task if not running
            if playback_task is None or playback_task.done():
                playback_task = asyncio.create_task(play_audio())

            try:
                async with client.audio.speech.with_streaming_response.create(
                    model=self.model,
                    voice=self.voice,
                    input=text,
                    response_format="pcm",
                    speed=self.speed,
                ) as response:
                    async for chunk in response.iter_bytes(chunk_size=self.chunk_size):
                        await audio_queue.put(chunk)
            except Exception as e:  # noqa: BLE001
                print(f"\n❌ TTS error: {e}", file=sys.stderr)

        async def handler(ctx: RunContext, event: RichAgentStreamEvent[Any]) -> None:
            nonlocal text_buffer, playback_task

            match event:
                case (
                    PartStartEvent(part=TextPart(content=delta))
                    | PartDeltaEvent(delta=TextPartDelta(content_delta=delta))
                ):
                    text_buffer += delta

                    # Check for sentence boundaries
                    if any(term in text_buffer for term in sentence_terminators):
                        last_term = max(
                            (text_buffer.rfind(term) for term in sentence_terminators),
                            default=-1,
                        )

                        if last_term > 0 and last_term >= self.min_text_length:
                            sentence = text_buffer[: last_term + 1].strip()
                            text_buffer = text_buffer[last_term + 1 :]

                            if sentence:
                                await synthesize_text(sentence)

                case StreamCompleteEvent():
                    # Process remaining text
                    if text_buffer.strip():
                        await synthesize_text(text_buffer.strip())
                        text_buffer = ""

                    # Signal playback to stop
                    await audio_queue.put(None)
                    if playback_task and not playback_task.done():
                        await playback_task

        return handler


EventHandlerConfig = Annotated[
    BuiltinEventHandlerConfig | CallbackEventHandlerConfig | TTSEventHandlerConfig,
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
