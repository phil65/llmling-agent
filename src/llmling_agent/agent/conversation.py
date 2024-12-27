"""Conversation management for LLMling agent."""

from __future__ import annotations

from contextlib import asynccontextmanager
import tempfile
from typing import TYPE_CHECKING, Any, Literal

from llmling.prompts import BasePrompt, PromptMessage, StaticPrompt
from pydantic_ai.messages import ModelRequest, UserPromptPart
from upath import UPath

from llmling_agent.log import get_logger
from llmling_agent.pydantic_ai_utils import convert_model_message


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    import os

    from pydantic_ai.messages import ModelMessage

    from llmling_agent.agent.agent import LLMlingAgent
    from llmling_agent.models.messages import ChatMessage

logger = get_logger(__name__)

OverrideMode = Literal["replace", "append"]
type PromptInput = str | BasePrompt


def _to_base_prompt(prompt: PromptInput) -> BasePrompt:
    """Convert input to BasePrompt instance."""
    if isinstance(prompt, str):
        return StaticPrompt(
            name="System prompt",
            description="System prompt",
            messages=[PromptMessage(role="system", content=prompt)],
        )
    return prompt


class ConversationManager:
    """Manages conversation state and system prompts."""

    def __init__(
        self,
        agent: LLMlingAgent[Any, Any],
        initial_prompts: str | Sequence[str] | None = None,
    ):
        """Initialize conversation manager.

        Args:
            agent: instance to manage
            initial_prompts: Initial system prompts that start each conversation
        """
        self._agent = agent
        self._initial_prompts: list[BasePrompt] = []
        self._current_history: list[ModelMessage] = []
        self._last_messages: list[ModelMessage] = []

        # Add initial prompts
        if initial_prompts is not None:
            prompts_list = (
                [initial_prompts] if isinstance(initial_prompts, str) else initial_prompts
            )
            for prompt in prompts_list:
                self._initial_prompts.append(
                    StaticPrompt(
                        name="Initial system prompt",
                        description="Initial system prompt",
                        messages=[PromptMessage(role="system", content=prompt)],
                    )
                )

    @asynccontextmanager
    async def temporary(
        self,
        *,
        sys_prompts: PromptInput | Sequence[PromptInput] | None = None,
        mode: OverrideMode = "append",
    ) -> AsyncIterator[None]:
        """Start temporary conversation with different system prompts."""
        # Store original state
        original_prompts = list(self._initial_prompts)
        original_system_prompts = (
            self._agent._pydantic_agent._system_prompts
        )  # Store pydantic-ai prompts
        original_history = self._current_history

        try:
            if sys_prompts is not None:
                # Convert to list of BasePrompt
                new_prompts: list[BasePrompt] = []
                if isinstance(sys_prompts, str | BasePrompt):
                    new_prompts = [_to_base_prompt(sys_prompts)]
                else:
                    new_prompts = [_to_base_prompt(prompt) for prompt in sys_prompts]

                self._initial_prompts = (
                    original_prompts + new_prompts if mode == "append" else new_prompts
                )

                # Update pydantic-ai's system prompts
                formatted_prompts = await self.get_all_prompts()
                self._agent._pydantic_agent._system_prompts = tuple(formatted_prompts)

            # Force new conversation
            self._current_history = []
            yield
        finally:
            # Restore complete original state
            self._initial_prompts = original_prompts
            self._agent._pydantic_agent._system_prompts = original_system_prompts
            self._current_history = original_history

    def add_prompt(self, prompt: PromptInput):
        """Add a system prompt.

        Args:
            prompt: String content or BasePrompt instance to add
        """
        self._initial_prompts.append(_to_base_prompt(prompt))

    async def get_all_prompts(self) -> list[str]:
        """Get all formatted system prompts in order."""
        result: list[str] = []

        for prompt in self._initial_prompts:
            try:
                messages = await prompt.format()
                result.extend(
                    msg.get_text_content() for msg in messages if msg.role == "system"
                )
            except Exception:
                logger.exception("Error formatting prompt")

        return result

    def get_history(self) -> list[ModelMessage]:
        """Get current conversation history."""
        return self._current_history

    def set_history(self, history: list[ModelMessage]):
        """Update conversation history after run."""
        self._current_history = history

    def clear(self):
        """Clear conversation history and prompts."""
        self._initial_prompts.clear()
        self._current_history = []
        self._last_messages = []

    @property
    def last_run_messages(self) -> list[ChatMessage]:
        """Get messages from the last run converted to our format."""
        return [convert_model_message(msg) for msg in self._last_messages]

    async def add_context_message(
        self,
        content: str,
        source: str | None = None,
        **metadata: Any,
    ):
        """Add a context message.

        Args:
            content: Text content to add
            source: Description of content source
            **metadata: Additional metadata to include with the message
        """
        meta_str = ""
        if metadata:
            meta_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
            meta_str = f"\nMetadata:\n{meta_str}\n"

        header = f"Content from {source}:" if source else "Additional context:"
        formatted = f"{header}{meta_str}\n{content}\n"
        message = ModelRequest(parts=[UserPromptPart(content=formatted)])
        self._current_history.append(message)

    async def add_context_from_path(
        self,
        path: str | os.PathLike[str],
        *,
        convert_to_md: bool = False,
        **metadata: Any,
    ):
        """Add file or URL content as context message.

        Args:
            path: Any UPath-supported path
            convert_to_md: Whether to convert content to markdown
            **metadata: Additional metadata to include with the message

        Raises:
            ValueError: If content cannot be loaded or converted
        """
        upath = UPath(path)

        if convert_to_md:
            try:
                from markitdown import MarkItDown

                md = MarkItDown()

                # Direct handling for local paths and http(s) URLs
                if upath.protocol in ("", "file", "http", "https"):
                    result = md.convert(upath.path)
                else:
                    with tempfile.NamedTemporaryFile(suffix=upath.suffix) as tmp:
                        tmp.write(upath.read_bytes())
                        tmp.flush()
                        result = md.convert(tmp.name)

                content = result.text_content
                source = f"markdown:{upath.name}"

            except Exception as e:
                msg = f"Failed to convert {path} to markdown: {e}"
                raise ValueError(msg) from e
        else:
            try:
                content = upath.read_text()
                source = f"{upath.protocol}:{upath.name}"
            except Exception as e:
                msg = f"Failed to read {path}: {e}"
                raise ValueError(msg) from e

        await self.add_context_message(content, source=source, **metadata)

    async def add_context_from_resource(
        self,
        resource_name: str,
        **params: Any,
    ):
        """Add content from runtime resource as context message.

        Args:
            resource_name: Name of the resource to load
            **params: Parameters to pass to resource loader

        Raises:
            RuntimeError: If no runtime is available
            ValueError: If resource loading fails
        """
        if not self._agent.runtime:
            msg = "No runtime available to load resources"
            raise RuntimeError(msg)

        content = await self._agent.runtime.load_resource(resource_name, **params)
        await self.add_context_message(str(content), source=f"resource:{resource_name}")

    async def add_context_from_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, Any] | None = None,
        **metadata: Any,
    ):
        """Add rendered prompt content as context message.

        Args:
            prompt_name: Name of the prompt to render
            arguments: Optional arguments for prompt formatting
            **metadata: Additional metadata to include with the message

        Raises:
            RuntimeError: If no runtime is available
            ValueError: If prompt loading or rendering fails
        """
        if not self._agent.runtime:
            msg = "No runtime available to load prompts"
            raise RuntimeError(msg)

        messages = await self._agent.runtime.render_prompt(prompt_name, arguments)
        content = "\n\n".join(msg.get_text_content() for msg in messages)

        await self.add_context_message(
            content,
            source=f"prompt:{prompt_name}",
            prompt_args=arguments,
            **metadata,
        )


if __name__ == "__main__":
    path = UPath("http://tmp/test.txt")
    print(str(path.path))