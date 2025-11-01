"""Base class for message processing nodes."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from anyenv import method_spawner
from psygnal import Signal
from pydantic_ai import UserPromptPart

from llmling_agent.messaging.messageemitter import MessageEmitter
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.prompts.convert import convert_prompts
from llmling_agent.tools import ToolCallInfo
from llmling_agent_providers.pydanticai.convert_content import content_to_pydantic_ai


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    import os

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.models.content import Content
    from llmling_agent.talk.stats import AggregatedMessageStats, MessageStats


class MessageNode[TDeps, TResult](MessageEmitter[TDeps, TResult]):
    """Base class for all message processing nodes."""

    tool_used = Signal(ToolCallInfo)
    """Signal emitted when node uses a tool."""

    async def pre_run(
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str] | ChatMessage,
    ) -> tuple[ChatMessage[Any], list[Content | str]]:
        """Hook to prepare a MessgeNode run call.

        Args:
            *prompt: The prompt(s) to prepare.

        Returns:
            A tuple of:
                - Either incoming message, or a constructed incoming message based
                  on the prompt(s).
                - A list of prompts to be sent to the model.
        """
        if len(prompt) == 1 and isinstance(prompt[0], ChatMessage):
            user_msg = prompt[0]
            prompts = await convert_prompts([user_msg.content])
            # Update received message's chain to show it came through its source
            user_msg = user_msg.forwarded(prompt[0])
            # clear cost info to avoid double-counting
            user_msg = replace(user_msg, role="user", cost_info=None)
            final_prompt = "\n\n".join(str(p) for p in prompts)
        else:
            prompts = await convert_prompts(prompt)
            final_prompt = "\n\n".join(str(p) for p in prompts)
            # use format_prompts?
            user_msg = ChatMessage[str](
                content=final_prompt,
                role="user",
                conversation_id=str(uuid4()),
                kind="request",
                parts=[
                    UserPromptPart(content=[content_to_pydantic_ai(i) for i in prompts])
                ],
            )
        self.message_received.emit(user_msg)
        self.context.current_prompt = final_prompt
        return user_msg, prompts

    # async def post_run(
    #     self,
    #     message: ChatMessage[TResult],
    #     previous_message: ChatMessage[Any] | None,
    #     wait_for_connections: bool | None = None,
    # ) -> ChatMessage[Any]:
    #     # For chain processing, update the response's chain
    #     if previous_message:
    #         message = message.forwarded(previous_message)
    #         conversation_id = previous_message.conversation_id
    #     else:
    #         conversation_id = str(uuid4())
    #     # Set conversation_id on response message
    #     message = replace(message, conversation_id=conversation_id)
    #     self.message_sent.emit(message)
    #     await self.log_message(response_msg)
    #     await self.connections.route_message(message, wait=wait_for_connections)
    #     return message

    # @overload
    # async def run(
    #     self,
    #     *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str] | ChatMessage,
    #     wait_for_connections: bool | None = None,
    #     store_history: bool = True,
    #     output_type: None,
    #     **kwargs: Any,
    # ) -> ChatMessage[TResult]: ...

    # @overload
    # async def run[OutputTypeT](
    #     self,
    #     *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str] | ChatMessage,
    #     wait_for_connections: bool | None = None,
    #     store_history: bool = True,
    #     output_type: type[OutputTypeT],
    #     **kwargs: Any,
    # ) -> ChatMessage[OutputTypeT]: ...

    @method_spawner
    async def run[OutputTypeT](
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str] | ChatMessage,
        wait_for_connections: bool | None = None,
        store_history: bool = True,
        output_type: type[OutputTypeT] | None = None,
        **kwargs: Any,
    ) -> ChatMessage[Any]:
        """Execute node with prompts and handle message routing.

        Args:
            prompt: Input prompts
            wait_for_connections: Whether to wait for forwarded messages
            store_history: Whether to store in conversation history
            output_type: Type of output to expect
            **kwargs: Additional arguments for _run
        """
        from llmling_agent import Agent

        user_msg, prompts = await self.pre_run(*prompt)
        message = await self._run(
            *prompts,
            store_history=store_history,
            conversation_id=user_msg.conversation_id,
            output_type=output_type,
            **kwargs,
        )

        # For chain processing, update the response's chain
        if len(prompt) == 1 and isinstance(prompt[0], ChatMessage):
            message = message.forwarded(prompt[0])

        if store_history and isinstance(self, Agent):
            self.conversation.add_chat_messages([user_msg, message])
        self.message_sent.emit(message)
        await self.connections.route_message(message, wait=wait_for_connections)
        return message

    @abstractmethod
    async def get_stats(self) -> MessageStats | AggregatedMessageStats:
        """Get message statistics for this node."""

    @abstractmethod
    def run_iter(
        self,
        *prompts: Any,
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages during execution."""
