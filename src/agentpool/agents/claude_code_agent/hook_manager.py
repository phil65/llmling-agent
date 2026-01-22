"""Hook manager for ClaudeCodeAgent.

Centralizes all hook-related logic:
- Built-in hooks (PreCompact, injection)
- AgentHooks integration
- Pending injection management
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from agentpool.log import get_logger


if TYPE_CHECKING:
    import asyncio

    from clawd_code_sdk.types import (
        HookContext,
        HookInput,
        HookMatcher,
        SyncHookJSONOutput,
    )

    from agentpool.hooks import AgentHooks

logger = get_logger(__name__)


class ClaudeCodeHookManager:
    """Manages SDK hooks for ClaudeCodeAgent.

    Responsibilities:
    - Builds SDK hooks configuration from multiple sources
    - Manages pending injection state
    - Provides clean API for hook-related operations

    Example:
        hook_manager = ClaudeCodeHookManager(
            agent_name="my-agent",
            agent_hooks=hooks,
            event_queue=queue,
            get_session_id=lambda: agent.session_id,
        )

        # Queue injection for next tool completion
        hook_manager.inject("Please also check the tests")

        # Get SDK hooks for ClaudeCode options
        sdk_hooks = hook_manager.build_hooks()
    """

    def __init__(
        self,
        *,
        agent_name: str,
        agent_hooks: AgentHooks | None = None,
        event_queue: asyncio.Queue[Any] | None = None,
        get_session_id: Callable[[], str | None] | None = None,
    ) -> None:
        """Initialize hook manager.

        Args:
            agent_name: Name of the agent (for logging/events)
            agent_hooks: Optional AgentHooks for pre/post tool hooks
            event_queue: Queue for emitting events (CompactionEvent, etc.)
            get_session_id: Callable to get current session ID
        """
        self.agent_name = agent_name
        self.agent_hooks = agent_hooks
        self._event_queue = event_queue
        self._get_session_id = get_session_id or (lambda: None)
        self._pending_injection: str | None = None

    def inject(self, message: str) -> None:
        """Queue a message to be injected on next tool completion.

        The message will be added as `additionalContext` in the PostToolUse
        hook response, making it visible to the model after the tool executes.

        Args:
            message: Message to inject into the conversation

        Note:
            Only one injection can be pending at a time. Calling inject()
            again before the previous injection is consumed will overwrite it.
        """
        self._pending_injection = message
        logger.debug("Queued injection", agent=self.agent_name, message_len=len(message))

    def has_pending_injection(self) -> bool:
        """Check if there's a pending injection."""
        return self._pending_injection is not None

    def clear_injection(self) -> None:
        """Clear pending injection without consuming it."""
        self._pending_injection = None

    def build_hooks(self) -> dict[str, list[HookMatcher]]:
        """Build complete SDK hooks configuration.

        Combines:
        - Built-in hooks (PreCompact, injection via PostToolUse)
        - AgentHooks (pre/post tool use)

        Returns:
            Dictionary mapping hook event names to HookMatcher lists
        """
        from clawd_code_sdk.types import HookMatcher

        from agentpool.agents.claude_code_agent.converters import build_sdk_hooks_from_agent_hooks

        result: dict[str, list[Any]] = {}

        # Add PreCompact hook for compaction events
        result["PreCompact"] = [HookMatcher(matcher=None, hooks=[self._on_pre_compact])]

        # Add PostToolUse hook for injection
        result["PostToolUse"] = [HookMatcher(matcher="*", hooks=[self._on_post_tool_use])]

        # Merge AgentHooks if present
        if self.agent_hooks:
            agent_hooks = build_sdk_hooks_from_agent_hooks(self.agent_hooks, self.agent_name)
            for event_name, matchers in agent_hooks.items():
                if event_name in result:
                    result[event_name].extend(matchers)
                else:
                    result[event_name] = matchers

        return result  # type: ignore[return-value]

    async def _on_pre_compact(
        self,
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        """Handle PreCompact hook by emitting a CompactionEvent."""
        from agentpool.agents.events import CompactionEvent

        trigger_value = input_data.get("trigger", "auto")
        trigger: Literal["auto", "manual"] = "manual" if trigger_value == "manual" else "auto"

        session_id = self._get_session_id() or "unknown"
        compaction_event = CompactionEvent(
            session_id=session_id,
            trigger=trigger,
            phase="starting",
        )

        if self._event_queue:
            await self._event_queue.put(compaction_event)

        return {"continue_": True}

    async def _on_post_tool_use(
        self,
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        """Handle PostToolUse hook for injection and observation.

        If there's a pending injection, it's consumed and added as
        additionalContext in the response.
        """
        result: SyncHookJSONOutput = {"continue_": True}

        # Check for pending injection
        if self._pending_injection:
            injection = self._pending_injection
            self._pending_injection = None

            tool_name = input_data.get("tool_name", "unknown")
            logger.debug(
                "Injecting context after tool use",
                agent=self.agent_name,
                tool=tool_name,
                injection_len=len(injection),
            )

            result["hookSpecificOutput"] = {
                "hookEventName": "PostToolUse",
                "additionalContext": injection,
            }

        return result
