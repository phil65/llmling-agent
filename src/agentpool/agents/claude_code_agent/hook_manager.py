"""Hook manager for ClaudeCodeAgent.

Centralizes all hook-related logic:
- Built-in hooks (PreCompact, injection)
- AgentHooks integration
- Injection consumption from PromptInjectionManager
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from agentpool.log import get_logger


if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable

    from clawd_code_sdk.types import HookContext, HookInput, HookMatcher, SyncHookJSONOutput

    from agentpool.agents.prompt_injection import PromptInjectionManager
    from agentpool.hooks import AgentHooks

logger = get_logger(__name__)


class ClaudeCodeHookManager:
    """Manages SDK hooks for ClaudeCodeAgent.

    Responsibilities:
    - Builds SDK hooks configuration from multiple sources
    - Consumes injections from PromptInjectionManager
    - Provides clean API for hook-related operations

    Example:
        hook_manager = ClaudeCodeHookManager(
            agent_name="my-agent",
            agent_hooks=hooks,
            event_queue=queue,
            get_session_id=lambda: agent.session_id,
            injection_manager=agent._injection_manager,
        )

        # Injections are queued via agent.inject_prompt()
        # Hook manager consumes them in PostToolUse hooks

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
        injection_manager: PromptInjectionManager | None = None,
    ) -> None:
        """Initialize hook manager.

        Args:
            agent_name: Name of the agent (for logging/events)
            agent_hooks: Optional AgentHooks for pre/post tool hooks
            event_queue: Queue for emitting events (CompactionEvent, etc.)
            get_session_id: Callable to get current session ID
            injection_manager: Shared injection manager from BaseAgent
        """
        self.agent_name = agent_name
        self.agent_hooks = agent_hooks
        self._event_queue = event_queue
        self._get_session_id = get_session_id or (lambda: None)
        self._injection_manager = injection_manager

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

        return result

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
        compaction_event = CompactionEvent(session_id=session_id, trigger=trigger, phase="starting")
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

        Consumes pending injection from the shared PromptInjectionManager
        and adds it as additionalContext in the response.
        """
        result: SyncHookJSONOutput = {"continue_": True}
        # Consume pending injection from shared manager
        if self._injection_manager and (injection := await self._injection_manager.consume()):
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
