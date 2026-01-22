"""Hook manager for NativeAgent.

Centralizes all hook-related logic:
- AgentHooks integration (pre/post run, pre/post tool)
- Injection consumption from PromptInjectionManager
- Combined hook result handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentpool.hooks.base import HookResult
from agentpool.log import get_logger


if TYPE_CHECKING:
    from agentpool.agents.prompt_injection import PromptInjectionManager
    from agentpool.hooks import AgentHooks

logger = get_logger(__name__)


class NativeAgentHookManager:
    """Manages hooks and injection for NativeAgent.

    Responsibilities:
    - Wraps AgentHooks and delegates to it
    - Consumes injections from PromptInjectionManager
    - Combines injection with post-tool hook results

    Example:
        hook_manager = NativeAgentHookManager(
            agent_name="my-agent",
            agent_hooks=hooks,
            injection_manager=agent._injection_manager,
        )

        # Injections are queued via agent.inject_prompt()
        # Hook manager consumes them in post-tool hooks
        result = await hook_manager.run_post_tool_hooks(...)
        # result["additional_context"] contains the injection
    """

    def __init__(
        self,
        *,
        agent_name: str,
        agent_hooks: AgentHooks | None = None,
        injection_manager: PromptInjectionManager | None = None,
    ) -> None:
        """Initialize hook manager.

        Args:
            agent_name: Name of the agent (for logging)
            agent_hooks: Optional AgentHooks for pre/post hooks
            injection_manager: Shared injection manager from BaseAgent
        """
        self.agent_name = agent_name
        self.agent_hooks = agent_hooks
        self._injection_manager = injection_manager

    def has_hooks(self) -> bool:
        """Check if any hooks are configured."""
        return bool(self.agent_hooks and self.agent_hooks.has_hooks())

    async def run_pre_run_hooks(
        self,
        *,
        agent_name: str,
        prompt: str,
        session_id: str | None = None,
    ) -> HookResult:
        """Execute pre-run hooks.

        Args:
            agent_name: Name of the agent.
            prompt: The prompt being processed.
            session_id: Optional conversation identifier.

        Returns:
            Hook result. If decision is "deny", the run should be blocked.
        """
        if self.agent_hooks:
            return await self.agent_hooks.run_pre_run_hooks(
                agent_name=agent_name,
                prompt=prompt,
                session_id=session_id,
            )
        return HookResult(decision="allow")

    async def run_post_run_hooks(
        self,
        *,
        agent_name: str,
        prompt: str,
        result: Any,
        session_id: str | None = None,
    ) -> HookResult:
        """Execute post-run hooks.

        Args:
            agent_name: Name of the agent.
            prompt: The prompt that was processed.
            result: The result from the run.
            session_id: Optional conversation identifier.

        Returns:
            Hook result.
        """
        if self.agent_hooks:
            return await self.agent_hooks.run_post_run_hooks(
                agent_name=agent_name,
                prompt=prompt,
                result=result,
                session_id=session_id,
            )
        return HookResult(decision="allow")

    async def run_pre_tool_hooks(
        self,
        *,
        agent_name: str,
        tool_name: str,
        tool_input: dict[str, Any],
        session_id: str | None = None,
    ) -> HookResult:
        """Execute pre-tool-use hooks.

        Args:
            agent_name: Name of the agent.
            tool_name: Name of the tool being called.
            tool_input: Input arguments for the tool.
            session_id: Optional conversation identifier.

        Returns:
            Hook result. If decision is "deny", the tool call should be blocked.
            May include modified_input to change tool arguments.
        """
        if self.agent_hooks:
            return await self.agent_hooks.run_pre_tool_hooks(
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=tool_input,
                session_id=session_id,
            )
        return HookResult(decision="allow")

    async def run_post_tool_hooks(
        self,
        *,
        agent_name: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        duration_ms: float,
        session_id: str | None = None,
    ) -> HookResult:
        """Execute post-tool-use hooks and consume pending injection.

        This method combines:
        - Results from AgentHooks.run_post_tool_hooks()
        - Pending injection from PromptInjectionManager (if any)

        The injection is consumed after being included in the result.

        Args:
            agent_name: Name of the agent.
            tool_name: Name of the tool that was called.
            tool_input: Input arguments that were passed to the tool.
            tool_output: Output from the tool.
            duration_ms: How long the tool took to execute.
            session_id: Optional conversation identifier.

        Returns:
            Combined hook result. May include additional_context from hooks
            and/or pending injection.
        """
        # Get result from AgentHooks
        if self.agent_hooks:
            result = await self.agent_hooks.run_post_tool_hooks(
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                duration_ms=duration_ms,
                session_id=session_id,
            )
        else:
            result = HookResult(decision="allow")

        # Consume pending injection from shared manager
        if self._injection_manager:
            injection = await self._injection_manager.consume()
            if injection:
                logger.debug(
                    "Consuming injection after tool use",
                    agent=self.agent_name,
                    tool=tool_name,
                    injection_len=len(injection),
                )

                # Combine with existing additional_context
                existing_context = result.get("additional_context")
                if existing_context:
                    result["additional_context"] = f"{existing_context}\n\n{injection}"
                else:
                    result["additional_context"] = injection

        return result
