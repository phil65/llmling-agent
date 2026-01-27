"""Configuration models for Codex agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field

from agentpool.models.agents import AnyToolConfig  # noqa: TC001
from agentpool.models.fields import OutputTypeField  # noqa: TC001
from agentpool_config.nodes import BaseAgentConfig
from codex_adapter import ApprovalPolicy, ReasoningEffort, SandboxMode  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentpool.agents.codex_agent import CodexAgent
    from agentpool.common_types import AnyEventHandlerType
    from agentpool.delegation import AgentPool
    from agentpool.resource_providers import ResourceProvider
    from agentpool.ui.base import InputProvider


class CodexAgentConfig(BaseAgentConfig):
    """Configuration for Codex app-server agent.

    Wraps the Codex app-server via its JSON-RPC protocol, enabling
    integration with agentpool's agent ecosystem.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "x-icon": "octicon:code-16",
            "x-doc-title": "Codex Agent Configuration",
        }
    )

    type: Literal["codex"] = Field("codex", init=False)
    """Top-level discriminator for agent type."""

    model: str | None = Field(
        default=None,
        title="Model",
        examples=["gpt-5.1-codex-max", "gpt-5.1-codex-mini", "gpt-5.2"],
    )
    """Model to use for the Codex session.

    Codex app-server provides its own set of models (e.g., gpt-5.1-codex-max).
    Use the agent's get_available_models() method to fetch the current list from your server.
    """

    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        title="Reasoning Effort",
        examples=["medium", "high", "xhigh"],
    )
    """Reasoning effort level for the model."""

    approval_policy: ApprovalPolicy = Field(
        default="never",
        title="Approval Policy",
        examples=["never", "on-request", "on-failure", "untrusted"],
    )
    """Tool call approval policy.

    - "never": Execute tools without requesting approval (default for programmatic use)
    - "on-request": Request approval when tools request it
    - "on-failure": Request approval when tool execution fails
    - "untrusted": Always request approval (most restrictive)
    """

    sandbox: SandboxMode = Field(
        default="workspace-write",
        title="Sandbox Mode",
        examples=["workspace-write", "read-only", "danger-full-access"],
    )
    """Sandbox mode for file operations.
    - "read-only": Can only read files, no modifications
    - "workspace-write": Can write within workspace directory (default if not specified)
    - "danger-full-access": Full filesystem access (use with caution)
    - "externalSandbox": Use external sandbox environment

    If not specified, Codex uses its default sandbox policy.
    """

    base_instructions: str | None = Field(default=None, title="Base Instructions")
    """Base system instructions for the Codex session."""

    developer_instructions: str | None = Field(default=None, title="Developer Instructions")
    """Developer-provided instructions for the Codex session."""

    tools: list[AnyToolConfig | str] = Field(
        default_factory=list,
        title="Tools",
        examples=[[{"type": "agent_management"}, "webbrowser:open"]],
    )
    """Tools and toolsets to expose to Codex via MCP bridge.

    These will be started as an in-process MCP server and made available
    to the Codex agent.
    """

    output_type: OutputTypeField = None
    """Structured output type for responses."""

    def get_agent[TDeps](
        self,
        event_handlers: Sequence[AnyEventHandlerType] | None = None,
        input_provider: InputProvider | None = None,
        pool: AgentPool[Any] | None = None,
        deps_type: type[TDeps] | None = None,  # type: ignore[valid-type]
    ) -> CodexAgent[TDeps, Any]:
        from agentpool.agents.codex_agent import CodexAgent

        return CodexAgent.from_config(
            self,
            event_handlers=event_handlers,
            input_provider=input_provider,
            agent_pool=pool,
            deps_type=deps_type,
        )

    def get_tool_providers(self) -> list[ResourceProvider]:
        """Get all resource providers for this agent's tools.

        Processes the unified tools list, separating:
        - Toolsets: Each becomes its own ResourceProvider
        - Single tools: Aggregated into a single StaticResourceProvider

        Returns:
            List of ResourceProvider instances
        """
        from agentpool import log
        from agentpool.resource_providers import StaticResourceProvider
        from agentpool.tools.base import Tool
        from agentpool_config import BaseToolConfig
        from agentpool_config.toolsets import BaseToolsetConfig

        logger = log.get_logger(__name__)
        providers = []
        static_tools: list[Tool] = []

        for tool_config in self.tools:
            try:
                if isinstance(tool_config, BaseToolsetConfig):
                    providers.append(tool_config.get_provider())
                elif isinstance(tool_config, str):
                    static_tools.append(Tool.from_callable(tool_config))
                elif isinstance(tool_config, BaseToolConfig):
                    static_tools.append(tool_config.get_tool())
            except Exception:
                logger.exception("Failed to load tool", config=tool_config)
                continue

        if static_tools:
            providers.append(StaticResourceProvider(tools=static_tools))

        return providers
