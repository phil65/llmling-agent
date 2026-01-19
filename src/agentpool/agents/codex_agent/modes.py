"""Mode categories for CodexAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from agentpool.agents.modes import ConfigOptionChanged, ModeCategoryProtocol, ModeInfo


if TYPE_CHECKING:
    from agentpool.agents.codex_agent.codex_agent import CodexAgent


# =============================================================================
# Mode definitions (static data)
# =============================================================================

POLICY_MODES = [
    ModeInfo(
        id="never",
        name="Auto-Execute",
        description="Execute tools without approval (default for programmatic use)",
        category_id="mode",
    ),
    ModeInfo(
        id="on-request",
        name="On Request",
        description="Ask for approval only when tool explicitly requests it",
        category_id="mode",
    ),
    ModeInfo(
        id="on-failure",
        name="On Failure",
        description="Ask for approval when a tool execution fails",
        category_id="mode",
    ),
    ModeInfo(
        id="untrusted",
        name="Always Confirm",
        description="Request approval before executing any tool",
        category_id="mode",
    ),
]

SANDBOX_MODES = [
    ModeInfo(
        id="read-only",
        name="Read Only",
        description="Sandbox with read-only file access",
        category_id="sandbox",
    ),
    ModeInfo(
        id="workspace-write",
        name="Workspace Write",
        description="Can write files within workspace directory",
        category_id="sandbox",
    ),
    ModeInfo(
        id="danger-full-access",
        name="Full Access",
        description="Full filesystem access (dangerous)",
        category_id="sandbox",
    ),
    ModeInfo(
        id="external-sandbox",
        name="External Sandbox",
        description="Use external sandbox environment",
        category_id="sandbox",
    ),
]

EFFORT_MODES = [
    ModeInfo(
        id="low",
        name="Low Effort",
        description="Fast responses with lighter reasoning",
        category_id="thought_level",
    ),
    ModeInfo(
        id="medium",
        name="Medium Effort",
        description="Balanced reasoning depth for everyday tasks",
        category_id="thought_level",
    ),
    ModeInfo(
        id="high",
        name="High Effort",
        description="Deep reasoning for complex problems",
        category_id="thought_level",
    ),
    ModeInfo(
        id="xhigh",
        name="Extra High Effort",
        description="Maximum reasoning depth for complex problems",
        category_id="thought_level",
    ),
]


# =============================================================================
# Mode category implementations
# =============================================================================


class CodexApprovalCategory(ModeCategoryProtocol["CodexAgent"]):
    """Approval policy mode category for Codex."""

    id: ClassVar[str] = "mode"
    name: ClassVar[str] = "Tool Approval"
    available_modes: ClassVar[list[ModeInfo]] = POLICY_MODES
    category: ClassVar[str] = "mode"

    def get_current(self, agent: CodexAgent) -> str:
        """Get current approval policy from agent."""
        return agent._approval_policy

    async def apply(self, agent: CodexAgent, mode_id: str) -> None:
        """Apply approval policy mode."""
        valid_ids = {m.id for m in self.available_modes}
        if mode_id not in valid_ids:
            msg = f"Invalid mode '{mode_id}' for category '{self.id}'. Valid: {valid_ids}"
            raise ValueError(msg)

        agent._approval_policy = mode_id  # type: ignore[assignment]
        agent.log.info("Approval policy changed", policy=mode_id)
        change = ConfigOptionChanged(config_id=self.id, value_id=mode_id)
        await agent.state_updated.emit(change)


class CodexEffortCategory(ModeCategoryProtocol["CodexAgent"]):
    """Reasoning effort mode category for Codex."""

    id: ClassVar[str] = "thought_level"
    name: ClassVar[str] = "Reasoning Effort"
    available_modes: ClassVar[list[ModeInfo]] = EFFORT_MODES
    category: ClassVar[str] = "thought_level"

    def get_current(self, agent: CodexAgent) -> str:
        """Get current reasoning effort from agent."""
        return agent._current_effort or "medium"

    async def apply(self, agent: CodexAgent, mode_id: str) -> None:
        """Apply reasoning effort mode."""
        if mode_id not in (valid_ids := {m.id for m in self.available_modes}):
            msg = f"Invalid mode {mode_id!r} for category {self.id!r}. Valid: {valid_ids}"
            raise ValueError(msg)
        # Just store it - effort is passed per-turn, no restart needed
        agent._current_effort = mode_id  # type: ignore[assignment]
        agent.log.info("Reasoning effort changed", effort=mode_id)
        change = ConfigOptionChanged(config_id=self.id, value_id=mode_id)
        await agent.state_updated.emit(change)


class CodexSandboxCategory(ModeCategoryProtocol["CodexAgent"]):
    """Sandbox mode category for Codex."""

    id: ClassVar[str] = "sandbox"
    name: ClassVar[str] = "Sandbox Mode"
    available_modes: ClassVar[list[ModeInfo]] = SANDBOX_MODES
    category: ClassVar[str] = "other"

    def get_current(self, agent: CodexAgent) -> str:
        """Get current sandbox mode from agent."""
        return agent._current_sandbox or "workspace-write"

    async def apply(self, agent: CodexAgent, mode_id: str) -> None:
        """Apply sandbox mode."""
        valid_ids = {m.id for m in self.available_modes}
        if mode_id not in valid_ids:
            msg = f"Invalid mode '{mode_id}' for category '{self.id}'. Valid: {valid_ids}"
            raise ValueError(msg)
        agent._current_sandbox = mode_id  # type: ignore[assignment]
        agent.log.info("Sandbox mode changed", sandbox=mode_id)
        change = ConfigOptionChanged(config_id=self.id, value_id=mode_id)
        await agent.state_updated.emit(change)


class CodexModelCategory(ModeCategoryProtocol["CodexAgent"]):
    """Model selection category for Codex."""

    id: ClassVar[str] = "model"
    name: ClassVar[str] = "Model"
    available_modes: ClassVar[list[ModeInfo]] = []  # Populated dynamically
    category: ClassVar[str] = "model"

    def get_current(self, agent: CodexAgent) -> str:
        """Get current model from agent."""
        return agent._current_model or ""

    async def apply(self, agent: CodexAgent, mode_id: str) -> None:
        """Apply model selection."""
        # Model validation is optional since models are dynamic
        agent._current_model = mode_id
        agent.log.info("Model changed", model=mode_id)
        change = ConfigOptionChanged(config_id=self.id, value_id=mode_id)
        await agent.state_updated.emit(change)
