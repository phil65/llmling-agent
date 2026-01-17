from __future__ import annotations

from agentpool.agents.modes import ModeInfo


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
        id="externalSandbox",
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
