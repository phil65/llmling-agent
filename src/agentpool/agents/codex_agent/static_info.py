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
        id="unlessTrusted",
        name="Unless Trusted",
        description="Auto-approve trusted operations, ask for others",
        category_id="mode",
    ),
    ModeInfo(
        id="auto",
        name="Auto-Approve Safe",
        description="Auto-approve low-risk tools, ask for high-risk",
        category_id="mode",
    ),
    ModeInfo(
        id="always",
        name="Always Confirm",
        description="Request approval before executing any tool",
        category_id="mode",
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
