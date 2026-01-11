"""Static model information."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokonomics.model_discovery.model_info import ModelInfo, ModelPricing

from agentpool.agents.modes import ModeInfo


if TYPE_CHECKING:
    from claude_agent_sdk import PermissionMode


VALID_MODES: set[PermissionMode] = {
    "default",
    "acceptEdits",
    "plan",
    "bypassPermissions",
}

# Static Claude Code models - these are the simple IDs the SDK accepts
# Use id_override to ensure pydantic_ai_id returns simple names like "opus"

OPUS = ModelInfo(
    id="claude-opus-4-5",
    name="Claude Opus",
    provider="anthropic",
    description="Claude Opus - most capable model",
    context_window=200000,
    max_output_tokens=32000,
    input_modalities={"text", "image"},
    output_modalities={"text"},
    pricing=ModelPricing(
        prompt=0.000005,  # $5 per 1M tokens
        completion=0.000025,  # $25 per 1M tokens
    ),
    id_override="opus",  # Claude Code SDK uses simple names
)
SONNET = ModelInfo(
    id="claude-sonnet-4-5",
    name="Claude Sonnet",
    provider="anthropic",
    description="Claude Sonnet - balanced performance and speed",
    context_window=200000,
    max_output_tokens=16000,
    input_modalities={"text", "image"},
    output_modalities={"text"},
    pricing=ModelPricing(
        prompt=0.000003,  # $3 per 1M tokens
        completion=0.000015,  # $15 per 1M tokens
    ),
    id_override="sonnet",  # Claude Code SDK uses simple names
)
HAIKU = ModelInfo(
    id="claude-haiku-4-5",
    name="Claude Haiku",
    provider="anthropic",
    description="Claude Haiku - fast and cost-effective",
    context_window=200000,
    max_output_tokens=8000,
    input_modalities={"text", "image"},
    output_modalities={"text"},
    pricing=ModelPricing(
        prompt=0.000001,  # $1.00 per 1M tokens
        completion=0.000005,  # $5 per 1M tokens
    ),
    id_override="haiku",  # Claude Code SDK uses simple names
)

MODELS = [OPUS, SONNET, HAIKU]


MODES = [
    ModeInfo(
        id="default",
        name="Default",
        description="Require confirmation for tool usage",
        category_id="permissions",
    ),
    ModeInfo(
        id="acceptEdits",
        name="Accept Edits",
        description="Auto-approve file edits without confirmation",
        category_id="permissions",
    ),
    ModeInfo(
        id="plan",
        name="Plan",
        description="Planning mode - no tool execution",
        category_id="permissions",
    ),
    ModeInfo(
        id="bypassPermissions",
        name="Bypass Permissions",
        description="Skip all permission checks (use with caution)",
        category_id="permissions",
    ),
]
