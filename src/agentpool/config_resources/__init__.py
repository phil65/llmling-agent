"""Package resources for AgentPool configuration."""

from __future__ import annotations

import importlib.resources
from typing import Final

_RESOURCES = importlib.resources.files("agentpool.config_resources")

AGENTS_TEMPLATE: Final[str] = str(_RESOURCES / "agents_template.yml")
"""Path to the agents template configuration."""

ACP_ASSISTANT: Final[str] = str(_RESOURCES / "acp_assistant.yml")
"""Path to default ACP assistant configuration."""

CLAUDE_CODE_ASSISTANT: Final[str] = str(_RESOURCES / "claude_code_agent.yml")
"""Path to default Claude code assistant configuration."""

__all__ = ["ACP_ASSISTANT", "AGENTS_TEMPLATE", "CLAUDE_CODE_ASSISTANT"]
