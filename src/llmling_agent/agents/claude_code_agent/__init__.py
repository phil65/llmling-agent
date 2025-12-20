"""Claude Code Agent - Native Claude Agent SDK integration.

This module provides an agent implementation that wraps the Claude Agent SDK's
ClaudeSDKClient for native integration with llmling-agent.
"""

from __future__ import annotations

from llmling_agent.agents.claude_code_agent.claude_code_agent import ClaudeCodeAgent

__all__ = ["ClaudeCodeAgent"]
