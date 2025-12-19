"""Configuration for file-based agent definitions.

Supports loading agents from markdown files with YAML frontmatter in various formats:
- Claude Code: https://code.claude.com/docs/en/sub-agents.md
- OpenCode: https://github.com/sst/opencode
- LLMling (native): Full AgentConfig fields in frontmatter
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field
from schemez import Schema


FileAgentFormat = Literal["claude", "opencode", "llmling", "auto"]


class FileAgentConfig(Schema):
    """Configuration for a file-based agent definition.

    Allows explicit configuration of file-based agents instead of just a path string.
    Useful for overriding auto-detection or adding future options.

    Example:
        ```yaml
        file_agents:
          # Simple path (auto-detect format)
          reviewer: .claude/agents/reviewer.md

          # Explicit config
          debugger:
            path: ./agents/debugger.md
            format: opencode
        ```
    """

    path: str = Field(
        ...,
        description="Path to the agent markdown file (local or remote via UPath)",
        examples=[".claude/agents/reviewer.md", "https://example.com/agents/helper.md"],
    )

    format: FileAgentFormat = Field(
        default="auto",
        description="File format to use for parsing. 'auto' detects based on content.",
    )


# Type alias for manifest usage: either a simple path string or full config
FileAgentReference = Annotated[
    str | FileAgentConfig,
    Field(
        description="Agent file reference - either a path string or explicit config",
        examples=[
            ".claude/agents/reviewer.md",
            {"path": "./agents/debugger.md", "format": "opencode"},
        ],
    ),
]
