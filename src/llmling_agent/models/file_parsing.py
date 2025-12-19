"""Parsing logic for file-based agent definitions.

Supports loading agents from markdown files with YAML frontmatter in various formats:
- Claude Code: https://code.claude.com/docs/en/sub-agents.md
- OpenCode: https://github.com/sst/opencode
- LLMling (native): Full NativeAgentConfig fields in frontmatter
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from upathtools import to_upath

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.models.agents import NativeAgentConfig
    from llmling_agent.models.file_agents import FileAgentConfig


logger = get_logger(__name__)


# Claude Code model alias mapping
CLAUDE_MODEL_ALIASES: dict[str, str] = {
    "sonnet": "anthropic:claude-sonnet-4-20250514",
    "opus": "anthropic:claude-opus-4-20250514",
    "haiku": "anthropic:claude-haiku-3-5-20241022",
}

# Claude Code permissionMode to ToolConfirmationMode mapping
PERMISSION_MODE_MAP: dict[str, Literal["always", "never", "per_tool"]] = {
    "default": "per_tool",
    "acceptEdits": "never",
    "bypassPermissions": "never",
    # 'plan' and 'ignore' don't map well, default to per_tool
}

# Fields that pass through directly to NativeAgentConfig
PASSTHROUGH_FIELDS = {
    "inherits",
    "toolsets",
    "session",
    "output_type",
    "retries",
    "output_retries",
    "end_strategy",
    "avatar",
    "config_file_path",
    "knowledge",
    "workers",
    "debug",
    "usage_limits",
    "tool_mode",
    "display_name",
    "triggers",
}


def extract_frontmatter(content: str, file_path: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and body from markdown content.

    Args:
        content: Full markdown file content
        file_path: Path for error messages

    Returns:
        Tuple of (metadata dict, system prompt body)

    Raises:
        ValueError: If frontmatter is missing or invalid
    """
    import yamling

    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
    if not frontmatter_match:
        msg = f"No YAML frontmatter found in {file_path}"
        raise ValueError(msg)

    try:
        metadata = yamling.load_yaml(frontmatter_match.group(1))
    except yamling.YAMLError as e:
        msg = f"Invalid YAML frontmatter in {file_path}: {e}"
        raise ValueError(msg) from e

    if not isinstance(metadata, dict):
        msg = f"YAML frontmatter must be a dictionary in {file_path}"
        raise ValueError(msg)  # noqa: TRY004

    system_prompt = content[frontmatter_match.end() :].strip()
    return metadata, system_prompt


def detect_format(metadata: dict[str, Any]) -> Literal["claude", "opencode", "native"]:
    """Detect the file format based on frontmatter content.

    Args:
        metadata: Parsed YAML frontmatter

    Returns:
        Detected format: "claude", "opencode", or "llmling"
    """
    # OpenCode indicators
    is_opencode = (
        any(key in metadata for key in ["mode", "temperature", "maxSteps", "disable"])
        or ("tools" in metadata and isinstance(metadata["tools"], dict))
        or ("permission" in metadata and isinstance(metadata["permission"], dict))
    )
    if is_opencode:
        return "opencode"

    # Native format indicators (llmling-agent specific fields)
    native_fields = {"toolsets", "session", "knowledge", "workers", "triggers"}
    if any(field in metadata for field in native_fields):
        return "native"

    # Default to Claude Code format
    return "claude"


def parse_claude_format(
    metadata: dict[str, Any],
    system_prompt: str,
    file_path: str,
    *,
    skills_registry: Any | None = None,
) -> dict[str, Any]:
    """Parse Claude Code format frontmatter.

    Args:
        metadata: Parsed YAML frontmatter
        system_prompt: Markdown body content
        file_path: Path for logging
        skills_registry: Optional skills registry for loading skills

    Returns:
        Dict of NativeAgentConfig kwargs
    """
    config_kwargs: dict[str, Any] = {}

    # Description
    if description := metadata.get("description"):
        config_kwargs["description"] = description

    # Model handling
    if model := metadata.get("model"):
        if model == "inherit":
            pass  # Leave as None, will use default
        elif model in CLAUDE_MODEL_ALIASES:
            config_kwargs["model"] = CLAUDE_MODEL_ALIASES[model]
        else:
            config_kwargs["model"] = model

    # Permission mode mapping
    if permission_mode := metadata.get("permissionMode"):
        if mapped := PERMISSION_MODE_MAP.get(permission_mode):
            config_kwargs["requires_tool_confirmation"] = mapped
        else:
            logger.warning(
                "Unknown permissionMode %r in %s, using default",
                permission_mode,
                file_path,
            )

    # Tools string format (comma-separated) - not yet supported
    if (tools := metadata.get("tools")) and isinstance(tools, str):
        logger.debug("Claude Code tools string %r in %s (not yet supported)", tools, file_path)

    # Skills handling
    if (skills_str := metadata.get("skills")) and skills_registry is not None:
        skill_names = [s.strip() for s in skills_str.split(",")]
        for skill_name in skill_names:
            if skill_name not in skills_registry:
                logger.warning(
                    "Skill %r from %s not found in registry, ignoring",
                    skill_name,
                    file_path,
                )

    # System prompt from markdown body
    if system_prompt:
        config_kwargs["system_prompts"] = [system_prompt]

    # Pass through llmling-agent specific fields
    for field in PASSTHROUGH_FIELDS:
        if field in metadata:
            config_kwargs[field] = metadata[field]

    return config_kwargs


def parse_opencode_format(
    metadata: dict[str, Any],
    system_prompt: str,
    file_path: str,
) -> dict[str, Any]:
    """Parse OpenCode format frontmatter.

    Args:
        metadata: Parsed YAML frontmatter
        system_prompt: Markdown body content
        file_path: Path for logging

    Returns:
        Dict of NativeAgentConfig kwargs
    """
    config_kwargs: dict[str, Any] = {}

    # Description
    if description := metadata.get("description"):
        config_kwargs["description"] = description

    # Model handling
    if model := metadata.get("model"):
        if model == "inherit":
            pass
        elif model in CLAUDE_MODEL_ALIASES:
            config_kwargs["model"] = CLAUDE_MODEL_ALIASES[model]
        else:
            config_kwargs["model"] = model

    # Temperature (logged, not directly supported)
    if temperature := metadata.get("temperature"):
        logger.debug(
            "OpenCode temperature %r in %s (not directly supported)", temperature, file_path
        )

    # MaxSteps (logged, not directly supported)
    if max_steps := metadata.get("maxSteps"):
        logger.debug("OpenCode maxSteps %r in %s (not directly supported)", max_steps, file_path)

    # Disable (logged, not directly supported)
    if disable := metadata.get("disable"):
        logger.debug("OpenCode disable %r in %s (not directly supported)", disable, file_path)

    # Mode (informational only)
    if mode := metadata.get("mode"):
        logger.debug("OpenCode mode %r in %s (informational only)", mode, file_path)

    # Permission handling (granular per-tool)
    if permission := metadata.get("permission"):
        edit_perm = permission.get("edit") if isinstance(permission, dict) else None
        if edit_perm in ("deny", "ask"):
            config_kwargs["requires_tool_confirmation"] = (
                "always" if edit_perm == "ask" else "never"
            )
        logger.debug("OpenCode permission %r in %s (partial mapping)", permission, file_path)

    # Tools dict format (not yet supported)
    if (tools := metadata.get("tools")) and isinstance(tools, dict):
        logger.debug("OpenCode tools dict %r in %s (not yet supported)", tools, file_path)

    # System prompt from markdown body
    if system_prompt:
        config_kwargs["system_prompts"] = [system_prompt]

    # Pass through llmling-agent specific fields
    for field in PASSTHROUGH_FIELDS:
        if field in metadata:
            config_kwargs[field] = metadata[field]

    return config_kwargs


def parse_native_format(
    metadata: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    """Parse native format frontmatter.

    This format allows full NativeAgentConfig fields in the frontmatter,
    with the markdown body used as system prompt.

    Args:
        metadata: Parsed YAML frontmatter
        system_prompt: Markdown body content

    Returns:
        Dict of NativeAgentConfig kwargs
    """
    # Start with all metadata (it's already in NativeAgentConfig format)
    config_kwargs = dict(metadata)

    # Add system prompt from body if present and not already defined
    if system_prompt and "system_prompts" not in config_kwargs:
        config_kwargs["system_prompts"] = [system_prompt]

    return config_kwargs


def parse_agent_file(
    file_path: str,
    *,
    file_format: Literal["claude", "opencode", "native", "auto"] = "auto",
    skills_registry: Any | None = None,
) -> NativeAgentConfig:
    """Parse agent markdown file to NativeAgentConfig.

    Supports Claude Code, OpenCode, and native formats with auto-detection.
    Also supports local and remote paths via UPath.

    Args:
        file_path: Path to .md file with YAML frontmatter (local or remote)
        file_format: File format to use ("auto" for detection, or explicit format)
        skills_registry: Optional skills registry for loading skills

    Returns:
        Parsed NativeAgentConfig

    Raises:
        ValueError: If file cannot be parsed
    """
    from llmling_agent.models.agents import NativeAgentConfig

    path = to_upath(file_path)
    content = path.read_text("utf-8")

    # Extract frontmatter and body
    metadata, system_prompt = extract_frontmatter(content, file_path)

    # Detect or use specified format
    detected_format = detect_format(metadata) if file_format == "auto" else file_format

    # Parse based on format
    if detected_format == "claude":
        config_kwargs = parse_claude_format(
            metadata, system_prompt, file_path, skills_registry=skills_registry
        )
    elif detected_format == "opencode":
        config_kwargs = parse_opencode_format(metadata, system_prompt, file_path)
    elif detected_format == "native":
        config_kwargs = parse_native_format(metadata, system_prompt)
    else:
        msg = f"Unknown format {detected_format!r} for {file_path}"
        raise ValueError(msg)

    return NativeAgentConfig(**config_kwargs)


def parse_file_agent_reference(
    reference: str | FileAgentConfig,
    *,
    skills_registry: Any | None = None,
) -> NativeAgentConfig:
    """Parse a file agent reference (path string or config) to NativeAgentConfig.

    Args:
        reference: Either a path string (auto-detect format) or FileAgentConfig
            with explicit type discriminator
        skills_registry: Optional skills registry for loading skills

    Returns:
        Parsed NativeAgentConfig
    """
    if isinstance(reference, str):
        # Simple path string: auto-detect format
        return parse_agent_file(reference, skills_registry=skills_registry)

    # Explicit config: use type as format
    return parse_agent_file(
        reference.path,
        file_format=reference.type,
        skills_registry=skills_registry,
    )
