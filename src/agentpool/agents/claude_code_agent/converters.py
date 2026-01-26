"""Claude Agent SDK to native event converters.

This module provides conversion from Claude Agent SDK message types to native
agentpool streaming events, enabling ClaudeCodeAgent to yield the same
event types as native agents.
"""

from __future__ import annotations

from difflib import unified_diff
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never, cast
import uuid

from pydantic import TypeAdapter
from pydantic_ai import PartDeltaEvent, TextPartDelta, ThinkingPartDelta

from agentpool.agents.events import ToolCallCompleteEvent, ToolCallStartEvent


if TYPE_CHECKING:
    from clawd_code_sdk import ContentBlock, McpServerConfig, Message, PermissionResult
    from clawd_code_sdk.types import SystemPromptPreset

    from agentpool.agents.context import ConfirmationResult
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool_config.mcp_server import MCPServerConfig as NativeMCPServerConfig


def content_block_to_event(block: ContentBlock, index: int = 0) -> RichAgentStreamEvent[Any] | None:
    """Convert a Claude SDK ContentBlock to a streaming event.

    Args:
        block: Claude SDK content block
        index: Part index for the event

    Returns:
        Corresponding streaming event, or None if not mappable
    """
    from clawd_code_sdk import TextBlock, ThinkingBlock, ToolUseBlock

    from agentpool.agents.events.infer_info import derive_rich_tool_info

    match block:
        case TextBlock(text=text):
            return PartDeltaEvent(index=index, delta=TextPartDelta(content_delta=text))
        case ThinkingBlock(thinking=thinking):
            return PartDeltaEvent(index=index, delta=ThinkingPartDelta(content_delta=thinking))
        case ToolUseBlock(id=tool_id, name=name, input=input_data):
            rich_info = derive_rich_tool_info(name, input_data)
            return ToolCallStartEvent(
                tool_call_id=tool_id,
                tool_name=name,
                title=rich_info.title,
                kind=rich_info.kind,
                locations=rich_info.locations,
                content=rich_info.content,
                raw_input=input_data,
            )
        case _:
            return None


def confirmation_result_to_native(result: ConfirmationResult) -> PermissionResult:
    from clawd_code_sdk import PermissionResultAllow, PermissionResultDeny

    match result:
        case "allow":
            return PermissionResultAllow()
        case "skip":
            return PermissionResultDeny(message="User skipped tool execution")
        case "abort_run" | "abort_chain":
            return PermissionResultDeny(message="User aborted execution", interrupt=True)
        case _ as unreachable:
            raise assert_never(unreachable)


def to_claude_system_prompt(
    system_prompt: str, include_default: bool = True
) -> SystemPromptPreset | str:
    from clawd_code_sdk.types import SystemPromptPreset

    if include_default:
        # Use SystemPromptPreset to append to builtin prompt
        return SystemPromptPreset(type="preset", preset="claude_code", append=system_prompt)
    return system_prompt


def claude_message_to_events(
    message: Message,
    agent_name: str = "",
) -> list[RichAgentStreamEvent[Any]]:
    """Convert a Claude SDK Message to a list of streaming events.

    Args:
        message: Claude SDK message (UserMessage, AssistantMessage, etc.)
        agent_name: Name of the agent for event attribution

    Returns:
        List of corresponding streaming events
    """
    from clawd_code_sdk import AssistantMessage, ToolResultBlock, ToolUseBlock

    pending_tool_calls = {}
    events: list[RichAgentStreamEvent[Any]] = []
    match message:
        case AssistantMessage(content=content):
            for idx, block in enumerate(content):
                # Track tool use blocks for later pairing with results
                if isinstance(block, ToolUseBlock):
                    pending_tool_calls[block.id] = block

                # Handle tool results - pair with pending tool call
                if isinstance(block, ToolResultBlock):
                    if tool_use := pending_tool_calls.pop(block.tool_use_id, None):
                        complete_event = ToolCallCompleteEvent(
                            tool_name=tool_use.name,
                            tool_call_id=block.tool_use_id,
                            tool_input=tool_use.input,
                            tool_result=block.content,
                            agent_name=agent_name,
                            message_id="",
                        )
                        events.append(complete_event)
                    continue

                # Convert other blocks to events
                if event := content_block_to_event(block, index=idx):
                    events.append(event)

        case _:
            # UserMessage, SystemMessage, ResultMessage - no events to emit
            pass

    return events


def convert_mcp_servers_to_sdk_format(
    mcp_servers: list[NativeMCPServerConfig],
) -> dict[str, McpServerConfig]:
    """Convert internal MCPServerConfig to Claude SDK format.

    Returns:
        Dict mapping server names to SDK-compatible config dicts
    """
    from urllib.parse import urlparse

    from clawd_code_sdk import McpServerConfig

    from agentpool_config.mcp_server import (
        SSEMCPServerConfig,
        StdioMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )

    result: dict[str, McpServerConfig] = {}

    for idx, server in enumerate(mcp_servers):
        # Determine server name
        if server.name:
            name = server.name
        elif isinstance(server, StdioMCPServerConfig) and server.args:
            name = server.args[-1].split("/")[-1].split("@")[0]
        elif isinstance(server, StdioMCPServerConfig):
            name = server.command
        elif isinstance(server, SSEMCPServerConfig | StreamableHTTPMCPServerConfig):
            name = urlparse(str(server.url)).hostname or f"server_{idx}"
        else:
            name = f"server_{idx}"

        # Build SDK-compatible config
        config: dict[str, Any]
        match server:
            case StdioMCPServerConfig(command=command, args=args):
                config = {"type": "stdio", "command": command, "args": args}
                if server.env:
                    config["env"] = server.get_env_vars()
            case SSEMCPServerConfig(url=url):
                config = {"type": "sse", "url": str(url)}
                if server.headers:
                    config["headers"] = server.headers
            case StreamableHTTPMCPServerConfig(url=url):
                config = {"type": "http", "url": str(url)}
                if server.headers:
                    config["headers"] = server.headers

        result[name] = cast(McpServerConfig, config)

    return result


def to_output_format(output_type: type) -> dict[str, Any] | None:
    """Convert to SDK output format dict."""
    # Build structured output format if needed
    output_format: dict[str, Any] | None = None
    if output_type is not str:
        adapter = TypeAdapter[Any](output_type)
        schema = adapter.json_schema()
        output_format = {"type": "json_schema", "schema": schema}
    return output_format


def convert_to_opencode_metadata(  # noqa: PLR0911
    tool_name: str,
    tool_use_result: dict[str, Any] | str | None,
    tool_input: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Convert Claude Code SDK tool_use_result to OpenCode metadata format."""
    # Handle None or string results (bash errors come as plain strings)
    if tool_use_result is None or not isinstance(tool_use_result, dict):
        return None

    # Dispatch to appropriate converter based on tool name
    match tool_name.lower():
        case "write":
            return _convert_write_result(tool_use_result)
        case "edit":
            return _convert_edit_result(tool_use_result)
        case "read":
            return _convert_read_result(tool_use_result)
        case "bash":
            return _convert_bash_result(tool_use_result, tool_input)
        case "todowrite":
            return _convert_todowrite_result(tool_use_result)
        case _:
            return None


def _convert_write_result(result: dict[str, Any]) -> dict[str, Any] | None:
    """Convert Write tool result to OpenCode metadata."""
    file_path = result.get("filePath")
    content = result.get("content")
    if not file_path or content is None:
        return None
    return {"filePath": file_path, "content": content}


def _convert_edit_result(result: dict[str, Any]) -> dict[str, Any] | None:
    """Convert Edit tool result to OpenCode metadata."""
    file_path = result.get("filePath")
    original_file = result.get("originalFile")
    old_string = result.get("oldString", "")
    new_string = result.get("newString", "")
    structured_patch = result.get("structuredPatch", [])

    if not file_path:
        return None

    # Compute the "after" content by applying the edit
    after_content = original_file
    if original_file is not None and old_string and new_string:
        after_content = original_file.replace(old_string, new_string, 1)

    # Build unified diff from structuredPatch or compute it
    diff = _build_unified_diff(file_path, original_file, after_content, structured_patch)
    # Count additions and deletions
    additions, deletions = _count_diff_changes(structured_patch)
    return {
        "diff": diff,
        "filediff": {
            "file": file_path,
            "before": original_file or "",
            "after": after_content or "",
            "additions": additions,
            "deletions": deletions,
        },
    }


def _convert_read_result(result: dict[str, Any]) -> dict[str, Any] | None:
    """Convert Read tool result to OpenCode metadata."""
    # Read results have a nested "file" object
    file_info = result.get("file")
    if not file_info:
        return None

    return {
        "filePath": file_info.get("filePath"),
        "content": file_info.get("content"),
        "numLines": file_info.get("numLines"),
        "startLine": file_info.get("startLine"),
        "totalLines": file_info.get("totalLines"),
    }


def _convert_bash_result(
    result: dict[str, Any],
    tool_input: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert Bash tool result to OpenCode metadata."""
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    # Combine stdout and stderr
    output = stdout
    if stderr:
        output = f"{stdout}\n{stderr}" if stdout else stderr
    # Get description from tool input (Claude Code uses "description" field)
    description = ""
    if tool_input:
        description = tool_input.get("description", tool_input.get("command", ""))

    # Note: Claude Code SDK doesn't provide exit code in the success result structure,
    # it's only available in error strings. For successful commands, exit is 0.
    # The SDK result doesn't have an exit_code field, so we infer:
    # - If we got here with a dict result, the command likely succeeded (exit 0)
    # - Errors come as strings, not dicts
    exit_code: int | None = 0
    if result.get("interrupted"):
        exit_code = None  # Interrupted commands don't have a clean exit code
    return {"output": output, "exit": exit_code, "description": description}


def _convert_todowrite_result(result: dict[str, Any]) -> dict[str, Any] | None:
    """Convert TodoWrite tool result to OpenCode metadata."""
    new_todos = result.get("newTodos", [])
    if not new_todos:
        return None

    # Convert to OpenCode format, generating IDs and inferring priority
    todos = []
    for i, todo in enumerate(new_todos):
        content = todo.get("content", "")
        status = todo.get("status", "pending")

        # Infer priority from position (first items = higher priority)
        # or from content keywords
        priority = _infer_priority(content, i, len(new_todos))

        todos.append({
            "id": str(uuid.uuid4()),
            "content": content,
            "status": status,
            "priority": priority,
        })

    return {"todos": todos}


# Priority thresholds for position-based inference
_HIGH_PRIORITY_THRESHOLD = 0.33
_MEDIUM_PRIORITY_THRESHOLD = 0.67


def _infer_priority(content: str, index: int, total: int) -> str:
    """Infer priority from content keywords or position."""
    content_lower = content.lower()

    # Check for explicit priority keywords
    high_keywords = ("critical", "urgent", "asap", "immediately", "important", "soon", "priority")
    low_keywords = ("later", "eventually", "low priority", "nice to have")

    if any(kw in content_lower for kw in high_keywords):
        return "high"
    if any(kw in content_lower for kw in low_keywords):
        return "low"

    # Fall back to position-based priority
    # First third = high, middle third = medium, last third = low
    if total <= 1:
        return "medium"
    position_ratio = index / (total - 1) if total > 1 else 0
    if position_ratio < _HIGH_PRIORITY_THRESHOLD:
        return "high"
    if position_ratio < _MEDIUM_PRIORITY_THRESHOLD:
        return "medium"
    return "low"


def _build_unified_diff(
    file_path: str,
    before: str | None,
    after: str | None,
    structured_patch: list[dict[str, Any]],
) -> str:
    """Build unified diff string from structured patch or content."""
    # If we have both before and after, compute proper diff
    if before is not None and after is not None:
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)

        # Ensure trailing newline for proper diff
        if before_lines and not before_lines[-1].endswith("\n"):
            before_lines[-1] += "\n"
        if after_lines and not after_lines[-1].endswith("\n"):
            after_lines[-1] += "\n"

        name = Path(file_path).name
        diff_lines = unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
        )
        return "".join(diff_lines)

    # Fallback: reconstruct from structuredPatch
    if structured_patch:
        return _structured_patch_to_diff(file_path, structured_patch)

    return ""


def _structured_patch_to_diff(
    file_path: str,
    structured_patch: list[dict[str, Any]],
) -> str:
    """Convert Claude Code's structuredPatch to unified diff format.

    structuredPatch format:
        [
            {
                "oldStart": 1,
                "oldLines": 4,
                "newStart": 1,
                "newLines": 5,
                "lines": [" def hello_world():", "+    \"\"\"Docstring.\"\"\"", ...]
            }
        ]

    The lines array uses prefixes: " " (context), "+" (added), "-" (removed)
    """
    from pathlib import Path

    name = Path(file_path).name
    lines = [f"--- a/{name}", f"+++ b/{name}"]

    for hunk in structured_patch:
        old_start = hunk.get("oldStart", 1)
        old_lines = hunk.get("oldLines", 0)
        new_start = hunk.get("newStart", 1)
        new_lines = hunk.get("newLines", 0)
        hunk_lines = hunk.get("lines", [])

        # Add hunk header
        lines.append(f"@@ -{old_start},{old_lines} +{new_start},{new_lines} @@")

        # Add the diff lines (already prefixed with ' ', '+', or '-')
        lines.extend(hunk_lines)

    return "\n".join(lines) + "\n" if lines else ""


def _count_diff_changes(structured_patch: list[dict[str, Any]]) -> tuple[int, int]:
    """Count additions and deletions from structured patch."""
    additions = 0
    deletions = 0

    for hunk in structured_patch:
        for line in hunk.get("lines", []):
            if line.startswith("+"):
                additions += 1
            elif line.startswith("-"):
                deletions += 1

    return additions, deletions


def build_sdk_hooks_from_agent_hooks(
    hooks: Any,  # AgentHooks
    agent_name: str,
) -> dict[str, list[Any]]:
    """Convert AgentHooks to Claude SDK hooks format.

    Args:
        hooks: AgentHooks instance with pre/post tool hooks
        agent_name: Name of the agent for context

    Returns:
        Dictionary mapping hook event names to HookMatcher lists
    """
    from clawd_code_sdk.types import HookMatcher

    result: dict[str, list[Any]] = {}

    if not hooks:
        return result

    # Check if we have pre_tool_use hooks
    if hooks.pre_tool_use:

        async def on_pre_tool_use(
            input_data: Any,
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            """Adapter for pre_tool_use hooks."""
            tool_name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input", {})

            pre_result = await hooks.run_pre_tool_hooks(
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=tool_input,
                session_id=input_data.get("session_id"),
            )

            # Convert our hook result to SDK format
            decision = pre_result.get("decision")
            if decision == "deny":
                reason = pre_result.get("reason", "Blocked by pre-tool hook")
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }

            # Check for modified input
            output: dict[str, Any] = {}
            if modified := pre_result.get("modified_input"):
                output["hookSpecificOutput"] = {
                    "hookEventName": "PreToolUse",
                    "updatedInput": modified,
                }

            return output

        result["PreToolUse"] = [HookMatcher(matcher="*", hooks=[on_pre_tool_use])]  # type: ignore[list-item]

    # Check if we have post_tool_use hooks
    if hooks.post_tool_use:

        async def on_post_tool_use(
            input_data: Any,
            tool_use_id: str | None,
            context: Any,
        ) -> dict[str, Any]:
            """Adapter for post_tool_use hooks."""
            tool_name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input", {})
            tool_response = input_data.get("tool_response")

            await hooks.run_post_tool_hooks(
                agent_name=agent_name,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_response,
                duration_ms=0,  # SDK doesn't provide timing
                session_id=input_data.get("session_id"),
            )

            # Post hooks are observation-only in SDK, can add context
            return {}

        result["PostToolUse"] = [HookMatcher(matcher="*", hooks=[on_post_tool_use])]  # type: ignore[list-item]

    return result
