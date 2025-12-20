"""Command hook implementation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmling_agent.hooks.base import Hook, HookResult
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.hooks.base import HookEvent, HookInput


logger = get_logger(__name__)


class CommandHook(Hook):
    """Hook that executes a shell command.

    The command receives hook input as JSON via stdin and should return
    JSON output via stdout.

    Exit codes:
    - 0: Success, stdout parsed as JSON for result
    - 2: Block/deny, stderr used as reason
    - Other: Non-blocking error, logged but execution continues
    """

    def __init__(
        self,
        event: HookEvent,
        command: str,
        matcher: str | None = None,
        timeout: float = 60.0,
        enabled: bool = True,
        env: dict[str, str] | None = None,
    ):
        """Initialize command hook.

        Args:
            event: The lifecycle event this hook handles.
            command: Shell command to execute.
            matcher: Regex pattern for matching.
            timeout: Maximum execution time in seconds.
            enabled: Whether this hook is active.
            env: Additional environment variables.
        """
        super().__init__(event=event, matcher=matcher, timeout=timeout, enabled=enabled)
        self.command = command
        self.env = env or {}

    async def execute(self, input_data: HookInput) -> HookResult:
        """Execute the shell command.

        Args:
            input_data: The hook input data, passed as JSON to stdin.

        Returns:
            Hook result parsed from command output.
        """
        # Prepare environment
        env = os.environ.copy()
        env.update(self.env)

        # Expand $PROJECT_DIR if present
        command = self.command
        if "$PROJECT_DIR" in command:
            project_dir = env.get("PROJECT_DIR", Path.cwd())
            command = command.replace("$PROJECT_DIR", str(project_dir))

        # Serialize input
        input_json = json.dumps(dict(input_data))

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input_json.encode()),
                timeout=self.timeout,
            )

            stdout_str = stdout.decode().strip()
            stderr_str = stderr.decode().strip()

            # Handle exit codes
            if proc.returncode == 0:
                return _parse_success_output(stdout_str)
            if proc.returncode == 2:  # noqa: PLR2004
                # Blocking error
                reason = stderr_str or "Hook denied the operation"
                return HookResult(decision="deny", reason=reason)
            # Non-blocking error
            logger.warning("Hook command failed", returncode=proc.returncode, stderr=stderr_str)
            return HookResult(decision="allow")

        except TimeoutError:
            logger.exception("Hook command timed out", timeout=self.timeout, command=command)
            return HookResult(decision="allow")
        except Exception as e:
            logger.exception("Hook command failed", command=command)
            return HookResult(decision="allow", reason=str(e))


def _parse_success_output(stdout: str) -> HookResult:
    """Parse successful command output.

    Args:
        stdout: Command stdout.

    Returns:
        Parsed hook result.
    """
    if not stdout:
        return HookResult(decision="allow")

    try:
        data = json.loads(stdout)
        return _normalize_result(data)
    except json.JSONDecodeError:
        # Plain text output treated as additional context
        return HookResult(decision="allow", additional_context=stdout)


def _normalize_result(data: dict[str, Any]) -> HookResult:
    """Normalize command output to HookResult.

    Args:
        data: Parsed JSON data.

    Returns:
        Normalized hook result.
    """
    result: HookResult = {}

    # Handle decision field (support various naming conventions)
    decision = data.get("decision") or data.get("permissionDecision")
    if decision:
        # Normalize decision values
        if decision in ("approve", "allow"):
            result["decision"] = "allow"
        elif decision in ("block", "deny"):
            result["decision"] = "deny"
        elif decision == "ask":
            result["decision"] = "ask"

    # Handle reason field
    reason = data.get("reason") or data.get("permissionDecisionReason")
    if reason:
        result["reason"] = reason

    # Handle modified input
    if "modified_input" in data:
        result["modified_input"] = data["modified_input"]
    elif "updatedInput" in data:
        result["modified_input"] = data["updatedInput"]

    # Handle additional context
    if "additional_context" in data:
        result["additional_context"] = data["additional_context"]
    elif "additionalContext" in data:
        result["additional_context"] = data["additionalContext"]

    # Handle continue flag
    if "continue" in data:
        result["continue_"] = data["continue"]
    elif "continue_" in data:
        result["continue_"] = data["continue_"]

    return result
