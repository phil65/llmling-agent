"""Testing utilities for end-to-end ACP testing and CI integration.

This module provides:
- A lightweight test harness for running end-to-end tests against the agentpool
  ACP server using ACPAgent as the client
- GitHub CI integration for programmatically triggering and monitoring workflow runs

Example:
    ```python
    # ACP testing
    async def test_basic_prompt():
        async with acp_test_session("tests/fixtures/simple.yml") as agent:
            result = await agent.run("Say hello")
            assert result.content

    # CI testing
    async def test_commit_in_ci():
        result = await run_ci_tests("abc123")  # or "HEAD"
        assert result.all_passed
        print(result.summary())
    ```
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from evented.configs import EventConfig

    from agentpool.agents.acp_agent import ACPAgent
    from agentpool.common_types import BuiltinEventHandlerType, IndividualEventHandler


@asynccontextmanager
async def acp_test_session(
    config: str | Path | None = None,
    *,
    file_access: bool = True,
    terminal_access: bool = True,
    debug_messages: bool = False,
    debug_file: str | None = None,
    debug_commands: bool = False,
    agent: str | None = None,
    load_skills: bool = False,
    cwd: str | Path | None = None,
    event_configs: Sequence[EventConfig] | None = None,
    event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
) -> AsyncIterator[ACPAgent[Any]]:
    """Create an end-to-end ACP test session using agentpool as server.

    This context manager starts an ACPAgent connected to a agentpool serve-acp
    subprocess, enabling full round-trip testing of the ACP protocol.

    Args:
        config: Path to agent configuration YAML file. If None, uses default config.
        file_access: Enable file system access for agents.
        terminal_access: Enable terminal access for agents.
        debug_messages: Save raw JSON-RPC messages to debug file.
        debug_file: File path for JSON-RPC debug messages.
        debug_commands: Enable debug slash commands for testing.
        agent: Name of specific agent to use from config.
        load_skills: Load client-side skills from .claude/skills directory.
        cwd: Working directory for the ACP server subprocess.
        event_configs: Event configurations for the agent.
        event_handlers: Event handlers for the agent (e.g., ["detailed"] for logging).

    Yields:
        ACPAgent instance connected to the test server.

    Example:
        ```python
        async def test_echo():
            async with acp_test_session("my_config.yml") as agent:
                result = await agent.run("Hello!")
                assert "Hello" in result.content
        ```
    """
    from agentpool.agents.acp_agent import ACPAgent

    # Build command line arguments
    args = ["run", "agentpool", "serve-acp"]

    if config is not None:
        args.extend(["--config", str(config)])

    if not file_access:
        args.append("--no-file-access")

    if not terminal_access:
        args.append("--no-terminal-access")

    if debug_messages:
        args.append("--debug-messages")

    if debug_file:
        args.extend(["--debug-file", debug_file])

    if debug_commands:
        args.append("--debug-commands")

    if agent:
        args.extend(["--agent", agent])

    if not load_skills:
        args.append("--no-skills")

    working_dir = str(cwd) if cwd else str(Path.cwd())

    async with ACPAgent(
        command="uv",
        args=args,
        cwd=working_dir,
        event_configs=event_configs,
        event_handlers=event_handlers,
    ) as acp_agent:
        yield acp_agent


# --- GitHub CI Testing ---

CheckResult = Literal["success", "failure", "skipped", "cancelled", "pending"]


@dataclass
class CITestResult:
    """Result of a CI test run."""

    commit: str
    """The commit SHA that was tested."""

    run_id: int
    """GitHub Actions run ID."""

    run_url: str
    """URL to the workflow run."""

    lint: CheckResult = "pending"
    """Result of ruff check."""

    format: CheckResult = "pending"
    """Result of ruff format check."""

    typecheck: CheckResult = "pending"
    """Result of mypy type checking."""

    test: CheckResult = "pending"
    """Result of pytest."""

    duration_seconds: float = 0.0
    """Total duration of the CI run."""

    raw_jobs: list[dict[str, Any]] = field(default_factory=list)
    """Raw job data from GitHub API."""

    @property
    def all_passed(self) -> bool:
        """Check if all checks passed."""
        return all(
            result == "success" for result in [self.lint, self.format, self.typecheck, self.test]
        )

    @property
    def any_failed(self) -> bool:
        """Check if any check failed."""
        return any(
            result == "failure" for result in [self.lint, self.format, self.typecheck, self.test]
        )

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status_icons = {
            "success": "✓",
            "failure": "✗",
            "skipped": "○",
            "cancelled": "⊘",
            "pending": "…",
        }
        lines = [
            f"CI Results for {self.commit[:8]}",
            f"Run: {self.run_url}",
            "",
            f"  {status_icons[self.lint]} Lint (ruff check): {self.lint}",
            f"  {status_icons[self.format]} Format (ruff format): {self.format}",
            f"  {status_icons[self.typecheck]} Type check (mypy): {self.typecheck}",
            f"  {status_icons[self.test]} Tests (pytest): {self.test}",
            "",
            f"Duration: {self.duration_seconds:.1f}s",
        ]
        return "\n".join(lines)


def _run_gh(*args: str) -> str:
    """Run a gh CLI command and return output."""
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _resolve_commit(commit: str) -> str:
    """Resolve a commit reference to a full SHA."""
    if commit.upper() == "HEAD":
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    return commit


async def run_ci_tests(
    commit: str = "HEAD",
    *,
    repo: str | None = None,
    poll_interval: float = 10.0,
    timeout: float = 600.0,
) -> CITestResult:
    """Trigger CI tests for a commit and wait for results.

    This function triggers the test-commit.yml workflow via the GitHub CLI,
    polls for completion, and returns structured results.

    Args:
        commit: Commit SHA or "HEAD" to test. Defaults to HEAD.
        repo: Repository in "owner/repo" format. Auto-detected if None.
        poll_interval: Seconds between status checks. Defaults to 10.
        timeout: Maximum seconds to wait for completion. Defaults to 600 (10 min).

    Returns:
        CITestResult with individual check results.

    Raises:
        TimeoutError: If the workflow doesn't complete within timeout.
        subprocess.CalledProcessError: If gh CLI commands fail.

    Example:
        ```python
        result = await run_ci_tests("abc123")
        if result.all_passed:
            print("All checks passed!")
        else:
            print(result.summary())
        ```
    """
    import time

    commit_sha = _resolve_commit(commit)
    start_time = time.monotonic()

    # Build repo flag if specified
    repo_args = ["-R", repo] if repo else []

    # Trigger the workflow
    _run_gh(
        "workflow",
        "run",
        "test-commit.yml",
        "-f",
        f"commit={commit_sha}",
        *repo_args,
    )

    # Wait a moment for the run to be created
    await asyncio.sleep(2)

    # Find the run ID
    runs_json = _run_gh(
        "run",
        "list",
        "--workflow=test-commit.yml",
        "--json=databaseId,headSha,status,url",
        "--limit=5",
        *repo_args,
    )
    runs = json.loads(runs_json)

    # Find the run for our commit
    run_id: int | None = None
    run_url = ""
    for run in runs:
        # Match by commit SHA (workflow dispatch uses the branch HEAD, but we can match)
        if run["status"] in ("queued", "in_progress", "pending"):
            run_id = run["databaseId"]
            run_url = run["url"]
            break

    if run_id is None:
        msg = f"Could not find workflow run for commit {commit_sha}"
        raise RuntimeError(msg)

    # Poll for completion
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout:
            msg = f"Workflow run {run_id} did not complete within {timeout}s"
            raise TimeoutError(msg)

        run_json = _run_gh(
            "run",
            "view",
            str(run_id),
            "--json=status,conclusion,jobs",
            *repo_args,
        )
        run_data = json.loads(run_json)

        if run_data["status"] == "completed":
            break

        await asyncio.sleep(poll_interval)

    # Parse job results
    duration = time.monotonic() - start_time
    jobs = run_data.get("jobs", [])

    result = CITestResult(
        commit=commit_sha,
        run_id=run_id,
        run_url=run_url,
        duration_seconds=duration,
        raw_jobs=jobs,
    )

    # Map job names to results
    for job in jobs:
        name = job.get("name", "").lower()
        conclusion = job.get("conclusion", "pending")

        # Normalize conclusion to our type
        if conclusion not in ("success", "failure", "skipped", "cancelled"):
            conclusion = "pending"

        if "lint" in name and "format" not in name:
            result.lint = conclusion
        elif "format" in name:
            result.format = conclusion
        elif "type" in name or "mypy" in name:
            result.typecheck = conclusion
        elif "test" in name or "pytest" in name:
            result.test = conclusion

    return result


async def quick_ci_check(commit: str = "HEAD") -> bool:
    """Quick check if a commit passes all CI checks.

    Convenience wrapper around run_ci_tests that returns a simple boolean.

    Args:
        commit: Commit SHA or "HEAD" to test.

    Returns:
        True if all checks passed, False otherwise.
    """
    result = await run_ci_tests(commit)
    return result.all_passed
