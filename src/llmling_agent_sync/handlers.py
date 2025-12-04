"""Built-in change handlers for reconciliation."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent import Agent
    from llmling_agent_sync.models import FileChange


class AgentReconciler:
    """Handler that uses an Agent to reconcile file changes.

    Constructs a prompt from the FileChange metadata and sends it to the agent.
    """

    def __init__(
        self,
        agent: Agent,
        include_file_content: bool = True,
        max_diff_lines: int = 500,
    ):
        """Initialize the reconciler.

        Args:
            agent: The agent to use for reconciliation
            include_file_content: Whether to include current file content in prompt
            max_diff_lines: Maximum diff lines to include (truncate if larger)
        """
        self.agent = agent
        self.include_file_content = include_file_content
        self.max_diff_lines = max_diff_lines

    async def __call__(self, change: FileChange) -> str | None:
        """Process a file change and return updated content.

        Args:
            change: The file change to process

        Returns:
            New file content if changes needed, None otherwise
        """
        prompt = self._build_prompt(change)
        result = await self.agent.run(prompt)
        return result.data if result.data else None

    def _build_prompt(self, change: FileChange) -> str:
        """Build the reconciliation prompt from change data."""
        parts: list[str] = []

        # Base instruction
        base_instruction = change.metadata.prompt or (
            "Review this file and update it if needed based on the dependency changes."
        )
        parts.append(base_instruction)

        # File being reviewed
        parts.append(f"\n## File to review\n`{change.path}`")

        # Include current file content if requested
        if self.include_file_content:
            try:
                from pathlib import Path

                content = Path(change.path).read_text()
                parts.append(f"\n## Current content\n```\n{content}\n```")
            except (OSError, UnicodeDecodeError):
                pass

        # Changed dependencies
        parts.append(f"\n## Changed dependencies\n{', '.join(change.changed_deps)}")

        # Git diff (potentially truncated)
        if change.diff:
            diff = change.diff
            diff_lines = diff.splitlines()
            if len(diff_lines) > self.max_diff_lines:
                diff = "\n".join(diff_lines[: self.max_diff_lines])
                diff += f"\n\n... (truncated, {len(diff_lines) - self.max_diff_lines} more lines)"

            parts.append(f"\n## Git Diff\n```diff\n{diff}\n```")

        # Changed URLs with content
        if change.changed_urls:
            parts.append(f"\n## Changed URLs\n{', '.join(change.changed_urls)}")

            for url in change.changed_urls:
                url_content = change.url_contents.get(url)
                if url_content:
                    # Truncate long content
                    content_lines = url_content.splitlines()
                    if len(content_lines) > self.max_diff_lines:
                        url_content = "\n".join(content_lines[: self.max_diff_lines])
                        url_content += f"\n\n... (truncated, {len(content_lines) - self.max_diff_lines} more lines)"  # noqa: E501
                    parts.append(f"\n### Content: {url}\n```\n{url_content}\n```")

        # Reference URLs
        if change.metadata.urls:
            urls_formatted = "\n".join(f"- {url}" for url in change.metadata.urls)
            parts.append(f"\n## Reference URLs\n{urls_formatted}")

        # Additional context
        if change.metadata.context:
            context_formatted = "\n".join(
                f"- **{k}**: {v}" for k, v in change.metadata.context.items()
            )
            parts.append(f"\n## Additional context\n{context_formatted}")

        # Response instruction
        parts.append(
            "\n## Instructions\n"
            "If the file needs updates based on the dependency changes, "
            "respond with the complete updated file content.\n"
            "If no changes are needed, respond with exactly: NO_CHANGES_NEEDED"
        )

        return "\n".join(parts)


class LoggingHandler:
    """Simple handler that logs changes without modifying files.

    Useful for dry-run/preview scenarios.
    """

    def __init__(self, logger: Callable[[str], None] | None = None):
        """Initialize the handler.

        Args:
            logger: Optional logging function (defaults to print)
        """
        self._logger = logger or print

    async def __call__(self, change: FileChange) -> str | None:
        """Log the change and return None (no modifications)."""
        self._logger(f"File needs review: {change.path}")
        self._logger(f"  Dependencies: {change.changed_deps}")
        if change.metadata.prompt:
            self._logger(f"  Prompt: {change.metadata.prompt}")
        return None
