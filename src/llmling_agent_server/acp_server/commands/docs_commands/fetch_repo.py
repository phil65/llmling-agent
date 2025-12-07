"""Fetch contents from a GitHub repository via UIThub."""

from __future__ import annotations

import os
import uuid

import httpx
from pydantic_ai import UserPromptPart
from slashed import CommandContext  # noqa: TC002

from llmling_agent.log import get_logger
from llmling_agent.messaging.context import NodeContext  # noqa: TC001
from llmling_agent_commands.base import NodeCommand
from llmling_agent_server.acp_server.session import ACPSession  # noqa: TC001


logger = get_logger(__name__)


class FetchRepoCommand(NodeCommand):
    """Fetch contents from a GitHub repository via UIThub.

    Retrieves repository contents with various filtering options
    and displays them in a structured format.
    """

    name = "fetch-repo"
    category = "docs"

    async def execute_command(  # noqa: PLR0915
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        repo: str,
        *,
        branch: str | None = None,
        path: str | None = None,
        include_dirs: list[str] | None = None,
        disable_genignore: bool = False,
        exclude_dirs: list[str] | None = None,
        exclude_extensions: list[str] | None = None,
        include_extensions: list[str] | None = None,
        include_line_numbers: bool = True,
        max_file_size: int | None = None,
        max_tokens: int | None = None,
        omit_files: bool = False,
        yaml_string: str | None = None,
    ) -> None:
        """Fetch contents from a GitHub repository.

        Args:
            ctx: Command context with ACP session
            repo: GitHub path (owner/repo)
            branch: Branch name (defaults to main if not provided)
            path: File or directory path within the repository
            include_dirs: List of directories to include
            disable_genignore: Disable .genignore filtering
            exclude_dirs: List of directories to exclude
            exclude_extensions: List of file extensions to exclude
            include_extensions: List of file extensions to include
            include_line_numbers: Include line numbers in HTML/markdown output
            max_file_size: Maximum file size in bytes
            max_tokens: Maximum number of tokens in response
            omit_files: Only return directory structure without file contents
            yaml_string: URL encoded YAML string of file hierarchy to include
        """
        session = ctx.context.data
        assert session

        # Generate tool call ID
        tool_call_id = f"fetch-repo-{uuid.uuid4().hex[:8]}"

        try:
            # Build URL
            base_url = f"https://uithub.com/{repo}"
            if branch:
                base_url += f"/tree/{branch}"
            if path:
                base_url += f"/{path}"

            # Build parameters
            params = {}
            api_key = os.getenv("UITHUB_API_KEY")
            if api_key:
                params["apiKey"] = api_key

            if include_dirs:
                params["dir"] = ",".join(include_dirs)
            if disable_genignore:
                params["disableGenignore"] = "true"
            if exclude_dirs:
                params["exclude-dir"] = ",".join(exclude_dirs)
            if exclude_extensions:
                params["exclude-ext"] = ",".join(exclude_extensions)
            if include_extensions:
                params["ext"] = ",".join(include_extensions)
            if not include_line_numbers:
                params["lines"] = "false"
            if max_file_size:
                params["maxFileSize"] = str(max_file_size)
            if max_tokens:
                params["maxTokens"] = str(max_tokens)
            if omit_files:
                params["omitFiles"] = "true"
            if yaml_string:
                params["yamlString"] = yaml_string

            # Start tool call
            display_path = f"{repo}"
            if branch:
                display_path += f"@{branch}"
            if path:
                display_path += f":{path}"

            await session.notifications.tool_call_start(
                tool_call_id=tool_call_id,
                title=f"Fetching repository: {display_path}",
                kind="fetch",
            )

            # Make async HTTP request
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    base_url,
                    params=params,
                    headers={"accept": "text/markdown"},
                    timeout=30.0,
                )
                response.raise_for_status()
                content = response.text

            # Stage the content for use in agent context
            staged_part = UserPromptPart(
                content=f"Repository contents from {display_path}:\n\n{content}"
            )
            session.add_staged_parts([staged_part])

            # Send successful result - wrap in code block for proper display
            staged_count = session.get_staged_parts_count()
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="completed",
                title=f"Repository {display_path} fetched and staged ({staged_count} total parts)",
                content=[f"```\n{content}\n```"],
            )

        except httpx.HTTPStatusError as e:
            logger.exception(
                "HTTP error fetching repository", repo=repo, status=e.response.status_code
            )
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="failed",
                title=f"HTTP {e.response.status_code}: Failed to fetch {repo}",
            )
        except httpx.RequestError as e:
            logger.exception("Request error fetching repository", repo=repo)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="failed",
                title=f"Network error: {e}",
            )
        except Exception as e:
            logger.exception("Unexpected error fetching repository", repo=repo)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="failed",
                title=f"Error: {e}",
            )
