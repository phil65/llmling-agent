"""Documentation and repository fetching slash commands for ACP sessions."""

from __future__ import annotations

import os
import uuid

import httpx
from slashed import CommandContext, SlashedCommand  # noqa: TC002

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent_acp.session import ACPSession  # noqa: TC001


try:
    from pydantic_ai.messages import UserPromptPart
except ImportError:
    from pydantic_ai import UserPromptPart


logger = get_logger(__name__)


class GetSourceCommand(SlashedCommand):
    """Get Python source code using dot notation.

    Uses the llmling importing.py utility to fetch source code
    for any Python object accessible via dot notation.
    """

    name = "get-source"
    category = "docs"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext[ACPSession]],
        dot_path: str,
        *,
        cwd: str | None = None,
    ):
        """Get Python source code for an object.

        Args:
            ctx: Command context with ACP session
            dot_path: Dot notation path to Python object (e.g., 'requests.Session.get')
            cwd: Working directory to run in (defaults to session cwd)
        """
        session = ctx.context.data
        assert session

        # Generate tool call ID
        tool_call_id = f"get-source-{uuid.uuid4().hex[:8]}"

        try:
            # Check if we have terminal access
            if not (
                session.client_capabilities
                and session.client_capabilities.terminal
                and session.acp_agent.terminal_access
            ):
                await session.notifications.send_agent_text(
                    "❌ **Terminal access not available for source code fetching**"
                )
                return

            # Find importing.py path (using local copy)
            import llmling_agent.utils.importing

            importing_py_path = llmling_agent.utils.importing.__file__

            # Start tool call
            await session.notifications.tool_call_start(
                tool_call_id=tool_call_id,
                title=f"Getting source: {dot_path}",
                kind="read",
            )

            # Create terminal to run importing.py as script
            create_response = await session.requests.create_terminal(
                command="uv",
                args=["run", importing_py_path, dot_path],
                cwd=cwd or session.cwd,
                env={},
            )
            terminal_id = create_response.terminal_id

            # Wait for command to complete
            exit_result = await session.requests.wait_for_terminal_exit(terminal_id)

            # Get the source output
            output_response = await session.requests.terminal_output(terminal_id)

            # Release terminal
            await session.requests.release_terminal(terminal_id)

            # Check if command succeeded
            if exit_result.exit_code != 0:
                error_msg = (
                    output_response.output.strip()
                    or f"Exit code: {exit_result.exit_code}"
                )
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="failed",
                    title=f"Failed to get source for {dot_path}: {error_msg}",
                )
                return

            source_content = output_response.output.strip()
            if not source_content:
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="completed",
                    title="No source code found",
                )
                return

            # Stage the source content for use in agent context
            staged_part = UserPromptPart(
                content=f"Python source code for {dot_path}:\n\n{source_content}"
            )
            session.add_staged_parts([staged_part])

            # Send successful result - wrap in code block for proper display
            staged_count = session.get_staged_parts_count()
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="completed",
                title=f"Source code fetched and staged ({staged_count} total parts)",
                content=[f"```python\n{source_content}\n```"],
            )

        except Exception as e:
            logger.exception("Unexpected error fetching source code", dot_path=dot_path)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="failed",
                title=f"Error: {e}",
            )


class GitDiffCommand(SlashedCommand):
    """Fetch git diff for a specific commit.

    Executes git diff client-side to get the complete diff
    and displays it in a structured format.
    """

    name = "git-diff"
    category = "docs"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext[ACPSession]],
        commit: str,
        *,
        base_commit: str | None = None,
        cwd: str | None = None,
    ):
        """Fetch git diff for a specific commit.

        Args:
            ctx: Command context with ACP session
            commit: Commit hash or reference to diff
            base_commit: Base commit to compare against (defaults to commit~1)
            cwd: Working directory to run git in (defaults to session cwd)
        """
        session = ctx.context.data
        assert session

        # Generate tool call ID
        tool_call_id = f"git-diff-{uuid.uuid4().hex[:8]}"

        try:
            # Check if we have terminal access for running git
            if not (
                session.client_capabilities
                and session.client_capabilities.terminal
                and session.acp_agent.terminal_access
            ):
                await session.notifications.send_agent_text(
                    "❌ **Terminal access not available for git operations**"
                )
                return

            # Build git diff command
            if base_commit:
                git_command = ["git", "diff", base_commit, commit]
                display_title = f"Git diff: {base_commit}..{commit}"
            else:
                git_command = ["git", "diff", f"{commit}~1", commit]
                display_title = f"Git diff: {commit}"

            # Start tool call
            await session.notifications.tool_call_start(
                tool_call_id=tool_call_id,
                title=display_title,
                kind="execute",
            )

            # Create terminal to run git diff
            create_response = await session.requests.create_terminal(
                command=git_command[0],
                args=git_command[1:],
                cwd=cwd or session.cwd,
                env={},
            )
            terminal_id = create_response.terminal_id

            # Wait for git diff to complete
            exit_result = await session.requests.wait_for_terminal_exit(terminal_id)

            # Get the diff output
            output_response = await session.requests.terminal_output(terminal_id)

            # Release terminal
            await session.requests.release_terminal(terminal_id)

            # Check if git command succeeded
            if exit_result.exit_code != 0:
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="failed",
                    title=f"Git diff failed (exit code: {exit_result.exit_code})",
                )
                return

            diff_content = output_response.output.strip()
            if not diff_content:
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="completed",
                    title="No changes found in diff",
                )
                return

            # Stage the diff content for use in agent context
            staged_part = UserPromptPart(
                content=f"Git diff for {display_title}:\n\n{diff_content}"
            )
            session.add_staged_parts([staged_part])

            # Send successful result - wrap in code block for proper display
            staged_count = session.get_staged_parts_count()
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="completed",
                title=f"Git diff fetched and staged ({staged_count} total parts)",
                content=[f"```diff\n{diff_content}\n```"],
            )

        except Exception as e:
            logger.exception("Unexpected error fetching git diff", commit=commit)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="failed",
                title=f"Error: {e}",
            )


class UrlToMarkdownCommand(SlashedCommand):
    """Convert a webpage to markdown using urltomarkdown.herokuapp.com.

    Fetches a web page and converts it to markdown format,
    making it ideal for staging as AI context.
    """

    name = "url-to-markdown"
    category = "docs"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext[ACPSession]],
        url: str,
        *,
        title: bool = True,
        links: bool = True,
        clean: bool = True,
    ):
        """Convert a webpage to markdown.

        Args:
            ctx: Command context with ACP session
            url: URL to convert to markdown
            title: Include page title as H1 header
            links: Include links in markdown output
            clean: Clean/filter content before conversion
        """
        session = ctx.context.data
        assert session

        # Generate tool call ID
        tool_call_id = f"url-to-markdown-{uuid.uuid4().hex[:8]}"

        try:
            # Build API URL and parameters
            api_url = "https://urltomarkdown.herokuapp.com/"
            params = {"url": url}

            if title:
                params["title"] = "true"
            if not links:
                params["links"] = "false"
            if not clean:
                params["clean"] = "false"

            # Start tool call
            await session.notifications.tool_call_start(
                tool_call_id=tool_call_id,
                title=f"Converting to markdown: {url}",
                kind="fetch",
            )

            # Make async HTTP request
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    api_url,
                    params=params,
                    timeout=30.0,
                )
                response.raise_for_status()
                markdown_content = response.text

            # Get title from header if available
            page_title = ""
            if "X-Title" in response.headers:
                import urllib.parse

                page_title = urllib.parse.unquote(response.headers["X-Title"])
                page_title = f" - {page_title}"

            # Stage the markdown content for use in agent context
            staged_part = UserPromptPart(
                content=f"Webpage content from {url}{page_title}:\n\n{markdown_content}"
            )
            session.add_staged_parts([staged_part])

            # Send successful result - wrap in code block for proper display
            staged_count = session.get_staged_parts_count()
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="completed",
                title=f"Webpage converted and staged ({staged_count} total parts)",
                content=[f"```markdown\n{markdown_content}\n```"],
            )

        except httpx.HTTPStatusError as e:
            logger.exception(
                "HTTP error converting URL", url=url, status=e.response.status_code
            )
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="failed",
                title=f"HTTP {e.response.status_code}: Failed to convert {url}",
            )
        except httpx.RequestError as e:
            logger.exception("Request error converting URL", url=url)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="failed",
                title=f"Network error: {e}",
            )
        except Exception as e:
            logger.exception("Unexpected error converting URL", url=url)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="failed",
                title=f"Error: {e}",
            )


class FetchRepoCommand(SlashedCommand):
    """Fetch contents from a GitHub repository via UIThub.

    Retrieves repository contents with various filtering options
    and displays them in a structured format.
    """

    name = "fetch-repo"
    category = "docs"

    async def execute_command(  # noqa: PLR0915
        self,
        ctx: CommandContext[AgentContext[ACPSession]],
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
    ):
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
                title=f"Repository {display_path} fetched and staged "
                f"({staged_count} total parts)",
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


def get_docs_commands() -> list[type[SlashedCommand]]:
    """Get all documentation-related slash commands."""
    return [
        GetSourceCommand,
        GitDiffCommand,
        FetchRepoCommand,
        UrlToMarkdownCommand,
    ]
