"""Documentation and repository fetching slash commands for ACP sessions."""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx
from pydantic_ai import UserPromptPart
from slashed import CommandContext, SlashedCommand  # noqa: TC002

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent_server.acp_server.session import ACPSession  # noqa: TC001


logger = get_logger(__name__)


# Script constants for client-side execution
SCHEMA_EXTRACTION_SCRIPT = """
import sys
import json
import importlib

def import_callable(path):
    \"\"\"Simple import_callable implementation.\"\"\"
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            module_path = ".".join(parts[:i])
            module = importlib.import_module(module_path)
            obj = module
            for part in parts[i:]:
                obj = getattr(obj, part)
            return obj
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"Could not import: {path}")

try:
    model_class = import_callable("{input_path}")
    schema = model_class.model_json_schema()
    print(json.dumps(schema, indent=2))
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
"""


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

            # Run importing.py as script
            output, exit_code = await session.requests.run_command(
                command="uv",
                args=["run", importing_py_path, dot_path],
                cwd=cwd or session.cwd,
            )

            # Check if command succeeded
            if exit_code != 0:
                error_msg = output.strip() or f"Exit code: {exit_code}"
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="failed",
                    title=f"Failed to get source for {dot_path}: {error_msg}",
                )
                return

            source_content = output.strip()
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

            # Run git diff command
            output, exit_code = await session.requests.run_command(
                command=git_command[0],
                args=git_command[1:],
                cwd=cwd or session.cwd,
            )

            # Check if git command succeeded
            if exit_code != 0:
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="failed",
                    title=f"Git diff failed (exit code: {exit_code})",
                )
                return

            diff_content = output.strip()
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


class GetSchemaCommand(SlashedCommand):
    """Get Python code from Pydantic model schema.

    Supports both dot notation paths to BaseModel classes and URLs to OpenAPI schemas.
    Uses datamodel-codegen to generate clean Python code from schemas.
    """

    name = "get-schema"
    category = "docs"

    async def execute_command(
        self,
        ctx: CommandContext[AgentContext[ACPSession]],
        input_path: str,
        *,
        class_name: str | None = None,
        cwd: str | None = None,
    ):
        """Get Python code from Pydantic model schema.

        Args:
            ctx: Command context with ACP session
            input_path: Dot notation path to BaseModel or URL to OpenAPI schema
            class_name: Optional custom class name for generated code
            cwd: Working directory to run in (defaults to session cwd)
        """
        from datamodel_code_generator import DataModelType, LiteralType, PythonVersion
        from datamodel_code_generator.model import get_data_model_types
        from datamodel_code_generator.parser.jsonschema import JsonSchemaParser
        from datamodel_code_generator.parser.openapi import OpenAPIParser

        session = ctx.context.data
        assert session

        # Generate tool call ID
        tool_call_id = f"get-schema-{uuid.uuid4().hex[:8]}"

        try:
            # Check if we have terminal access
            if not (
                session.client_capabilities
                and session.client_capabilities.terminal
                and session.acp_agent.terminal_access
            ):
                await session.notifications.send_agent_text(
                    "❌ **Terminal access not available for schema generation**"
                )
                return

            # Determine if input is URL or dot path
            is_url = input_path.startswith(("http://", "https://"))

            # Start tool call
            display_title = (
                f"Generating schema from URL: {input_path}"
                if is_url
                else f"Generating schema from model: {input_path}"
            )
            await session.notifications.tool_call_start(
                tool_call_id=tool_call_id,
                title=display_title,
                kind="read",
            )

            if is_url:
                # Direct URL approach - use OpenAPIParser directly
                def _generate_from_url():
                    data_model_types = get_data_model_types(
                        DataModelType.PydanticV2BaseModel,
                        target_python_version=PythonVersion.PY_312,
                    )
                    parser = OpenAPIParser(
                        source=input_path,
                        data_model_type=data_model_types.data_model,
                        data_model_root_type=data_model_types.root_model,
                        data_model_field_type=data_model_types.field_model,
                        data_type_manager_type=data_model_types.data_type_manager,
                        dump_resolve_reference_action=data_model_types.dump_resolve_reference_action,
                        class_name=class_name,
                        use_union_operator=True,
                        use_schema_description=True,
                        enum_field_as_literal=LiteralType.All,
                    )
                    return parser.parse()

                try:
                    generated_code = await asyncio.to_thread(_generate_from_url)
                except Exception as e:  # noqa: BLE001
                    await session.notifications.tool_call_progress(
                        tool_call_id=tool_call_id,
                        status="failed",
                        title=f"Failed to generate from URL: {e!s}",
                    )
                    return

                generated_code = generated_code.strip()

            else:
                # Dot path approach - client-side schema extraction
                # Step 1: CLIENT-SIDE - Extract schema via inline script
                schema_script = SCHEMA_EXTRACTION_SCRIPT.format(input_path=input_path)

                # Run schema extraction client-side using uv run python -c
                output, exit_code = await session.requests.run_command(
                    command="uv",
                    args=["run", "python", "-c", schema_script],
                    cwd=cwd or session.cwd,
                )

                if exit_code != 0:
                    error_msg = output.strip() or f"Exit code: {exit_code}"
                    await session.notifications.tool_call_progress(
                        tool_call_id=tool_call_id,
                        status="failed",
                        title=f"Failed to extract schema: {error_msg}",
                    )
                    return

                schema_json = output.strip()
                if not schema_json:
                    await session.notifications.tool_call_progress(
                        tool_call_id=tool_call_id,
                        status="failed",
                        title="No schema extracted from model",
                    )
                    return

                # Step 2: SERVER-SIDE - Generate code from schema using JsonSchemaParser
                def _generate_from_schema():
                    data_model_types = get_data_model_types(
                        DataModelType.PydanticV2BaseModel,
                        target_python_version=PythonVersion.PY_312,
                    )
                    parser = JsonSchemaParser(
                        source=schema_json,
                        data_model_type=data_model_types.data_model,
                        data_model_root_type=data_model_types.root_model,
                        data_model_field_type=data_model_types.field_model,
                        data_type_manager_type=data_model_types.data_type_manager,
                        dump_resolve_reference_action=data_model_types.dump_resolve_reference_action,
                        class_name=class_name or input_path.split(".")[-1],
                        use_union_operator=True,
                        use_schema_description=True,
                        enum_field_as_literal=LiteralType.All,
                    )
                    return parser.parse()

                try:
                    generated_code = await asyncio.to_thread(_generate_from_schema)
                except Exception as e:  # noqa: BLE001
                    await session.notifications.tool_call_progress(
                        tool_call_id=tool_call_id,
                        status="failed",
                        title=f"Failed to generate code: {e!s}",
                    )
                    return

                generated_code = generated_code.strip()

            if not generated_code:
                await session.notifications.tool_call_progress(
                    tool_call_id=tool_call_id,
                    status="completed",
                    title="No code generated from schema",
                )
                return

            # Stage the generated code for use in agent context
            staged_part = UserPromptPart(
                content=f"Generated Python code from {input_path}:\n\n{generated_code}"
            )
            session.add_staged_parts([staged_part])

            # Send successful result - wrap in code block for proper display
            staged_count = session.get_staged_parts_count()
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="completed",
                title=f"Schema code generated and staged ({staged_count} total parts)",
                content=[f"```python\n{generated_code}\n```"],
            )

        except Exception as e:
            logger.exception(
                "Unexpected error generating schema code", input_path=input_path
            )
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
        GetSchemaCommand,
        GitDiffCommand,
        FetchRepoCommand,
        UrlToMarkdownCommand,
    ]


if __name__ == "__main__":
    from slashed import CommandStore

    cmd = GetSchemaCommand()
    store = CommandStore()
    store.register_command(GetSchemaCommand())

    async def main():
        await store.execute_command_with_context("get-schema")
