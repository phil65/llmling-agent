"""Convert a webpage to markdown using urltomarkdown.herokuapp.com."""

from __future__ import annotations

import urllib.parse
import uuid

import httpx
from pydantic_ai import UserPromptPart
from slashed import CommandContext, SlashedCommand  # noqa: TC002

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent_server.acp_server.session import ACPSession  # noqa: TC001


logger = get_logger(__name__)


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
    ) -> None:
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
