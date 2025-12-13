"""Get Python code from Pydantic model schema."""

from __future__ import annotations

import asyncio
import uuid

from pydantic_ai import UserPromptPart
from schemez import jsonschema_to_code, openapi_to_code
from slashed import CommandContext  # noqa: TC002

from llmling_agent.log import get_logger
from llmling_agent.messaging.context import NodeContext  # noqa: TC001
from llmling_agent_commands.base import NodeCommand
from llmling_agent_server.acp_server.session import ACPSession  # noqa: TC001

from .helpers import SCHEMA_EXTRACTION_SCRIPT


logger = get_logger(__name__)


class GetSchemaCommand(NodeCommand):
    """Get Python code from Pydantic model schema.

    Supports both dot notation paths to BaseModel classes and URLs to OpenAPI schemas.
    Uses datamodel-codegen to generate clean Python code from schemas.
    """

    name = "get-schema"
    category = "docs"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext[ACPSession]],
        input_path: str,
        *,
        class_name: str | None = None,
        cwd: str | None = None,
    ) -> None:
        """Get Python code from Pydantic model schema.

        Args:
            ctx: Command context with ACP session
            input_path: Dot notation path to BaseModel or URL to OpenAPI schema
            class_name: Optional custom class name for generated code
            cwd: Working directory to run in (defaults to session cwd)
        """
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
                try:
                    generated_code = await asyncio.to_thread(
                        openapi_to_code, input_path=input_path, class_name=class_name
                    )
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

                # Step 2: SERVER-SIDE - Generate code from schema

                try:
                    generated_code = await asyncio.to_thread(
                        jsonschema_to_code,
                        schema_json=schema_json,
                        class_name=class_name,
                        input_path=input_path,
                    )
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
            logger.exception("Unexpected error generating schema code", input_path=input_path)
            await session.notifications.tool_call_progress(
                tool_call_id=tool_call_id,
                status="failed",
                title=f"Error: {e}",
            )
