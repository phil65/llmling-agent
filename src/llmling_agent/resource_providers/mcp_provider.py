"""Tool management for LLMling agents."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Self

from pydantic_ai import UsageLimits

from llmling_agent.log import get_logger
from llmling_agent.mcp_server import MCPClient
from llmling_agent.models.content import AudioBase64Content, ImageBase64Content
from llmling_agent.prompts.prompts import Prompt
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent_config.mcp_server import BaseMCPServerConfig
from llmling_agent_config.resources import ResourceInfo


if TYPE_CHECKING:
    from types import TracebackType

    from fastmcp.client.elicitation import ElicitResult
    from mcp import types
    from mcp.client.session import RequestContext
    from mcp.types import SamplingMessage

    from llmling_agent.mcp_server.client import ContextualProgressHandler
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.models.content import BaseContent
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


class MCPResourceProvider(ResourceProvider):
    """Manages MCP server connections and tools."""

    def __init__(
        self,
        server: MCPServerConfig | str,
        name: str = "mcp",
        owner: str | None = None,
        context: NodeContext | None = None,
        progress_handler: ContextualProgressHandler | None = None,
        accessible_roots: list[str] | None = None,
    ):
        super().__init__(name, owner=owner)
        self.server = (
            BaseMCPServerConfig.from_string(server) if isinstance(server, str) else server
        )
        self.context = context
        self.exit_stack = AsyncExitStack()
        self._progress_handler = progress_handler
        self._accessible_roots = accessible_roots
        self.client = MCPClient(
            config=self.server,
            elicitation_callback=self._elicitation_callback,
            sampling_callback=self._sampling_callback,
            progress_handler=self._progress_handler,
            accessible_roots=self._accessible_roots,
        )

    def __repr__(self) -> str:
        return f"MCPResourceProvider({self.server!r})"

    async def __aenter__(self) -> Self:
        try:
            await self.exit_stack.enter_async_context(self.client)
        except Exception as e:
            # Clean up in case of error
            await self.__aexit__(type(e), e, e.__traceback__)
            msg = "Failed to initialize MCP manager"
            raise RuntimeError(msg) from e

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        try:
            try:
                # Clean up exit stack (which includes MCP clients)
                await self.exit_stack.aclose()
            except RuntimeError as e:
                if "different task" in str(e):
                    # Handle task context mismatch
                    current_task = asyncio.current_task()
                    if current_task:
                        loop = asyncio.get_running_loop()
                        await loop.create_task(self.exit_stack.aclose())
                else:
                    raise

        except Exception as e:
            msg = "Error during MCP manager cleanup"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e

    async def _elicitation_callback(
        self,
        message: str,
        response_type: type[Any],
        params: types.ElicitRequestParams,
        context: RequestContext,
    ) -> ElicitResult[dict[str, Any]] | dict[str, Any] | None:
        """Handle elicitation requests from MCP server."""
        from fastmcp.client.elicitation import ElicitResult
        from mcp import types

        from llmling_agent.agent.context import AgentContext

        if self.context and isinstance(self.context, AgentContext):
            legacy_result = await self.context.handle_elicitation(params)
            # Convert legacy MCP result to FastMCP format
            if isinstance(legacy_result, types.ElicitResult):
                if legacy_result.action == "accept" and legacy_result.content:
                    return legacy_result.content
                return ElicitResult(action=legacy_result.action)
            if isinstance(legacy_result, types.ErrorData):
                return ElicitResult(action="cancel")
            return ElicitResult(action="decline")

        return ElicitResult(action="decline")

    async def _sampling_callback(
        self,
        messages: list[SamplingMessage],
        params: types.CreateMessageRequestParams,
        context: RequestContext,
    ) -> str:
        """Handle MCP sampling by creating a new agent with specified preferences."""
        from mcp import types

        from llmling_agent.agent import Agent

        try:
            # Convert messages to prompts for the agent
            prompts: list[BaseContent | str] = []
            for mcp_msg in messages:
                match mcp_msg.content:
                    case types.TextContent(text=text):
                        prompts.append(text)
                    case types.ImageContent(data=data, mimeType=mime_type):
                        our_image = ImageBase64Content(data=data, mime_type=mime_type)
                        prompts.append(our_image)
                    case types.AudioContent(data=data, mimeType=mime_type):
                        fmt = mime_type.removeprefix("audio/")
                        our_audio = AudioBase64Content(data=data, format=fmt)
                        prompts.append(our_audio)

            # Extract model from preferences
            model = None
            if (
                params.modelPreferences
                and params.modelPreferences.hints
                and params.modelPreferences.hints[0].name
            ):
                model = params.modelPreferences.hints[0].name

            # Create usage limits from sampling parameters
            usage_limits = UsageLimits(
                output_tokens_limit=params.maxTokens,
                request_limit=1,  # Single sampling request
            )

            # TODO: Apply temperature from params.temperature
            # Currently no direct way to pass temperature to Agent constructor
            # May need provider-level configuration or runtime model settings

            # Create agent with sampling parameters
            agent = Agent(
                name="mcp-sampling-agent",
                model=model,
                system_prompt=params.systemPrompt or "",
                session=False,  # Don't store history for sampling
            )

            async with agent:
                # Pass all prompts directly to the agent
                result = await agent.run(
                    *prompts,
                    store_history=False,
                    usage_limits=usage_limits,
                )

                return str(result.content)

        except Exception as e:
            logger.exception("Sampling failed")
            return f"Sampling failed: {e!s}"

    async def setup_server(self, config: MCPServerConfig) -> None:
        """Set up a single MCP server connection."""

    async def get_tools(self) -> list[Tool]:
        """Get all tools from all connected servers."""
        all_tools: list[Tool] = []

        for tool in self.client._available_tools:
            try:
                tool_info = self.client.convert_tool(tool)
                all_tools.append(tool_info)
            except Exception:
                logger.exception("Failed to create MCP tool", name=tool.name)
                continue
        logger.debug("Fetched MCP tools", num_tools=len(all_tools))
        return all_tools

    async def list_prompts(self) -> list[Prompt]:
        """Get all available prompts from MCP servers."""
        try:
            result = await self.client.list_prompts()
            client_prompts: list[Prompt] = []
            for prompt in result:
                try:
                    converted = Prompt.from_fastmcp(self.client, prompt)
                    client_prompts.append(converted)
                except Exception:
                    logger.exception("Failed to convert prompt", name=prompt.name)
        except Exception:
            logger.exception("Failed to get prompts from MCP server")
            return []
        else:
            return client_prompts

    async def list_resources(self) -> list[ResourceInfo]:
        """Get all available resources from MCP servers."""
        try:
            result = await self.client.list_resources()
            client_resources: list[ResourceInfo] = []
            for resource in result:
                try:
                    converted = await ResourceInfo.from_mcp_resource(resource)
                    client_resources.append(converted)
                except Exception:
                    logger.exception("Failed to convert resource", name=resource.name)
        except Exception:
            logger.exception("Failed to get resources from MCP server")
            return []
        else:
            return client_resources


if __name__ == "__main__":
    from llmling_agent_config.mcp_server import StdioMCPServerConfig

    cfg = StdioMCPServerConfig(
        command="uv",
        args=["run", "/home/phil65/dev/oss/llmling-agent/tests/mcp/server.py"],
    )

    async def main():
        manager = MCPResourceProvider(cfg)
        async with manager:
            prompts = await manager.list_prompts()
            print(f"Found prompts: {prompts}")

            # Test static prompt (no arguments)
            static_prompt = next(p for p in prompts if p.name == "static_prompt")
            print(f"\n--- Testing static prompt: {static_prompt} ---")
            components = await static_prompt.get_components()
            assert components, "No prompt components found"
            print(f"Found {len(components)} prompt components:")
            for i, component in enumerate(components):
                comp_type = type(component).__name__
                print(f"  {i + 1}. {comp_type}: {component.content}")

            # Test dynamic prompt (with arguments)
            dynamic_prompt = next(p for p in prompts if p.name == "dynamic_prompt")
            print(f"\n--- Testing dynamic prompt: {dynamic_prompt} ---")
            components = await dynamic_prompt.get_components(
                arguments={"some_arg": "Hello, world!"}
            )
            assert components, "No prompt components found"
            print(f"Found {len(components)} prompt components:")
            for i, component in enumerate(components):
                comp_type = type(component).__name__
                print(f"  {i + 1}. {comp_type}: {component.content}")

    asyncio.run(main())
