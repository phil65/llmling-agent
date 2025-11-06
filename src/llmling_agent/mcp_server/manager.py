"""Tool management for LLMling agents."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Self

from pydantic_ai import SystemPromptPart, UsageLimits, UserPromptPart

from llmling_agent.log import get_logger
from llmling_agent.mcp_server import MCPClient
from llmling_agent.models.content import AudioBase64Content, ImageBase64Content
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent_config.mcp_server import BaseMCPServerConfig
from llmling_agent_config.resources import ResourceInfo


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from fastmcp.client.elicitation import ElicitResult
    from mcp import types
    from mcp.client.session import RequestContext
    from mcp.types import Prompt as MCPPrompt, Resource as MCPResource, SamplingMessage

    from llmling_agent.mcp_server.client import ContextualProgressHandler
    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.models.content import BaseContent
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.mcp_server import MCPServerConfig


class Prompt:
    """A prompt that can be rendered from an MCP server."""

    def __init__(
        self,
        name: str,
        description: str | None,
        arguments: list[dict[str, Any]] | None,
        client: MCPClient,
    ):
        self.name = name
        self.description = description
        self.arguments = arguments or []
        self._client = client

    def __repr__(self) -> str:
        return f"Prompt(name={self.name!r}, description={self.description!r})"

    async def get_components(
        self, arguments: dict[str, str] | None = None
    ) -> list[SystemPromptPart | UserPromptPart]:
        """Get prompt as pydantic-ai message components.

        Args:
            arguments: Arguments to pass to the prompt template

        Returns:
            List of message parts ready for agent usage

        Raises:
            RuntimeError: If prompt fetch fails
            ValueError: If prompt contains unsupported message types
        """
        try:
            result = await self._client.get_prompt(self.name, arguments)
        except Exception as e:
            msg = f"Failed to get prompt {self.name!r}: {e}"
            raise RuntimeError(msg) from e

        # Convert MCP messages to pydantic-ai parts
        from mcp.types import (
            AudioContent,
            EmbeddedResource,
            ImageContent,
            ResourceLink,
            TextContent,
            TextResourceContents,
        )

        parts: list[SystemPromptPart | UserPromptPart] = []

        for message in result.messages:
            # Extract text content from MCP message
            text_content = ""

            match message.content:
                case TextContent(text=text):
                    text_content = text
                case EmbeddedResource(resource=resource):
                    if isinstance(resource, TextResourceContents):
                        text_content = resource.text
                    else:
                        text_content = f"[Resource: {resource.uri}]"
                case ResourceLink(uri=uri, description=desc):
                    text_content = f"[Resource Link: {uri}]"
                    if desc:
                        text_content += f" - {desc}"
                case ImageContent(mimeType=mime_type):
                    text_content = f"[Image: {mime_type}]"
                case AudioContent(mimeType=mime_type):
                    text_content = f"[Audio: {mime_type}]"
                case _:
                    # Fallback to string representation
                    text_content = str(message.content)

            # Convert based on role
            match message.role:
                case "system":
                    parts.append(SystemPromptPart(content=text_content))
                case "user":
                    parts.append(UserPromptPart(content=text_content))
                case "assistant":
                    # Convert assistant messages to user parts for context
                    parts.append(UserPromptPart(content=f"Assistant: {text_content}"))
                case _:
                    logger.warning(
                        "Unsupported message role in MCP prompt",
                        role=message.role,
                        prompt_name=self.name,
                    )

        if not parts:
            msg = f"No supported message parts found in prompt {self.name!r}"
            raise ValueError(msg)

        return parts


logger = get_logger(__name__)


def convert_mcp_prompt(client: MCPClient, prompt: MCPPrompt) -> Prompt:
    """Convert MCP prompt to our Prompt class."""
    arguments = [
        {
            "name": arg.name,
            "description": arg.description,
            "required": arg.required or False,
        }
        for arg in prompt.arguments or []
    ]

    return Prompt(
        name=prompt.name,
        description=prompt.description,
        arguments=arguments,
        client=client,
    )


async def convert_mcp_resource(resource: MCPResource) -> ResourceInfo:
    """Convert MCP resource to ResourceInfo."""
    return ResourceInfo(
        name=resource.name, uri=str(resource.uri), description=resource.description
    )


class MCPManager(ResourceProvider):
    """Manages MCP server connections and tools."""

    def __init__(
        self,
        name: str = "mcp",
        owner: str | None = None,
        servers: Sequence[MCPServerConfig | str] | None = None,
        context: NodeContext | None = None,
        progress_handler: ContextualProgressHandler | None = None,
        accessible_roots: list[str] | None = None,
    ):
        super().__init__(name, owner=owner)
        self.servers: list[MCPServerConfig] = []
        for server in servers or []:
            self.add_server_config(server)
        self.context = context
        self.clients: dict[str, MCPClient] = {}
        self.exit_stack = AsyncExitStack()
        self._progress_handler = progress_handler
        self._accessible_roots = accessible_roots

    def add_server_config(self, cfg: MCPServerConfig | str):
        """Add a new MCP server to the manager."""
        resolved = BaseMCPServerConfig.from_string(cfg) if isinstance(cfg, str) else cfg
        self.servers.append(resolved)

    def __repr__(self) -> str:
        return f"MCPManager({self.servers!r})"

    async def __aenter__(self) -> Self:
        try:
            # Setup directly provided servers and context servers concurrently
            tasks = [self.setup_server(server) for server in self.servers]
            if self.context and self.context.config and self.context.config.mcp_servers:
                tasks.extend(
                    self.setup_server(server)
                    for server in self.context.config.get_mcp_servers()
                )

            if tasks:
                await asyncio.gather(*tasks)

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
        await self.cleanup()

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
        if not config.enabled:
            return

        client = MCPClient(
            config=config,
            elicitation_callback=self._elicitation_callback,
            sampling_callback=self._sampling_callback,
            progress_handler=self._progress_handler,
            accessible_roots=self._accessible_roots,
        )
        client = await self.exit_stack.enter_async_context(client)

        self.clients[config.client_id] = client

    async def get_tools(self) -> list[Tool]:
        """Get all tools from all connected servers."""
        all_tools: list[Tool] = []

        for client in self.clients.values():
            for tool in client._available_tools:
                try:
                    tool_info = client.convert_tool(tool)
                    all_tools.append(tool_info)
                except Exception:
                    logger.exception("Failed to create MCP tool", name=tool.name)
                    continue
        logger.debug("Fetched MCP tools", num_tools=len(all_tools))
        return all_tools

    async def list_prompts(self) -> list[Prompt]:
        """Get all available prompts from MCP servers."""

        async def get_client_prompts(client: MCPClient) -> list[Prompt]:
            try:
                result = await client.list_prompts()
                client_prompts: list[Prompt] = []
                for prompt in result:
                    try:
                        converted = convert_mcp_prompt(client, prompt)
                        client_prompts.append(converted)
                    except Exception:
                        logger.exception("Failed to convert prompt", name=prompt.name)
            except Exception:
                logger.exception("Failed to get prompts from MCP server")
                return []
            else:
                return client_prompts

        results = await asyncio.gather(*[
            get_client_prompts(client) for client in self.clients.values()
        ])
        return [prompt for client_prompts in results for prompt in client_prompts]

    async def list_resources(self) -> list[ResourceInfo]:
        """Get all available resources from MCP servers."""

        async def get_client_resources(client: MCPClient) -> list[ResourceInfo]:
            try:
                result = await client.list_resources()
                client_resources: list[ResourceInfo] = []
                for resource in result:
                    try:
                        converted = await convert_mcp_resource(resource)
                        client_resources.append(converted)
                    except Exception:
                        logger.exception("Failed to convert resource", name=resource.name)
            except Exception:
                logger.exception("Failed to get resources from MCP server")
                return []
            else:
                return client_resources

        results = await asyncio.gather(
            *[get_client_resources(client) for client in self.clients.values()],
            return_exceptions=False,
        )
        return [resource for client_resources in results for resource in client_resources]

    async def cleanup(self) -> None:
        """Clean up all MCP connections."""
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

            self.clients.clear()

        except Exception as e:
            msg = "Error during MCP manager cleanup"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e

    @property
    def active_servers(self) -> list[str]:
        """Get IDs of active servers."""
        return list(self.clients)


if __name__ == "__main__":
    from llmling_agent_config.mcp_server import StdioMCPServerConfig

    cfg = StdioMCPServerConfig(
        command="uv",
        args=["run", "/home/phil65/dev/oss/llmling-agent/tests/mcp/server.py"],
    )

    async def main():
        manager = MCPManager()
        async with manager:
            await manager.setup_server(cfg)
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
