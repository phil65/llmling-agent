"""Models for PydanticAI builtin tools configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from llmling_agent_config.tools import BaseToolConfig


if TYPE_CHECKING:
    from pydantic_ai.builtin_tools import (
        AbstractBuiltinTool,
        CodeExecutionTool,
        ImageGenerationTool,
        MCPServerTool,
        MemoryTool,
        UrlContextTool,
        WebSearchTool,
    )


class BaseBuiltinToolConfig(BaseToolConfig):
    """Base configuration for PydanticAI builtin tools."""

    def get_builtin_tool(self) -> AbstractBuiltinTool:
        """Convert config to PydanticAI builtin tool instance."""
        raise NotImplementedError


class WebSearchToolConfig(BaseBuiltinToolConfig):
    """Configuration for PydanticAI web search builtin tool."""

    type: Literal["web_search"] = Field("web_search", init=False)
    """Web search builtin tool."""

    search_context_size: Literal["low", "medium", "high"] = "medium"
    """The search context size parameter controls how much context is retrieved."""

    user_location: dict[str, str] | None = None
    """User location for localizing search results (city, country, region, timezone)."""

    blocked_domains: list[str] | None = None
    """Domains that will never appear in results."""

    allowed_domains: list[str] | None = None
    """Only these domains will be included in results."""

    max_uses: int | None = None
    """Maximum number of times the tool can be used."""

    def get_builtin_tool(self) -> WebSearchTool:
        """Convert config to WebSearchTool instance."""
        from pydantic_ai.builtin_tools import WebSearchTool

        return WebSearchTool(
            search_context_size=self.search_context_size,
            user_location=self.user_location,
            blocked_domains=self.blocked_domains,
            allowed_domains=self.allowed_domains,
            max_uses=self.max_uses,
        )


class CodeExecutionToolConfig(BaseBuiltinToolConfig):
    """Configuration for PydanticAI code execution builtin tool."""

    type: Literal["code_execution"] = Field("code_execution", init=False)
    """Code execution builtin tool."""

    def get_builtin_tool(self) -> CodeExecutionTool:
        """Convert config to CodeExecutionTool instance."""
        from pydantic_ai.builtin_tools import CodeExecutionTool

        return CodeExecutionTool()


class UrlContextToolConfig(BaseBuiltinToolConfig):
    """Configuration for PydanticAI URL context builtin tool."""

    type: Literal["url_context"] = Field("url_context", init=False)
    """URL context builtin tool."""

    def get_builtin_tool(self) -> UrlContextTool:
        """Convert config to UrlContextTool instance."""
        from pydantic_ai.builtin_tools import UrlContextTool

        return UrlContextTool()


class ImageGenerationToolConfig(BaseBuiltinToolConfig):
    """Configuration for PydanticAI image generation builtin tool."""

    type: Literal["image_generation"] = Field("image_generation", init=False)
    """Image generation builtin tool."""

    background: Literal["transparent", "opaque", "auto"] = "auto"
    """Background type for the generated image."""

    input_fidelity: Literal["high", "low"] | None = None
    """Control how much effort the model will exert to match input image features."""

    moderation: Literal["auto", "low"] = "auto"
    """Moderation level for the generated image."""

    output_compression: int = 100
    """Compression level for the output image."""

    output_format: Literal["png", "webp", "jpeg"] | None = None
    """The output format of the generated image."""

    partial_images: int = 0
    """Number of partial images to generate in streaming mode."""

    quality: Literal["low", "medium", "high", "auto"] = "auto"
    """The quality of the generated image."""

    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] = "auto"
    """The size of the generated image."""

    def get_builtin_tool(self) -> ImageGenerationTool:
        """Convert config to ImageGenerationTool instance."""
        from pydantic_ai.builtin_tools import ImageGenerationTool

        return ImageGenerationTool(
            background=self.background,
            input_fidelity=self.input_fidelity,
            moderation=self.moderation,
            output_compression=self.output_compression,
            output_format=self.output_format,
            partial_images=self.partial_images,
            quality=self.quality,
            size=self.size,
        )


class MemoryToolConfig(BaseBuiltinToolConfig):
    """Configuration for PydanticAI memory builtin tool."""

    type: Literal["memory"] = Field("memory", init=False)
    """Memory builtin tool."""

    def get_builtin_tool(self) -> MemoryTool:
        """Convert config to MemoryTool instance."""
        from pydantic_ai.builtin_tools import MemoryTool

        return MemoryTool()


class MCPServerToolConfig(BaseBuiltinToolConfig):
    """Configuration for PydanticAI MCP server builtin tool."""

    type: Literal["mcp_server"] = Field("mcp_server", init=False)
    """MCP server builtin tool."""

    server_id: str = Field(alias="id")
    """A unique identifier for the MCP server."""

    url: str
    """The URL of the MCP server to use."""

    authorization_token: str | None = None
    """Authorization header to use when making requests to the MCP server."""

    description: str | None = None
    """A description of the MCP server."""

    allowed_tools: list[str] | None = None
    """A list of tools that the MCP server can use."""

    headers: dict[str, str] | None = None
    """Optional HTTP headers to send to the MCP server."""

    def get_builtin_tool(self) -> MCPServerTool:
        """Convert config to MCPServerTool instance."""
        from pydantic_ai.builtin_tools import MCPServerTool

        return MCPServerTool(
            id=self.server_id,
            url=self.url,
            authorization_token=self.authorization_token,
            description=self.description,
            allowed_tools=self.allowed_tools,
            headers=self.headers,
        )


# Union type for builtin tool configs
BuiltinToolConfig = Annotated[
    WebSearchToolConfig
    | CodeExecutionToolConfig
    | UrlContextToolConfig
    | ImageGenerationToolConfig
    | MemoryToolConfig
    | MCPServerToolConfig,
    Field(discriminator="type"),
]
