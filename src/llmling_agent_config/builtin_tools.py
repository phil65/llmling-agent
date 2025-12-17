"""Models for PydanticAI builtin tools configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field
from pydantic_ai import (
    CodeExecutionTool,
    ImageGenerationTool,
    MCPServerTool,
    WebFetchTool,
    WebSearchTool,
)
from pydantic_ai.builtin_tools import WebSearchUserLocation  # noqa: TC002

from llmling_agent_config.tools import BaseToolConfig


if TYPE_CHECKING:
    from pydantic_ai.builtin_tools import AbstractBuiltinTool, MemoryTool


class BaseBuiltinToolConfig(BaseToolConfig):
    """Base configuration for PydanticAI builtin tools."""

    def get_builtin_tool(self) -> AbstractBuiltinTool:
        """Convert config to PydanticAI builtin tool instance."""
        raise NotImplementedError


class WebSearchToolConfig(BaseBuiltinToolConfig):
    """Configuration for PydanticAI web search builtin tool."""

    model_config = ConfigDict(title="Web Search Tool")

    type: Literal["web_search"] = Field("web_search", init=False)
    """Web search builtin tool."""

    search_context_size: Literal["low", "medium", "high"] = Field(
        default="medium",
        examples=["low", "medium", "high"],
        title="Search context size",
    )
    """The search context size parameter controls how much context is retrieved."""

    user_location: WebSearchUserLocation | None = Field(default=None, title="User location")
    """User location for localizing search results (city, country, region, timezone)."""

    blocked_domains: list[str] | None = Field(
        default=None,
        examples=[["spam.com", "ads.example.com"], ["blocked.site"]],
        title="Blocked domains",
    )
    """Domains that will never appear in results."""

    allowed_domains: list[str] | None = Field(
        default=None,
        examples=[["wikipedia.org", "github.com"], ["trusted.site"]],
        title="Allowed domains",
    )
    """Only these domains will be included in results."""

    max_uses: int | None = Field(default=None, examples=[5, 10, 20], title="Maximum uses")
    """Maximum number of times the tool can be used."""

    def get_builtin_tool(self) -> WebSearchTool:
        """Convert config to WebSearchTool instance."""
        return WebSearchTool(
            search_context_size=self.search_context_size,
            user_location=self.user_location,
            blocked_domains=self.blocked_domains,
            allowed_domains=self.allowed_domains,
            max_uses=self.max_uses,
        )


class CodeExecutionToolConfig(BaseBuiltinToolConfig):
    """Configuration for PydanticAI code execution builtin tool."""

    model_config = ConfigDict(title="Code Execution Tool")

    type: Literal["code_execution"] = Field("code_execution", init=False)
    """Code execution builtin tool."""

    def get_builtin_tool(self) -> CodeExecutionTool:
        """Convert config to CodeExecutionTool instance."""
        return CodeExecutionTool()


class WebFetchToolConfig(BaseBuiltinToolConfig):
    """Configuration for PydanticAI URL context builtin tool."""

    model_config = ConfigDict(title="Url Context Tool")

    type: Literal["web_fetch"] = Field("web_fetch", init=False)
    """URL context builtin tool."""

    def get_builtin_tool(self) -> WebFetchTool:
        """Convert config to WebFetchTool instance."""
        return WebFetchTool()


class ImageGenerationToolConfig(BaseBuiltinToolConfig):
    """Configuration for PydanticAI image generation builtin tool."""

    model_config = ConfigDict(title="Image Generation Tool")

    type: Literal["image_generation"] = Field("image_generation", init=False)
    """Image generation builtin tool."""

    background: Literal["transparent", "opaque", "auto"] = Field(
        default="auto",
        examples=["transparent", "opaque", "auto"],
        title="Background type",
    )
    """Background type for the generated image."""

    input_fidelity: Literal["high", "low"] | None = Field(
        default=None,
        examples=["high", "low"],
        title="Input fidelity",
    )
    """Control how much effort the model will exert to match input image features."""

    moderation: Literal["auto", "low"] = Field(
        default="auto",
        examples=["auto", "low"],
        title="Moderation level",
    )
    """Moderation level for the generated image."""

    output_compression: int = Field(
        default=100,
        ge=0,
        le=100,
        examples=[80, 90, 100],
        title="Output compression",
    )
    """Compression level for the output image."""

    output_format: Literal["png", "webp", "jpeg"] | None = Field(
        default=None,
        examples=["png", "webp", "jpeg"],
        title="Output format",
    )
    """The output format of the generated image."""

    partial_images: int = Field(
        default=0,
        ge=0,
        examples=[0, 2, 4],
        title="Partial images count",
    )
    """Number of partial images to generate in streaming mode."""

    quality: Literal["low", "medium", "high", "auto"] = Field(
        default="auto",
        examples=["low", "medium", "high", "auto"],
        title="Image quality",
    )
    """The quality of the generated image."""

    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] = Field(
        default="auto",
        examples=["1024x1024", "1024x1536", "1536x1024", "auto"],
        title="Image size",
    )
    """The size of the generated image."""

    def get_builtin_tool(self) -> ImageGenerationTool:
        """Convert config to ImageGenerationTool instance."""
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

    model_config = ConfigDict(title="Memory Tool")

    type: Literal["memory"] = Field("memory", init=False)
    """Memory builtin tool."""

    def get_builtin_tool(self) -> MemoryTool:
        """Convert config to MemoryTool instance."""
        from pydantic_ai import MemoryTool

        return MemoryTool()


class MCPServerToolConfig(BaseBuiltinToolConfig):
    """Configuration for PydanticAI MCP server builtin tool."""

    model_config = ConfigDict(title="MCP Server Tool")

    type: Literal["mcp_server"] = Field("mcp_server", init=False)
    """MCP server builtin tool."""

    server_id: str = Field(
        alias="id",
        examples=["my_mcp_server", "code_tools", "web_api"],
        title="Server ID",
    )
    """A unique identifier for the MCP server."""

    url: str = Field(
        examples=["https://api.example.com/mcp", "http://localhost:8080"],
        title="Server URL",
    )
    """The URL of the MCP server to use."""

    authorization_token: str | None = Field(
        default=None,
        examples=["Bearer token123", "api_key_abc"],
        title="Authorization token",
    )
    """Authorization header to use when making requests to the MCP server."""

    description: str | None = Field(
        default=None,
        examples=["External API tools", "Code execution server"],
        title="Server description",
    )
    """A description of the MCP server."""

    allowed_tools: list[str] | None = Field(
        default=None,
        examples=[["search", "fetch"], ["execute", "compile"]],
        title="Allowed tools",
    )
    """A list of tools that the MCP server can use."""

    headers: dict[str, str] | None = Field(default=None, title="HTTP headers")
    """Optional HTTP headers to send to the MCP server."""

    def get_builtin_tool(self) -> MCPServerTool:
        """Convert config to MCPServerTool instance."""
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
    | WebFetchToolConfig
    | ImageGenerationToolConfig
    | MemoryToolConfig
    | MCPServerToolConfig,
    Field(discriminator="type"),
]
