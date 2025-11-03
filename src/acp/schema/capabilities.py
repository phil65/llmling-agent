"""Capability schema."""

from __future__ import annotations

from pydantic import Field

from acp.schema.base import AnnotatedObject


class FileSystemCapability(AnnotatedObject):
    """File system capabilities that a client may support.

    See protocol docs: [FileSystem](https://agentclientprotocol.com/protocol/initialization#filesystem)
    """

    read_text_file: bool | None = False
    """Whether the Client supports `fs/read_text_file` requests."""

    write_text_file: bool | None = False
    """Whether the Client supports `fs/write_text_file` requests."""


class ClientCapabilities(AnnotatedObject):
    """Capabilities supported by the client.

    Advertised during initialization to inform the agent about
    available features and methods.

    See protocol docs: [Client Capabilities](https://agentclientprotocol.com/protocol/initialization#client-capabilities)
    """

    fs: FileSystemCapability | None = Field(default_factory=FileSystemCapability)
    """File system capabilities supported by the client.

    Determines which file operations the agent can request.
    """

    terminal: bool | None = False
    """Whether the Client support all `terminal/*` methods."""

    @classmethod
    def create(
        cls,
        read_text_file: bool | None = False,
        write_text_file: bool | None = False,
        terminal: bool | None = False,
    ):
        """Create a new instance of ClientCapabilities.

        Args:
            read_text_file: Whether the Client supports `fs/read_text_file` requests.
            write_text_file: Whether the Client supports `fs/write_text_file` requests.
            terminal: Whether the Client supports all `terminal/*` methods.

        Returns:
            A new instance of ClientCapabilities.
        """
        return cls(
            fs=FileSystemCapability(
                read_text_file=read_text_file, write_text_file=write_text_file
            ),
            terminal=terminal,
        )


class PromptCapabilities(AnnotatedObject):
    """Prompt capabilities supported by the agent in `session/prompt` requests.

    Baseline agent functionality requires support for [`ContentBlock::Text`]
    and [`ContentBlock::ResourceContentBlock`] in prompt requests.

    Other variants must be explicitly opted in to.
    Capabilities for different types of content in prompt requests.

    Indicates which content types beyond the baseline (text and resource links)
    the agent can process.

    See protocol docs: [Prompt Capabilities](https://agentclientprotocol.com/protocol/initialization#prompt-capabilities)
    """

    audio: bool | None = False
    """Agent supports [`ContentBlock::Audio`]."""

    embedded_context: bool | None = False
    """Agent supports embedded context in `session/prompt` requests.

    When enabled, the Client is allowed to include [`ContentBlock::Resource`]
    in prompt requests for pieces of context that are referenced in the message.
    """

    image: bool | None = False
    """Agent supports [`ContentBlock::Image`]."""


class McpCapabilities(AnnotatedObject):
    """MCP capabilities supported by the agent."""

    http: bool | None = False
    """Agent supports [`McpServer::Http`]."""

    sse: bool | None = False
    """Agent supports [`McpServer::Sse`]."""


class AgentCapabilities(AnnotatedObject):
    """Capabilities supported by the agent.

    Advertised during initialization to inform the client about
    available features and content types.

    See protocol docs: [Agent Capabilities](https://agentclientprotocol.com/protocol/initialization#agent-capabilities)
    """

    load_session: bool | None = False
    """Whether the agent supports `session/load`."""

    mcp_capabilities: McpCapabilities | None = Field(default_factory=McpCapabilities)
    """MCP capabilities supported by the agent."""

    prompt_capabilities: PromptCapabilities | None = Field(
        default_factory=PromptCapabilities
    )
    """Prompt capabilities supported by the agent."""

    @classmethod
    def create(
        cls,
        load_session: bool | None = False,
        http_mcp_servers: bool = False,
        sse_mcp_servers: bool = False,
        audio_prompts: bool = False,
        embedded_context_prompts: bool = False,
        image_prompts: bool = False,
    ):
        """Create an instance of AgentCapabilities.

        Args:
            load_session: Whether the agent supports `session/load`.
            http_mcp_servers: Whether the agent supports HTTP MCP servers.
            sse_mcp_servers: Whether the agent supports SSE MCP servers.
            audio_prompts: Whether the agent supports audio prompts.
            embedded_context_prompts: Whether the agent supports embedded context prompts.
            image_prompts: Whether the agent supports image prompts.
        """
        return cls(
            load_session=load_session,
            mcp_capabilities=McpCapabilities(
                http=http_mcp_servers,
                sse=sse_mcp_servers,
            ),
            prompt_capabilities=PromptCapabilities(
                audio=audio_prompts,
                embedded_context=embedded_context_prompts,
                image=image_prompts,
            ),
        )
