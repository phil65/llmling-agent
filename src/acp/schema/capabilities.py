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
