"""Base tool classes."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any, Literal

import logfire
from pydantic_ai.tools import Tool as PydanticAiTool
import schemez

from agentpool.log import get_logger
from agentpool.utils.inspection import (
    dataclasses_no_defaults_repr,
    execute,
    get_fn_name,
    get_fn_qualname,
)
from agentpool_config.tools import ToolHints


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp.types import Tool as MCPTool
    from pydantic_ai import UserContent
    from schemez import FunctionSchema, Property

    from agentpool.common_types import ToolSource
    from agentpool.tools.manager import ToolState

logger = get_logger(__name__)
ToolKind = Literal[
    "read",
    "edit",
    "delete",
    "move",
    "search",
    "execute",
    "think",
    "fetch",
    "switch_mode",
    "other",
]


@dataclass
class ToolResult:
    """Structured tool result with content for LLM and metadata for UI.

    This abstraction allows tools to return rich data that gets converted to
    agent-specific formats (pydantic-ai ToolReturn, FastMCP ToolResult, etc.).

    Attributes:
        content: What the LLM sees - can be string or list of content blocks
        structured_content: Machine-readable JSON data (optional)
        metadata: UI/application data that is NOT sent to the LLM
    """

    content: str | list[UserContent]
    """Content sent to the LLM (text, images, etc.)"""

    structured_content: dict[str, Any] | None = None
    """Structured JSON data for programmatic access (optional)"""

    metadata: dict[str, Any] | None = None
    """Metadata for UI/app use - NOT sent to LLM (diffs, diagnostics, etc.)."""


@dataclass
class Tool[TOutputType = Any]:
    """Base class for tools. Subclass and implement get_callable() or use FunctionTool."""

    name: str
    """The name of the tool."""

    description: str = ""
    """The description of the tool."""

    schema_override: schemez.OpenAIFunctionDefinition | None = None
    """Schema override. If not set, the schema is inferred from the callable."""

    hints: ToolHints = field(default_factory=ToolHints)
    """Hints for the tool."""

    import_path: str | None = None
    """The import path for the tool."""

    enabled: bool = True
    """Whether the tool is currently enabled"""

    source: ToolSource | str = "dynamic"
    """Where the tool came from."""

    requires_confirmation: bool = False
    """Whether tool execution needs explicit confirmation"""

    agent_name: str | None = None
    """The agent name as an identifier for agent-as-a-tool."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Additional tool metadata"""

    category: ToolKind | None = None
    """The category of the tool."""

    instructions: str | None = None
    """Instructions for how to use this tool effectively."""

    __repr__ = dataclasses_no_defaults_repr

    @abstractmethod
    def get_callable(self) -> Callable[..., TOutputType | Awaitable[TOutputType]]:
        """Get the callable for this tool. Subclasses must implement."""
        ...

    def to_pydantic_ai(self) -> PydanticAiTool:
        """Convert tool to Pydantic AI tool."""
        metadata = {**self.metadata, "agent_name": self.agent_name, "category": self.category}
        return PydanticAiTool(
            function=self.get_callable(),
            name=self.name,
            description=self.description,
            requires_approval=self.requires_confirmation,
            metadata=metadata,
        )

    @property
    def schema_obj(self) -> FunctionSchema:
        """Get the OpenAI function schema for the tool."""
        from pydantic_ai import RunContext

        from agentpool.agents.context import AgentContext

        return schemez.create_schema(
            self.get_callable(),
            name_override=self.name,
            description_override=self.description,
            exclude_types=[AgentContext, RunContext],
        )

    @property
    def schema(self) -> schemez.OpenAIFunctionTool:
        """Get the OpenAI function schema for the tool."""
        schema = self.schema_obj.model_dump_openai()
        if self.schema_override:
            schema["function"] = self.schema_override
        return schema

    def matches_filter(self, state: ToolState) -> bool:
        """Check if tool matches state filter."""
        match state:
            case "all":
                return True
            case "enabled":
                return self.enabled
            case "disabled":
                return not self.enabled

    @property
    def parameters(self) -> list[ToolParameter]:
        """Get information about tool parameters."""
        schema = self.schema["function"]
        properties: dict[str, Property] = schema.get("properties", {})  # type: ignore
        required: list[str] = schema.get("required", [])  # type: ignore

        return [
            ToolParameter(
                name=name,
                required=name in required,
                type_info=details.get("type"),
                description=details.get("description"),
            )
            for name, details in properties.items()
        ]

    def format_info(self, indent: str = "  ") -> str:
        """Format complete tool information."""
        lines = [f"{indent}→ {self.name}"]
        if self.description:
            lines.append(f"{indent}  {self.description}")
        if self.parameters:
            lines.append(f"{indent}  Parameters:")
            lines.extend(f"{indent}    {param}" for param in self.parameters)
        if self.metadata:
            lines.append(f"{indent}  Metadata:")
            lines.extend(f"{indent}    {k}: {v}" for k, v in self.metadata.items())
        return "\n".join(lines)

    @logfire.instrument("Executing tool {self.name} with args={args}, kwargs={kwargs}")
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute tool, handling both sync and async cases."""
        return await execute(self.get_callable(), *args, **kwargs, use_thread=True)

    async def execute_and_unwrap(self, *args: Any, **kwargs: Any) -> Any:
        """Execute tool and unwrap ToolResult if present.

        This is a convenience method for tests and direct tool usage that want
        plain content instead of ToolResult objects.

        Returns:
            If tool returns ToolResult, returns ToolResult.content.
            Otherwise returns the raw result.
        """
        result = await self.execute(*args, **kwargs)
        if isinstance(result, ToolResult):
            return result.content
        return result

    @classmethod
    def from_code(
        cls,
        code: str,
        name: str | None = None,
        description: str | None = None,
    ) -> FunctionTool[Any]:
        """Create a FunctionTool from a code string."""
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        func = next((v for v in namespace.values() if callable(v)), None)
        if not func:
            msg = "No callable found in provided code"
            raise ValueError(msg)
        return FunctionTool.from_callable(
            func, name_override=name, description_override=description
        )

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., TOutputType | Awaitable[TOutputType]] | str,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: schemez.OpenAIFunctionDefinition | None = None,
        hints: ToolHints | None = None,
        category: ToolKind | None = None,
        enabled: bool = True,
        source: ToolSource | str | None = None,
        **kwargs: Any,
    ) -> FunctionTool[TOutputType]:
        """Create a FunctionTool from a callable or import path."""
        return FunctionTool.from_callable(
            fn,
            name_override=name_override,
            description_override=description_override,
            schema_override=schema_override,
            hints=hints,
            category=category,
            enabled=enabled,
            source=source,
            **kwargs,
        )

    def to_mcp_tool(self) -> MCPTool:
        """Convert internal Tool to MCP Tool."""
        schema = self.schema
        from mcp.types import Tool as MCPTool, ToolAnnotations

        return MCPTool(
            name=schema["function"]["name"],
            description=schema["function"]["description"],
            inputSchema=schema["function"]["parameters"],  # pyright: ignore
            annotations=ToolAnnotations(
                title=self.name,
                readOnlyHint=self.hints.read_only if self.hints else None,
                destructiveHint=self.hints.destructive if self.hints else None,
                idempotentHint=self.hints.idempotent if self.hints else None,
                openWorldHint=self.hints.open_world if self.hints else None,
            ),
        )


@dataclass
class FunctionTool[TOutputType = Any](Tool[TOutputType]):
    """Tool wrapping a plain callable function."""

    callable: Callable[..., TOutputType | Awaitable[TOutputType]] = field(default=lambda: None)  # type: ignore[assignment]
    """The actual tool implementation."""

    def get_callable(self) -> Callable[..., TOutputType | Awaitable[TOutputType]]:
        """Return the wrapped callable."""
        return self.callable

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., TOutputType | Awaitable[TOutputType]] | str,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: schemez.OpenAIFunctionDefinition | None = None,
        hints: ToolHints | None = None,
        category: ToolKind | None = None,
        enabled: bool = True,
        source: ToolSource | str | None = None,
        **kwargs: Any,
    ) -> FunctionTool[TOutputType]:
        """Create a FunctionTool from a callable or import path string."""
        if isinstance(fn, str):
            import_path = fn
            from agentpool.utils import importing

            callable_obj = importing.import_callable(fn)
            name = getattr(callable_obj, "__name__", "unknown")
        else:
            callable_obj = fn
            module = fn.__module__
            if hasattr(fn, "__qualname__"):  # Regular function
                name = get_fn_name(fn)
                import_path = f"{module}.{get_fn_qualname(fn)}"
            else:  # Instance with __call__ method
                name = fn.__class__.__name__
                import_path = f"{module}.{fn.__class__.__qualname__}"

        return cls(
            name=name_override or name,
            description=description_override or inspect.getdoc(callable_obj) or "",
            callable=callable_obj,  # pyright: ignore[reportArgumentType]
            import_path=import_path,
            schema_override=schema_override,
            category=category,
            hints=hints or ToolHints(),
            enabled=enabled,
            source=source or "dynamic",
            **kwargs,
        )


@dataclass
class ToolParameter:
    """Information about a tool parameter."""

    name: str
    required: bool
    type_info: str | None = None
    description: str | None = None

    def __str__(self) -> str:
        """Format parameter info."""
        req = "*" if self.required else ""
        type_str = f": {self.type_info}" if self.type_info else ""
        desc = f" - {self.description}" if self.description else ""
        return f"{self.name}{req}{type_str}{desc}"


if __name__ == "__main__":
    import webbrowser

    t = Tool.from_callable(webbrowser.open)
