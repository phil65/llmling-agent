"""Base tool classes."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any, Literal, Self

import logfire
import schemez

from llmling_agent.log import get_logger
from llmling_agent.utils.inspection import execute
from llmling_agent_config.tools import ToolHints  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Callable

    from schemez.typedefs import Property, ToolParameters

    from llmling_agent.agent import AgentContext
    from llmling_agent.common_types import ToolSource
    from llmling_agent.tools.manager import ToolState


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


@dataclass(frozen=True)
class ToolContext:
    """Context for tool execution confirmation."""

    name: str
    """Name of the tool being executed"""

    args: dict[str, Any]
    """Arguments being passed to the tool"""

    schema: schemez.OpenAIFunctionTool
    """Complete OpenAI function schema"""

    runtime_ctx: AgentContext
    """Runtime context from agent"""

    @property
    def description(self) -> str | None:
        """Get tool description from schema."""
        return self.schema["function"].get("description")

    @property
    def parameters(self) -> ToolParameters:
        """Get parameter definitions from schema."""
        return self.schema["function"].get("parameters", {})  # type: ignore

    def __str__(self) -> str:
        """Format tool context for logging/display."""
        return (
            f"Tool: {self.name}\n"
            f"Arguments: {self.args}\n"
            f"Description: {self.description or 'N/A'}"
        )


@dataclass
class Tool:
    """Information about a registered tool."""

    callable: Callable
    """The actual tool implementation"""

    name: str

    description: str = ""

    schema_override: schemez.OpenAIFunctionDefinition | None = None

    hints: ToolHints | None = None

    import_path: str | None = None

    enabled: bool = True
    """Whether the tool is currently enabled"""

    source: ToolSource = "runtime"
    """Where the tool came from."""

    priority: int = 100
    """Priority for tool execution (lower = higher priority)"""

    requires_confirmation: bool = False
    """Whether tool execution needs explicit confirmation"""

    requires_capability: str | None = None
    """Optional capability required to use this tool"""

    agent_name: str | None = None
    """The agent name as an identifier for agent-as-a-tool."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Additional tool metadata"""

    cache_enabled: bool = False
    """Whether to enable caching for this tool."""

    category: ToolKind | None = None
    """The category of the tool."""

    @property
    def schema(self) -> schemez.OpenAIFunctionTool:
        """Get the OpenAI function schema for the tool."""
        schema = schemez.create_schema(self.callable).model_dump_openai()
        schema["function"]["name"] = self.name
        schema["function"]["description"] = self.description
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
        return await execute(self.callable, *args, **kwargs, use_thread=True)

    @classmethod
    def from_code(
        cls,
        code: str,
        name: str | None = None,
        description: str | None = None,
    ) -> Self:
        """Create a tool from a code string."""
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        func = next((v for v in namespace.values() if callable(v)), None)
        if not func:
            msg = "No callable found in provided code"
            raise ValueError(msg)
        return cls.from_callable(
            func, name_override=name, description_override=description
        )

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any] | str,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: schemez.OpenAIFunctionDefinition | None = None,
        hints: ToolHints | None = None,
        **kwargs: Any,
    ) -> Self:
        if isinstance(fn, str):
            import_path = fn
            from llmling_agent.utils import importing

            callable_obj = importing.import_callable(fn)
            name = getattr(callable_obj, "__name__", "unknown")
            import_path = fn
        else:
            callable_obj = fn
            module = fn.__module__
            if hasattr(fn, "__qualname__"):  # Regular function
                name = fn.__name__
                import_path = f"{module}.{fn.__qualname__}"
            else:  # Instance with __call__ method
                name = fn.__class__.__name__
                import_path = f"{module}.{fn.__class__.__qualname__}"

        return cls(
            callable=callable_obj,
            name=name_override or name,
            description=description_override or inspect.getdoc(callable_obj) or "",
            import_path=import_path,
            schema_override=schema_override,
            hints=hints,
            **kwargs,
        )

    @classmethod
    def from_crewai_tool(
        cls,
        tool: Any,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: schemez.OpenAIFunctionDefinition | None = None,
        **kwargs: Any,
    ) -> Self:
        """Allows importing crewai tools."""
        # vaidate_import("crewai_tools", "crewai")
        try:
            from crewai.tools import BaseTool as CrewAiBaseTool  # pyright: ignore
        except ImportError as e:
            msg = "crewai package not found. Please install it with 'pip install crewai'"
            raise ImportError(msg) from e

        if not isinstance(tool, CrewAiBaseTool):
            msg = f"Expected CrewAI BaseTool, got {type(tool)}"
            raise TypeError(msg)

        return cls.from_callable(
            tool._run,
            name_override=name_override or tool.__class__.__name__.removesuffix("Tool"),
            description_override=description_override or tool.description,
            schema_override=schema_override,
            **kwargs,
        )

    @classmethod
    def from_langchain_tool(
        cls,
        tool: Any,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: schemez.OpenAIFunctionDefinition | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a tool from a LangChain tool."""
        # vaidate_import("langchain_core", "langchain")
        try:
            from langchain_core.tools import (  # pyright: ignore
                BaseTool as LangChainBaseTool,
            )
        except ImportError as e:
            msg = "langchain-core package not found."
            raise ImportError(msg) from e

        if not isinstance(tool, LangChainBaseTool):
            msg = f"Expected LangChain BaseTool, got {type(tool)}"
            raise TypeError(msg)

        return cls.from_callable(
            tool.invoke,
            name_override=name_override or tool.name,
            description_override=description_override or tool.description,
            schema_override=schema_override,
            **kwargs,
        )

    @classmethod
    def from_autogen_tool(
        cls,
        tool: Any,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: schemez.OpenAIFunctionDefinition | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a tool from a AutoGen tool."""
        # vaidate_import("autogen_core", "autogen")
        try:
            from autogen_core import CancellationToken  # pyright: ignore
            from autogen_core.tools import BaseTool  # pyright: ignore
        except ImportError as e:
            msg = "autogent_core package not found."
            raise ImportError(msg) from e

        if not isinstance(tool, BaseTool):
            msg = f"Expected AutoGent BaseTool, got {type(tool)}"
            raise TypeError(msg)
        token = CancellationToken()

        input_model = tool.__class__.__orig_bases__[0].__args__[0]  # type: ignore

        name = name_override or tool.name or tool.__class__.__name__.removesuffix("Tool")
        description = (
            description_override
            or tool.description
            or inspect.getdoc(tool.__class__)
            or ""
        )

        async def wrapper(**kwargs: Any) -> Any:
            # Convert kwargs to the expected input model
            model = input_model(**kwargs)
            return await tool.run(model, cancellation_token=token)

        return cls.from_callable(
            wrapper,  # type: ignore
            name_override=name,
            description_override=description,
            schema_override=schema_override,
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
