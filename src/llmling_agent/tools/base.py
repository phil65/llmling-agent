"""Base tool classes."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any, Literal

import logfire
from pydantic_ai.tools import Tool as PydanticAiTool
import schemez

from llmling_agent.log import get_logger
from llmling_agent.utils.inspection import dataclasses_no_defaults_repr, execute
from llmling_agent_config.tools import ToolHints  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp.types import Tool as MCPTool
    from schemez import FunctionSchema, Property

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


@dataclass
class Tool[TOutputType = Any]:
    """Information about a registered tool."""

    callable: Callable[..., TOutputType]
    """The actual tool implementation"""

    name: str
    """The name of the tool."""

    description: str = ""
    """The description of the tool."""

    schema_override: schemez.OpenAIFunctionDefinition | None = None
    """Schema override. If not set, the schema is inferred from the callable."""

    hints: ToolHints | None = None
    """Hints for the tool."""

    import_path: str | None = None
    """The import path for the tool."""

    enabled: bool = True
    """Whether the tool is currently enabled"""

    source: ToolSource = "dynamic"
    """Where the tool came from."""

    requires_confirmation: bool = False
    """Whether tool execution needs explicit confirmation"""

    agent_name: str | None = None
    """The agent name as an identifier for agent-as-a-tool."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Additional tool metadata"""

    category: ToolKind | None = None
    """The category of the tool."""

    __repr__ = dataclasses_no_defaults_repr

    def to_pydantic_ai(self) -> PydanticAiTool:
        """Convert tool to Pydantic AI tool."""
        metadata = {
            **self.metadata,
            "agent_name": self.agent_name,
            "category": self.category,
        }
        return PydanticAiTool(
            function=self.callable,
            name=self.name,
            # takes_ctx=self.takes_ctx,
            # max_retries=self.max_retries,
            description=self.description,
            requires_approval=self.requires_confirmation,
            metadata=metadata,
        )

    @property
    def schema_obj(self) -> FunctionSchema:
        """Get the OpenAI function schema for the tool."""
        return schemez.create_schema(
            self.callable,
            name_override=self.name,
            description_override=self.description,
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
        return await execute(self.callable, *args, **kwargs, use_thread=True)

    @classmethod
    def from_code(
        cls,
        code: str,
        name: str | None = None,
        description: str | None = None,
    ) -> Tool[Any]:
        """Create a tool from a code string."""
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        func = next((v for v in namespace.values() if callable(v)), None)
        if not func:
            msg = "No callable found in provided code"
            raise ValueError(msg)
        return cls.from_callable(
            func,  # pyright: ignore[reportArgumentType]
            name_override=name,
            description_override=description,
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
        **kwargs: Any,
    ) -> Tool[TOutputType]:
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
            callable=callable_obj,  # pyright: ignore[reportArgumentType]
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
    ) -> Tool[Any]:
        """Allows importing crewai tools."""
        # vaidate_import("crewai_tools", "crewai")
        try:
            from crewai.tools import (  # type: ignore[import-not-found]
                BaseTool as CrewAiBaseTool,
            )
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
    ) -> Tool[Any]:
        """Create a tool from a LangChain tool."""
        # vaidate_import("langchain_core", "langchain")
        try:
            from langchain_core.tools import (  # type: ignore[import-not-found]
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
    ) -> Tool[Any]:
        """Create a tool from a AutoGen tool."""
        # vaidate_import("autogen_core", "autogen")
        try:
            from autogen_core import CancellationToken  # type: ignore[import-not-found]
            from autogen_core.tools import BaseTool  # type: ignore[import-not-found]
        except ImportError as e:
            msg = "autogent_core package not found."
            raise ImportError(msg) from e

        if not isinstance(tool, BaseTool):
            msg = f"Expected AutoGent BaseTool, got {type(tool)}"
            raise TypeError(msg)
        token = CancellationToken()

        input_model = tool.__class__.__orig_bases__[0].__args__[0]

        name = name_override or tool.name or tool.__class__.__name__.removesuffix("Tool")
        description = (
            description_override or tool.description or inspect.getdoc(tool.__class__) or ""
        )

        async def wrapper(**kwargs: Any) -> Any:
            # Convert kwargs to the expected input model
            model = input_model(**kwargs)
            return await tool.run(model, cancellation_token=token)

        return cls.from_callable(
            wrapper,
            name_override=name,
            description_override=description,
            schema_override=schema_override,
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
