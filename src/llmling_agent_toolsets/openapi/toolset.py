"""OpenAPI toolset provider."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal, Union
from uuid import UUID

from upath import UPath
from upathtools import read_path

from llmling_agent.log import get_logger
from llmling_agent.resource_providers import ResourceProvider
from llmling_agent.tools.base import Tool
from llmling_agent_toolsets.openapi.loader import load_openapi_spec, parse_operations


if TYPE_CHECKING:
    import httpx
    from upath.types import JoinablePathLike

logger = get_logger(__name__)

# Map OpenAPI formats to Python types
FORMAT_MAP = {
    "date": date,
    "date-time": datetime,
    "uuid": UUID,
    "email": str,
    "uri": str,
    "hostname": str,
    "ipv4": str,
    "ipv6": str,
    "byte": bytes,
    "binary": bytes,
    "password": str,
}


class OpenAPITools(ResourceProvider):
    """Provider for OpenAPI-based tools."""

    def __init__(
        self,
        spec: JoinablePathLike,
        base_url: str = "",
        name: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(name=name or f"openapi_{base_url}")
        self.spec_url = spec
        self.base_url = base_url
        self.headers = headers or {}
        self._client: httpx.AsyncClient | None = None
        self._spec: dict[str, Any] | None = None
        self._schemas: dict[str, dict[str, Any]] = {}
        self._operations: dict[str, Any] = {}

    async def get_tools(self) -> list[Tool]:
        """Get all API operations as tools."""
        if not self._spec:
            await self._load_spec()

        tools = []
        for op_id, config in self._operations.items():
            method = self._create_operation_method(op_id, config)
            meta = {"operation": op_id}
            tool = Tool.from_callable(method, source="openapi", metadata=meta)
            tools.append(tool)
        return tools

    async def _load_spec(self) -> dict[str, Any]:
        import httpx

        if not self._client:
            self._client = httpx.AsyncClient(base_url=self.base_url, headers=self.headers)

        try:
            spec_str = str(self.spec_url)
            if spec_str.startswith(("http://", "https://")):
                self._spec = load_openapi_spec(spec_str)
            else:
                path = UPath(self.spec_url)
                if path.exists():
                    self._spec = load_openapi_spec(path)
                else:
                    # Try reading via upathtools for remote/special paths
                    content = await read_path(self.spec_url)
                    import tempfile

                    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                        f.write(content)
                        temp_path = f.name
                    try:
                        self._spec = load_openapi_spec(temp_path)
                    finally:
                        UPath(temp_path).unlink(missing_ok=True)

            if not self._spec:
                msg = f"Empty or invalid OpenAPI spec from {self.spec_url}"
                raise ValueError(msg)  # noqa: TRY301

            self._schemas = self._spec.get("components", {}).get("schemas", {})
            self._operations = parse_operations(self._spec.get("paths", {}))

            if not self._operations:
                logger.warning(
                    "No operations found in spec %s.",
                    self.spec_url,
                )

        except Exception as e:
            msg = f"Failed to load OpenAPI spec from {self.spec_url}"
            raise ValueError(msg) from e
        else:
            return self._spec

    def _resolve_schema_ref(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Resolve schema reference."""
        if ref := schema.get("$ref"):  # noqa: SIM102
            if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
                name = ref.split("/")[-1]
                return self._schemas.get(name, schema)
        return schema

    def _get_type_for_schema(self, schema: dict[str, Any]) -> type | Any:  # noqa: PLR0911
        """Convert OpenAPI schema to Python type."""
        schema = self._resolve_schema_ref(schema)

        if "$ref" in schema:
            # Fallback for circular refs or refs to missing components
            logger.debug("Unresolved $ref in schema, using Any: %s", schema.get("$ref"))
            return Any

        match schema.get("type"):
            case "string":
                if enum := schema.get("enum"):
                    return Literal[tuple(enum)]
                if fmt := schema.get("format"):
                    return FORMAT_MAP.get(fmt, str)
                return str

            case "integer":
                return int

            case "number":
                return float

            case "boolean":
                return bool

            case "array":
                if items := schema.get("items"):
                    item_type = self._get_type_for_schema(items)
                    return list[item_type]  # type: ignore
                return list[Any]

            case "object":
                if additional_props := schema.get("additionalProperties"):
                    value_type = self._get_type_for_schema(additional_props)
                    type DictType = dict[str, value_type]  # type: ignore
                    return DictType
                return dict[str, Any]

            case "null":
                return type(None)

            case None if "oneOf" in schema:
                types = [self._get_type_for_schema(s) for s in schema["oneOf"]]
                return Union[tuple(types)]  # noqa: UP007

            case None if "anyOf" in schema:
                types = [self._get_type_for_schema(s) for s in schema["anyOf"]]
                return Union[tuple(types)]  # noqa: UP007

            case None if "allOf" in schema:
                return dict[str, Any]

            case _:
                return Any

    def _create_operation_method(self, op_id: str, config: dict[str, Any]) -> Any:
        """Create a method for an operation with proper type hints."""
        annotations: dict[str, Any] = {}
        required_params: set[str] = set()
        param_defaults: dict[str, Any] = {}

        for param in config["parameters"]:
            name = param["name"]
            schema = param.get("schema", {})

            param_type = self._get_type_for_schema(schema)
            annotations[name] = param_type | None if not param.get("required") else param_type

            if param.get("required"):
                required_params.add(name)

            if "default" in schema:
                param_defaults[name] = schema["default"]

        async def operation_method(**kwargs: Any) -> dict[str, Any]:
            """Dynamic method for API operation."""
            missing = required_params - set(kwargs)
            if missing:
                msg = f"Missing required parameters: {', '.join(missing)}"
                raise ValueError(msg)

            path = config["path"]
            request_params = {}
            request_body = {}

            for param in config["parameters"]:
                name = param["name"]
                if name not in kwargs and name in param_defaults:
                    kwargs[name] = param_defaults[name]

                if name in kwargs:
                    match param["in"]:
                        case "path":
                            path = path.replace(f"{{{name}}}", str(kwargs[name]))
                        case "query":
                            request_params[name] = kwargs[name]
                        case "body":
                            request_body[name] = kwargs[name]

            if not self._client:
                import httpx

                self._client = httpx.AsyncClient(base_url=self.base_url, headers=self.headers)

            response = await self._client.request(
                method=config["method"],
                url=path,
                params=request_params,
                json=request_body if request_body else None,
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

        operation_method.__name__ = op_id
        operation_method.__doc__ = self._create_docstring(config)
        operation_method.__annotations__ = {**annotations, "return": dict[str, Any]}

        return operation_method

    def _create_docstring(self, config: dict[str, Any]) -> str:
        """Create detailed docstring from operation info."""
        lines = []
        if description := config["description"]:
            lines.append(description)
            lines.append("")

        if config["parameters"]:
            lines.append("Args:")
            for param in config["parameters"]:
                schema = param.get("schema", {})
                description = schema.get("description", "No description")
                desc = param.get("description", description)
                required = " (required)" if param.get("required") else ""
                type_str = self._get_type_description(schema)
                lines.append(f"    {param['name']}: {desc}{required} ({type_str})")

        if responses := config["responses"]:
            lines.append("")
            lines.append("Returns:")
            resps = [r for code, r in responses.items() if code.startswith("2")]
            lines.extend(f"    {r.get('description', '')}" for r in resps)
        return "\n".join(lines)

    def _get_type_description(self, schema: dict[str, Any]) -> str:  # noqa: PLR0911
        """Get human-readable type description."""
        schema = self._resolve_schema_ref(schema)

        if "$ref" in schema:
            return "any"

        match schema.get("type"):
            case "string":
                if enum := schema.get("enum"):
                    return f"one of: {', '.join(repr(e) for e in enum)}"
                if fmt := schema.get("format"):
                    return f"string ({fmt})"
                return "string"

            case "array":
                if items := schema.get("items"):
                    item_type = self._get_type_description(items)
                    return f"array of {item_type}"
                return "array"

            case "object":
                if properties := schema.get("properties"):
                    prop_types = [
                        f"{k}: {self._get_type_description(v)}" for k, v in properties.items()
                    ]
                    return f"object with {', '.join(prop_types)}"
                return "object"

            case t:
                return str(t) if t else "any"
