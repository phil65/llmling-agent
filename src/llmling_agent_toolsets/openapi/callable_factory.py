"""Factory for creating typed callables from OpenAPI operations."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal, Union
from uuid import UUID

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


logger = get_logger(__name__)

FORMAT_MAP: dict[str, type] = {
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


class OpenAPICallableFactory:
    """Generates typed callables from OpenAPI operations."""

    def __init__(
        self,
        schemas: dict[str, dict[str, Any]],
        request_handler: Callable[..., Awaitable[dict[str, Any]]],
    ) -> None:
        """Initialize the factory.

        Args:
            schemas: Resolved component schemas from spec
            request_handler: Async callable performing HTTP requests.
                             Signature: (method, path, params, body) -> response_json
        """
        self._schemas = schemas
        self._request_handler = request_handler

    def resolve_schema_ref(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Resolve $ref to actual schema."""
        if (
            (ref := schema.get("$ref"))
            and isinstance(ref, str)
            and ref.startswith("#/components/schemas/")
        ):
            name = ref.split("/")[-1]
            return self._schemas.get(name, schema)
        return schema

    def get_type_for_schema(self, schema: dict[str, Any]) -> type | Any:  # noqa: PLR0911
        """Convert OpenAPI schema to Python type."""
        schema = self.resolve_schema_ref(schema)

        if "$ref" in schema:
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
                    item_type = self.get_type_for_schema(items)
                    return list[item_type]  # type: ignore
                return list[Any]

            case "object":
                if additional_props := schema.get("additionalProperties"):
                    value_type = self.get_type_for_schema(additional_props)
                    type DictType = dict[str, value_type]  # type: ignore
                    return DictType
                return dict[str, Any]

            case "null":
                return type(None)

            case None if "oneOf" in schema:
                types = [self.get_type_for_schema(s) for s in schema["oneOf"]]
                return Union[tuple(types)]  # noqa: UP007

            case None if "anyOf" in schema:
                types = [self.get_type_for_schema(s) for s in schema["anyOf"]]
                return Union[tuple(types)]  # noqa: UP007

            case None if "allOf" in schema:
                return dict[str, Any]

            case _:
                return Any

    def get_type_description(self, schema: dict[str, Any]) -> str:  # noqa: PLR0911
        """Get human-readable type description for docstrings."""
        schema = self.resolve_schema_ref(schema)

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
                    item_type = self.get_type_description(items)
                    return f"array of {item_type}"
                return "array"

            case "object":
                if properties := schema.get("properties"):
                    prop_types = [
                        f"{k}: {self.get_type_description(v)}" for k, v in properties.items()
                    ]
                    return f"object with {', '.join(prop_types)}"
                return "object"

            case t:
                return str(t) if t else "any"

    def create_docstring(self, config: dict[str, Any]) -> str:
        """Generate docstring from operation config."""
        lines: list[str] = []
        if description := config.get("description"):
            lines.append(description)
            lines.append("")

        if parameters := config.get("parameters"):
            lines.append("Args:")
            for param in parameters:
                schema = param.get("schema", {})
                description = schema.get("description", "No description")
                desc = param.get("description", description)
                required = " (required)" if param.get("required") else ""
                type_str = self.get_type_description(schema)
                lines.append(f"    {param['name']}: {desc}{required} ({type_str})")

        if responses := config.get("responses"):
            lines.append("")
            lines.append("Returns:")
            resps = [r for code, r in responses.items() if code.startswith("2")]
            lines.extend(f"    {r.get('description', '')}" for r in resps)

        return "\n".join(lines)

    def create_callable(
        self,
        operation_id: str,
        config: dict[str, Any],
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        """Create a typed async callable for an operation."""
        annotations: dict[str, Any] = {}
        required_params: set[str] = set()
        param_defaults: dict[str, Any] = {}

        for param in config.get("parameters", []):
            name = param["name"]
            schema = param.get("schema", {})

            param_type = self.get_type_for_schema(schema)
            is_required = param.get("required", False)
            annotations[name] = param_type if is_required else param_type | None

            if is_required:
                required_params.add(name)

            if "default" in schema:
                param_defaults[name] = schema["default"]

        parameters = config.get("parameters", [])
        path_template = config["path"]
        method = config["method"]
        request_handler = self._request_handler

        async def operation_method(**kwargs: Any) -> dict[str, Any]:
            """Dynamic method for API operation."""
            missing = required_params - set(kwargs)
            if missing:
                msg = f"Missing required parameters: {', '.join(sorted(missing))}"
                raise ValueError(msg)

            path = path_template
            request_params: dict[str, Any] = {}
            request_body: dict[str, Any] = {}

            for param in parameters:
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

            return await request_handler(
                method=method,
                path=path,
                params=request_params,
                body=request_body if request_body else None,
            )

        operation_method.__name__ = operation_id
        operation_method.__doc__ = self.create_docstring(config)
        operation_method.__annotations__ = {**annotations, "return": dict[str, Any]}

        return operation_method
