"""OpenAPI toolset implementation."""

from __future__ import annotations

import contextlib
from datetime import date, datetime
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, Literal, Union
from uuid import UUID

from upath import UPath
from upathtools import read_path

from llmling_agent.log import get_logger
from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    import httpx
    from jsonschema_path.typing import Schema
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


def dereference_openapi(
    input_path: str | UPath,
    redocly_path: str = "redocly",
    dereferenced: bool = True,
    remove_unused_components: bool = False,
    keep_url_references: bool = False,
    ext: str | None = None,
    config: str | None = None,
    extra_args: list[str] | None = None,
    error_on_missing: bool = False,
) -> str:
    """Bundle and dereference an OpenAPI spec using Redocly CLI.

    Args:
        input_path: Path or URL to the OpenAPI spec (YAML or JSON).
        redocly_path: Path to the Redocly CLI executable (default: 'redocly' from PATH).
        dereferenced: Produce a fully dereferenced bundle.
        remove_unused_components: Remove unused components.
        keep_url_references: Keep absolute URL references.
        ext: Output file extension (json, yaml, yml).
        config: Path to Redocly config file.
        extra_args: Additional CLI args as a list.
        error_on_missing: If True, raise if Redocly CLI is not found. If False,
                          return the input spec unchanged.

    Returns:
        The bundled OpenAPI spec as a string, or the original spec
        if Redocly is missing and error_on_missing is False.

    Raises:
        FileNotFoundError: If the Redocly CLI is not in PATH and error_on_missing is set.
        subprocess.CalledProcessError: If the Redocly CLI fails.
    """
    upath_input = UPath(input_path)
    exe = shutil.which(redocly_path)
    if exe is None:
        if error_on_missing:
            msg = (
                f"Redocly CLI executable {redocly_path!r} not found in PATH. "
                "Install it with 'npm install -g @redocly/cli' or specify the full path."
            )
            raise FileNotFoundError(msg)
        return upath_input.read_text(encoding="utf-8")

    with tempfile.NamedTemporaryFile(suffix=f".{ext or 'yaml'}", delete=False) as tmp:
        output_path = tmp.name

    cmd = [exe, "bundle", str(input_path)]
    if dereferenced:
        cmd.append("--dereferenced")
    if remove_unused_components:
        cmd.append("--remove-unused-components")
    if keep_url_references:
        cmd.append("--keep-url-references")
    if ext:
        cmd.extend(["--ext", ext])
    if config:
        cmd.extend(["--config", config])
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["--output", output_path])

    try:
        subprocess.run(cmd, check=True)
        with UPath(output_path).open(encoding="utf-8") as f:
            spec = f.read()
    finally:
        with contextlib.suppress(Exception):
            UPath(output_path).unlink()

    return spec


def parse_operations(paths: dict) -> dict[str, dict[str, Any]]:
    """Parse OpenAPI paths into operation configurations."""
    operations = {}
    for path, path_item in paths.items():
        for method, operation in path_item.items():
            if method not in {"get", "post", "put", "delete", "patch"}:
                continue

            # Generate operation ID if not provided
            op_id = operation.get("operationId")
            if not op_id:
                op_id = f"{method}_{path.replace('/', '_').strip('_')}"

            # Collect all parameters (path, query, header)
            params = operation.get("parameters", [])
            if (
                (request_body := operation.get("requestBody"))
                and (content := request_body.get("content", {}))
                and (json_schema := content.get("application/json", {}).get("schema"))
                and (properties := json_schema.get("properties", {}))
            ):
                # Convert request body to parameters
                for name, schema in properties.items():
                    params.append({
                        "name": name,
                        "in": "body",
                        "required": name in json_schema.get("required", []),
                        "schema": schema,
                        "description": schema.get("description", ""),
                    })

            operations[op_id] = {
                "method": method,
                "path": path,
                "description": operation.get("description", ""),
                "parameters": params,
                "responses": operation.get("responses", {}),
            }

    return operations


class OpenAPITools(ResourceProvider):
    """Provider for OpenAPI-based tools."""

    def __init__(
        self,
        spec: JoinablePathLike,
        base_url: str = "",
        name: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(name=name or f"openapi_{base_url}")
        self.spec_url = spec
        self.base_url = base_url
        self.headers = headers or {}
        self._client: httpx.AsyncClient | None = None
        self._spec: Schema | None = None
        self._schemas: dict[str, Any] = {}
        self._operations: dict[str, Any] = {}

    async def get_tools(self) -> list[Tool]:
        """Get all API operations as tools."""
        # Only load spec if not already loaded
        if not self._spec:
            await self._load_spec()
        # Create tools from operations
        tools = []
        for op_id, config in self._operations.items():
            method = self._create_operation_method(op_id, config)
            meta = {"operation": op_id}
            tool = Tool.from_callable(method, source="openapi", metadata=meta)
            tools.append(tool)
        return tools

    async def _load_spec(self):
        import httpx
        import yaml

        # Create client if needed
        if not self._client:
            self._client = httpx.AsyncClient(base_url=self.base_url, headers=self.headers)

        # Load spec
        try:
            if str(self.spec_url).startswith(("http://", "https://")):
                response = await self._client.get(str(self.spec_url))
                response.raise_for_status()
                content = response.text
            else:
                content = await read_path(self.spec_url)

            self._spec = yaml.safe_load(content)
            assert self._spec
            self._schemas = self._spec.get("components", {}).get("schemas", {})
            self._operations = parse_operations(self._spec.get("paths", {}))

        except Exception as e:
            msg = f"Failed to load OpenAPI spec from {self.spec_url}"
            raise ValueError(msg) from e
        else:
            return self._spec

    def _resolve_schema_ref(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Resolve schema reference."""
        if ref := schema.get("$ref"):
            # Extract schema name from #/components/schemas/Name
            name = ref.split("/")[-1]
            return self._schemas[name]
        return schema

    def _get_type_for_schema(self, schema: dict[str, Any]) -> type | Any:  # noqa: PLR0911
        """Convert OpenAPI schema to Python type."""
        schema = self._resolve_schema_ref(schema)

        match schema.get("type"):
            case "string":
                if enum := schema.get("enum"):
                    return Literal[tuple(enum)]  # type: ignore
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
                item_type = self._get_type_for_schema(schema["items"])
                return list[item_type]  # type: ignore

            case "object":
                if additional_props := schema.get("additionalProperties"):
                    # Dictionary with specified value type
                    value_type = self._get_type_for_schema(additional_props)
                    # Create type alias for the dict type
                    type DictType = dict[str, value_type]  # type: ignore
                    return DictType
                if _properties := schema.get("properties"):
                    # Convert to dict with specific types
                    return dict[str, Any]
                return dict[str, Any]

            case "null":
                return type(None)

            case None if "oneOf" in schema:
                types = [self._get_type_for_schema(s) for s in schema["oneOf"]]
                return Union[tuple(types)]  # type: ignore  # noqa: UP007

            case None if "anyOf" in schema:
                types = [self._get_type_for_schema(s) for s in schema["anyOf"]]
                return Union[tuple(types)]  # type: ignore  # noqa: UP007

            case None if "allOf" in schema:
                # For allOf, we'd need to merge schemas - using dict for now
                return dict[str, Any]

            case _:
                from typing import Any as AnyType

                return AnyType

    def _create_operation_method(
        self,
        op_id: str,
        config: dict[str, Any],
    ) -> Any:
        """Create a method for an operation with proper type hints."""
        # Create parameter annotations
        annotations: dict[str, Any] = {}
        required_params: set[str] = set()
        param_defaults: dict[str, Any] = {}

        for param in config["parameters"]:
            name = param["name"]
            schema = param.get("schema", {})

            # Get type
            param_type = self._get_type_for_schema(schema)
            annotations[name] = (
                param_type | None if not param.get("required") else param_type
            )

            # Track required params
            if param.get("required"):
                required_params.add(name)

            # Get default value if any
            if "default" in schema:
                param_defaults[name] = schema["default"]

        async def operation_method(**kwargs: Any) -> dict[str, Any]:
            """Dynamic method for API operation."""
            # Validate required parameters
            missing = required_params - set(kwargs)
            if missing:
                msg = f"Missing required parameters: {', '.join(missing)}"
                raise ValueError(msg)

            path = config["path"]
            request_params = {}
            request_body = {}

            # Process parameters based on their location
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

            # Send request
            if not self._client:
                import httpx

                self._client = httpx.AsyncClient(
                    base_url=self.base_url, headers=self.headers
                )
            response = await self._client.request(
                method=config["method"],
                url=path,
                params=request_params,
                json=request_body if request_body else None,
            )
            response.raise_for_status()
            return response.json()

        # Set method metadata
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

        # Add parameter descriptions
        if config["parameters"]:
            lines.append("Args:")
            for param in config["parameters"]:
                schema = param.get("schema", {})
                description = schema.get("description", "No description")
                desc = param.get("description", description)
                required = " (required)" if param.get("required") else ""
                type_str = self._get_type_description(schema)
                lines.append(f"    {param['name']}: {desc}{required} ({type_str})")

        # Add response info
        if responses := config["responses"]:
            lines.append("")
            lines.append("Returns:")
            resps = [r for code, r in responses.items() if code.startswith("2")]
            lines.extend(f"    {r.get('description', '')}" for r in resps)
        return "\n".join(lines)

    def _get_type_description(self, schema: dict[str, Any]) -> str:  # noqa: PLR0911
        """Get human-readable type description."""
        schema = self._resolve_schema_ref(schema)

        match schema.get("type"):
            case "string":
                if enum := schema.get("enum"):
                    return f"one of: {', '.join(repr(e) for e in enum)}"
                if fmt := schema.get("format"):
                    return f"string ({fmt})"
                return "string"

            case "array":
                item_type = self._get_type_description(schema["items"])
                return f"array of {item_type}"

            case "object":
                if properties := schema.get("properties"):
                    prop_types = [
                        f"{k}: {self._get_type_description(v)}"
                        for k, v in properties.items()
                    ]
                    return f"object with {', '.join(prop_types)}"
                return "object"

            case t:
                return str(t)
