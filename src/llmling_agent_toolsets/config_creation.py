"""Config creation toolset with schema validation."""

from __future__ import annotations

import json
import tomllib
from typing import TYPE_CHECKING, Literal

import jsonschema
from schemez.helpers import json_schema_to_pydantic_code
import upath
from upathtools.filesystems.file_filesystems.jsonschema_fs import JsonSchemaFileSystem
import yamling

from llmling_agent.resource_providers import StaticResourceProvider


if TYPE_CHECKING:
    from upath.types import JoinablePathLike


MarkupType = Literal["yaml", "json", "toml"]


def _parse_content(content: str, markup: MarkupType) -> dict:
    """Parse content based on markup type."""
    match markup:
        case "yaml":
            return yamling.load_yaml(content)
        case "json":
            return json.loads(content)
        case "toml":
            return tomllib.loads(content)


def _format_validation_error(error: jsonschema.ValidationError) -> str:
    """Format a validation error for user-friendly display."""
    path = " -> ".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
    return f"At '{path}': {error.message}"


class ConfigCreationTools(StaticResourceProvider):
    """Provider for config creation and validation tools."""

    def __init__(
        self,
        schema_path: JoinablePathLike,
        markup: MarkupType = "yaml",
        name: str = "config_creation",
    ) -> None:
        """Initialize the config creation toolset.

        Args:
            schema_path: Path to the JSON schema file
            markup: Markup language for configs (yaml, json, toml)
            name: Namespace for the tools
        """
        super().__init__(name=name)
        self._schema_path = upath.UPath(schema_path)
        self._markup = markup
        self._schema: dict | None = None
        self._schema_fs: JsonSchemaFileSystem | None = None

        self.add_tool(
            self.create_tool(
                self._create_config,
                category="write",
                read_only=False,
                idempotent=True,
                name_override="create_config",
                description_override=(
                    f"Create and validate a {markup.upper()} configuration. "
                    "Returns validation result and any errors."
                ),
            )
        )
        self.add_tool(
            self.create_tool(
                self._show_schema_as_code,
                category="read",
                read_only=True,
                idempotent=True,
                name_override="show_schema_as_code",
                description_override=(
                    "Show the JSON schema as Python Pydantic code for easier understanding."
                ),
            )
        )
        self.add_tool(
            self.create_tool(
                self._list_schema,
                category="read",
                read_only=True,
                idempotent=True,
                name_override="list_schema",
                description_override=(
                    "List contents at a path in the JSON schema. "
                    "Use '/' for root, '/$defs' for definitions, "
                    "'/$defs/{TypeName}/properties' for type properties."
                ),
            )
        )
        self.add_tool(
            self.create_tool(
                self._read_schema_node,
                category="read",
                read_only=True,
                idempotent=True,
                name_override="read_schema_node",
                description_override=(
                    "Read the JSON schema at a specific path. "
                    "E.g. '/$defs/AgentConfig/properties/model' to see the model field schema."
                ),
            )
        )

    def _load_schema(self) -> dict:
        """Load and cache the JSON schema."""
        if self._schema is None:
            content = self._schema_path.read_text()
            self._schema = json.loads(content)
        return self._schema

    def _get_schema_fs(self) -> JsonSchemaFileSystem:
        """Get or create the JSON schema filesystem."""
        if self._schema_fs is None:
            self._schema_fs = JsonSchemaFileSystem(schema_url=str(self._schema_path))
        return self._schema_fs

    async def _create_config(self, content: str) -> str:
        """Create and validate a configuration.

        Args:
            content: The configuration content in the configured markup format

        Returns:
            Validation result message
        """
        schema = self._load_schema()
        try:
            data = _parse_content(content, self._markup)
        except Exception as e:
            return f"Failed to parse {self._markup.upper()}: {e}"

        errors: list[str] = []
        validator = jsonschema.Draft202012Validator(schema)
        for error in validator.iter_errors(data):
            errors.append(_format_validation_error(error))

        if errors:
            error_list = "\n".join(f"- {e}" for e in errors[:10])
            suffix = f"\n... and {len(errors) - 10} more errors" if len(errors) > 10 else ""
            return f"Validation failed with {len(errors)} error(s):\n{error_list}{suffix}"

        return "Configuration is valid! Successfully validated against schema."

    async def _show_schema_as_code(self) -> str:
        """Show the JSON schema as Python Pydantic code.

        Returns:
            Python code representation of the schema
        """
        schema = self._load_schema()
        return json_schema_to_pydantic_code(
            schema,
            class_name="Config",
            base_class="pydantic.BaseModel",
        )

    async def _list_schema(self, path: str = "/") -> str:
        """List contents at a path in the JSON schema.

        Args:
            path: Path to list (e.g. '/', '/$defs', '/$defs/AgentConfig/properties')

        Returns:
            Formatted listing of schema contents at the path
        """
        fs = self._get_schema_fs()
        try:
            items = fs.ls(path, detail=True)
        except FileNotFoundError:
            return f"Path not found: {path}"

        if not items:
            return f"No contents at: {path}"

        lines = [f"Contents of {path}:\n"]
        for item in items:
            name = item["name"]
            item_type = item["type"]
            icon = "📁" if item_type == "directory" else "📄"

            parts = [f"{icon} {name}"]

            if schema_type := item.get("schema_type"):
                parts.append(f"[{schema_type}]")

            if item.get("required"):
                parts.append("(required)")

            if desc := item.get("description"):
                # Truncate long descriptions
                desc_short = desc[:60] + "..." if len(desc) > 60 else desc
                parts.append(f"- {desc_short}")

            lines.append("  " + " ".join(parts))

        return "\n".join(lines)

    async def _read_schema_node(self, path: str) -> str:
        """Read the JSON schema at a specific path.

        Args:
            path: Path to read (e.g. '/$defs/AgentConfig', '/properties/agents')

        Returns:
            JSON schema content at the path
        """
        fs = self._get_schema_fs()
        try:
            content = fs.cat(path)
            # Parse and re-format for readability
            schema_data = json.loads(content)
            return json.dumps(schema_data, indent=2)
        except FileNotFoundError:
            return f"Path not found: {path}"
        except json.JSONDecodeError as e:
            return f"Failed to parse schema at {path}: {e}"
