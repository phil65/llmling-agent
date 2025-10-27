# /// script
# dependencies = ["datamodel-code-generator[http]", "anyenv[httpx]"]
# ///


from __future__ import annotations

import asyncio
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Literal

import anyenv


ROOT = Path(__file__).resolve().parents[1]

SCHEMA_URL = "https://raw.githubusercontent.com/zed-industries/agent-client-protocol/refs/heads/main/schema/schema.json"
META_URL = "https://raw.githubusercontent.com/zed-industries/agent-client-protocol/refs/heads/main/schema/meta.json"


def convert_oneof_const_to_enum(schema: dict) -> dict:
    """Convert oneOf patterns with const values to enum format.

    This ensures datamodel-code-generator creates Literal types instead of plain str.
    Works on the schema WITHOUT dereferencing to preserve names.
    """
    schema = copy.deepcopy(schema)

    def process_schema(obj: dict, path: str = "") -> None:
        if isinstance(obj, dict):
            # Check if this is a oneOf pattern with all const values
            if "oneOf" in obj and isinstance(obj["oneOf"], list):
                all_const = all(
                    isinstance(item, dict)
                    and item.get("type") == "string"
                    and "const" in item
                    for item in obj["oneOf"]
                )
                if all_const:
                    # Convert to enum format
                    enum_values = [item["const"] for item in obj["oneOf"]]
                    obj["type"] = "string"
                    obj["enum"] = enum_values
                    # Keep the description if available
                    if "description" not in obj:
                        # Try to get description from parent or first item
                        for item in obj["oneOf"]:
                            if "description" in item:
                                obj["description"] = item["description"]
                                break
                    del obj["oneOf"]
                    print(f"Converted oneOf+const to enum at {path}")

            # Recursively process all values
            for key, value in list(obj.items()):
                if isinstance(value, dict):
                    process_schema(value, f"{path}.{key}")
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            process_schema(item, f"{path}.{key}[{i}]")

    # Process the entire schema including $defs
    process_schema(schema, "")
    return schema


async def generate_schema() -> None:
    """Generate schema.py from ACP schema."""
    schema_out = ROOT / "src" / "acp" / "schema.py"

    # Create a temporary file for the schema JSON
    temp_dir = Path(tempfile.gettempdir())
    temp_schema_path = temp_dir / "schema.json"

    # Fetch schema data asynchronously
    schema_data = await anyenv.get_json(SCHEMA_URL, return_type=dict)

    # Convert oneOf+const patterns to enums for proper Literal generation
    # Do NOT dereference - this preserves the semantic names from $defs
    preprocessed_schema = convert_oneof_const_to_enum(schema_data)

    temp_schema_path.write_text(anyenv.dump_json(preprocessed_schema, indent=True))

    try:
        cmd = [
            sys.executable,
            "-m",
            "datamodel_code_generator",
            "--input",
            str(temp_schema_path),
            "--input-file-type",
            "jsonschema",
            "--output",
            str(schema_out),
            "--target-python-version",
            "3.12",
            "--collapse-root-models",
            "--output-model-type",
            "pydantic_v2.BaseModel",
            "--base-class",
            "acp.base.Schema",
            "--use-annotated",
            "--no-alias",
            "--use-one-literal-as-default",
            "--enum-field-as-literal",
            "all",
            "--use-double-quotes",
            "--use-union-operator",
            "--use-standard-collections",
            # "--use-field-description",
            "--use-schema-description",
            "--snake-case-field",
            "--use-generic-container-types",
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode or 1, cmd, stdout, stderr
            )

        # Post-process to rename numbered classes
        _rename_numbered_classes(schema_out)
    finally:
        # Clean up temporary file
        temp_schema_path.unlink(missing_ok=True)


async def generate_meta() -> None:
    """Generate meta.py from ACP metadata."""
    meta_out = ROOT / "src" / "acp" / "meta.py"

    # Fetch meta data asynchronously
    meta_data = await anyenv.get_json(META_URL, return_type=dict)

    agent_methods = meta_data.get("agentMethods", {})
    client_methods = meta_data.get("clientMethods", {})
    version = meta_data.get("version", 1)

    # Extract just the values (actual method names)
    agent_method_names = sorted(agent_methods.values())
    client_method_names = sorted(client_methods.values())

    # Generate Literal types
    agent_literal = ",\n    ".join(f'"{name}"' for name in agent_method_names)
    client_literal = ",\n    ".join(f'"{name}"' for name in client_method_names)

    meta_out.write_text(
        f'"""Auto-generated metadata file."""\n\n'
        f"from typing import Literal\n\n"
        f"# This file is generated from {META_URL}.\n"
        f"# Do not edit by hand.\n\n"
        f"AgentMethod = Literal[\n    {agent_literal},\n]\n\n"
        f"ClientMethod = Literal[\n    {client_literal},\n]\n\n"
        f"PROTOCOL_VERSION = {int(version)}\n"
    )


async def main() -> None:
    """Main async entry point."""
    # Generate both files concurrently
    await asyncio.gather(generate_schema(), generate_meta())

    # Format generated files with ruff
    schema_out = ROOT / "src" / "acp" / "schema.py"
    meta_out = ROOT / "src" / "acp" / "meta.py"

    # Format files concurrently, but maintain format->check sequence per file
    async def format_file(path: Path) -> None:
        await _format_with_ruff(path, method="format")
        await _format_with_ruff(path, method="check")

    await asyncio.gather(format_file(schema_out), format_file(meta_out))


async def _format_with_ruff(path: Path, method: Literal["format", "check"]) -> None:
    """Format a Python file with ruff asynchronously."""
    try:
        if method == "format":
            cmd = ["uv", "run", "ruff", "format", str(path)]
        else:
            cmd = ["uv", "run", "ruff", "check", "--fix", "--unsafe-fixes", str(path)]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _stdout, stderr = await process.communicate()

        if process.returncode == 0:
            print(f"Formatted {path}")
        else:
            print(f"Warning: Failed to format {path}: {stderr.decode()}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Failed to format {path}: {e}", file=sys.stderr)


def _rename_numbered_classes(file_path: Path) -> None:
    """Rename numbered classes to more meaningful names."""
    rename_map = {
        "AgentOutgoingMessage1": "AgentRequestMessage",
        "AgentOutgoingMessage2": "AgentResponseMessage",
        "AgentOutgoingMessage3": "AgentErrorMessage",
        "AgentOutgoingMessage4": "AgentNotificationMessage",
        "AvailableCommandInput1": "CommandInputHint",
        "ClientOutgoingMessage1": "ClientRequestMessage",
        "ClientOutgoingMessage2": "ClientResponseMessage",
        "ClientOutgoingMessage3": "ClientErrorMessage",
        "ClientOutgoingMessage4": "ClientNotificationMessage",
        "ContentBlock1": "TextContentBlock",
        "ContentBlock2": "ImageContentBlock",
        "ContentBlock3": "AudioContentBlock",
        "ContentBlock4": "ResourceContentBlock",
        "ContentBlock5": "EmbeddedResourceContentBlock",
        "McpServer1": "HttpMcpServer",
        "McpServer2": "SseMcpServer",
        "McpServer3": "StdioMcpServer",
        "RequestPermissionOutcome1": "DeniedOutcome",
        "RequestPermissionOutcome2": "AllowedOutcome",
        "SessionUpdate1": "UserMessageChunk",
        "SessionUpdate2": "AgentMessageChunk",
        "SessionUpdate3": "AgentThoughtChunk",
        "SessionUpdate4": "ToolCallStart",
        "SessionUpdate5": "ToolCallProgress",
        "SessionUpdate6": "AgentPlan",
        "SessionUpdate7": "AvailableCommandsUpdate",
        "SessionUpdate8": "CurrentModeUpdate",
        "ToolCallContent1": "ContentToolCallContent",
        "ToolCallContent2": "FileEditToolCallContent",
        "ToolCallContent3": "TerminalToolCallContent",
    }

    content = file_path.read_text(encoding="utf-8")

    # Replace class definitions and all references
    # Sort by length descending to avoid partial matches (e.g., avoid replacing
    # "SessionUpdate1" before "SessionUpdate10")
    for old_name, new_name in sorted(
        rename_map.items(), key=lambda x: len(x[0]), reverse=True
    ):
        # Replace class definition
        content = content.replace(f"class {old_name}(", f"class {new_name}(")

        # Replace type annotations and references
        # Handle standalone usage
        content = content.replace(f"{old_name} |", f"{new_name} |")
        content = content.replace(f"| {old_name}", f"| {new_name}")
        content = content.replace(f": {old_name}", f": {new_name}")
        content = content.replace(f"[{old_name}]", f"[{new_name}]")
        content = content.replace(f"List[{old_name}]", f"List[{new_name}]")
        content = content.replace(f"Optional[{old_name}]", f"Optional[{new_name}]")
        content = content.replace(f"Union[{old_name}", f"Union[{new_name}")
        content = content.replace(f", {old_name}]", f", {new_name}]")
        content = content.replace(f"({old_name},", f"({new_name},")
        content = content.replace(f"({old_name})", f"({new_name})")

        # Handle line beginnings and isolated references
        content = content.replace(f"\n        {old_name}", f"\n        {new_name}")
        content = content.replace(
            f"\n            {old_name}", f"\n            {new_name}"
        )
        content = content.replace(f" {old_name}(", f" {new_name}(")
        content = content.replace(f"={old_name}(", f"={new_name}(")
        content = content.replace(f"return {old_name}(", f"return {new_name}(")

        # Handle imports and root model references
        content = content.replace(f"[{old_name}])", f"[{new_name}])")
        content = content.replace(
            f"root: Annotated[\n        {old_name},",
            f"root: Annotated[\n        {new_name},",
        )

        # Handle type union patterns in annotated types
        content = content.replace(
            f"Annotated[\n        {old_name}", f"Annotated[\n        {new_name}"
        )

    file_path.write_text(content)


if __name__ == "__main__":
    asyncio.run(main())
