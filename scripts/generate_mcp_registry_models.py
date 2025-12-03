#!/usr/bin/env python3
"""Generate Pydantic model code for MCP registry servers.

This script fetches server metadata from the official MCP registry and generates
typed configuration classes for each server, including:
- Environment variable models with validation
- Header models for authenticated endpoints
- Pre-configured server classes with defaults

Usage:
    uv run scripts/generate_mcp_registry_models.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import keyword
from pathlib import Path
import re
import sys
from typing import Any

import httpx


REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"
OUTPUT_DIR = Path(__file__).parent.parent / "src" / "llmling_agent_config" / "mcp_registry"


@dataclass
class EnvVarDef:
    """Definition of an environment variable."""

    name: str
    description: str | None = None
    is_required: bool = False
    is_secret: bool = False
    default: str | None = None
    format: str = "string"


@dataclass
class HeaderDef:
    """Definition of an HTTP header."""

    name: str
    description: str | None = None
    is_secret: bool = False
    value: str | None = None


@dataclass
class RemoteDef:
    """Definition of a remote endpoint."""

    type: str  # sse, streamable-http
    url: str
    headers: list[HeaderDef] = field(default_factory=list)


@dataclass
class PackageDef:
    """Definition of a package."""

    registry_type: str  # npm, pypi, oci
    identifier: str
    version: str | None = None
    runtime_hint: str | None = None
    transport_type: str = "stdio"
    env_vars: list[EnvVarDef] = field(default_factory=list)


@dataclass
class ServerDef:
    """Parsed server definition."""

    name: str
    description: str
    version: str
    packages: list[PackageDef] = field(default_factory=list)
    remotes: list[RemoteDef] = field(default_factory=list)


def sanitize_name(name: str) -> str:
    """Convert server name to valid Python identifier."""
    # ai.exa/exa -> ExaExa
    parts = re.split(r"[./\-_]", name)
    # Filter out common prefixes
    parts = [p for p in parts if p.lower() not in ("ai", "com", "io", "org", "mcp")]
    result = "".join(p.capitalize() for p in parts if p)
    if not result:
        result = "".join(p.capitalize() for p in re.split(r"[./\-_]", name) if p)
    # Ensure it's a valid identifier
    if keyword.iskeyword(result) or not result.isidentifier():
        result = f"MCP{result}"
    return result


def sanitize_env_var_name(name: str) -> str:
    """Ensure env var name is a valid Python identifier."""
    # Replace hyphens with underscores
    result = name.replace("-", "_")
    # Strip leading underscores (Pydantic doesn't allow them for fields)
    result = result.lstrip("_")
    if not result:
        result = "var"
    if keyword.iskeyword(result) or not result.isidentifier():
        result = f"var_{result}"
    return result


def parse_env_var(data: dict[str, Any]) -> EnvVarDef:
    """Parse environment variable from registry data."""
    return EnvVarDef(
        name=data.get("name", ""),
        description=data.get("description"),
        is_required=data.get("isRequired", False),
        is_secret=data.get("isSecret", False),
        default=data.get("default"),
        format=data.get("format", "string"),
    )


def parse_header(data: dict[str, Any]) -> HeaderDef:
    """Parse header from registry data."""
    return HeaderDef(
        name=data.get("name", ""),
        description=data.get("description"),
        is_secret=data.get("isSecret", False),
        value=data.get("value"),
    )


def parse_server(entry: dict[str, Any]) -> ServerDef | None:
    """Parse server entry from registry response."""
    server = entry.get("server", {})
    meta = entry.get("_meta", {})

    # Only process latest versions
    official_meta = meta.get("io.modelcontextprotocol.registry/official", {})
    if not official_meta.get("isLatest", False):
        return None

    name = server.get("name", "")
    if not name:
        return None

    packages: list[PackageDef] = []
    for pkg_data in server.get("packages", []):
        env_vars = [parse_env_var(ev) for ev in pkg_data.get("environmentVariables", [])]
        transport = pkg_data.get("transport", {})
        packages.append(
            PackageDef(
                registry_type=pkg_data.get("registryType", ""),
                identifier=pkg_data.get("identifier", ""),
                version=pkg_data.get("version"),
                runtime_hint=pkg_data.get("runtimeHint"),
                transport_type=transport.get("type", "stdio"),
                env_vars=env_vars,
            )
        )

    remotes: list[RemoteDef] = []
    for remote_data in server.get("remotes", []):
        headers = [parse_header(h) for h in remote_data.get("headers", [])]
        remotes.append(
            RemoteDef(
                type=remote_data.get("type", ""),
                url=remote_data.get("url", ""),
                headers=headers,
            )
        )

    return ServerDef(
        name=name,
        description=server.get("description", ""),
        version=server.get("version", ""),
        packages=packages,
        remotes=remotes,
    )


def get_env_vars_for_server(server: ServerDef) -> dict[str, EnvVarDef]:
    """Collect all env vars from all packages."""
    all_env_vars: dict[str, EnvVarDef] = {}
    for pkg in server.packages:
        for ev in pkg.env_vars:
            if ev.name and ev.name not in all_env_vars:
                all_env_vars[ev.name] = ev
    return all_env_vars


def get_headers_for_server(server: ServerDef) -> dict[str, HeaderDef]:
    """Collect all headers from all remotes."""
    all_headers: dict[str, HeaderDef] = {}
    for remote in server.remotes:
        for header in remote.headers:
            if header.name and header.name not in all_headers:
                all_headers[header.name] = header
    return all_headers


def generate_env_field(env_var: EnvVarDef) -> str:
    """Generate a field definition for an environment variable."""
    field_name = sanitize_env_var_name(env_var.name)
    field_parts = []

    # Determine type and default
    if env_var.is_required and env_var.default is None:
        type_hint = "str"
        field_parts.append("...")
    elif env_var.default is not None:
        type_hint = "str"
        escaped_default = env_var.default.replace('"', '\\"')
        field_parts.append(f'default="{escaped_default}"')
    else:
        type_hint = "str | None"
        field_parts.append("default=None")

    # Add description
    if env_var.description:
        escaped_desc = env_var.description.replace('"', '\\"')
        field_parts.append(f'description="{escaped_desc}"')

    return f"    {field_name}: {type_hint} = Field({', '.join(field_parts)})"


def generate_header_field(header: HeaderDef) -> str:
    """Generate a field definition for a header."""
    # Convert header name to valid Python identifier
    field_name = header.name.lower().replace("-", "_")
    if keyword.iskeyword(field_name) or not field_name.isidentifier():
        field_name = f"header_{field_name}"

    field_parts = []

    if header.value:
        # Has a template value as default
        escaped_value = header.value.replace('"', '\\"')
        field_parts.append(f'default="{escaped_value}"')
        type_hint = "str"
    else:
        # Required
        field_parts.append("...")
        type_hint = "str"

    if header.description:
        escaped_desc = header.description.replace('"', '\\"')
        field_parts.append(f'description="{escaped_desc}"')

    return f"    {field_name}: {type_hint} = Field({', '.join(field_parts)})"


def generate_server_config_code(server: ServerDef, class_name: str) -> str:
    """Generate the main server config class."""
    # Determine best transport
    base_class: str
    extra_fields: list[str] = []

    # Generate unique type discriminator for this registry server
    type_value = f"registry:{server.name}"
    extra_fields.append(f'    type: Literal["{type_value}"] = Field("{type_value}", init=False)')
    extra_fields.append("")

    # Prefer remotes (public endpoints)
    if server.remotes:
        remote = server.remotes[0]
        if remote.type == "sse":
            base_class = "SSEMCPServerConfig"
            extra_fields.append(f'    url: HttpUrl = Field(default=HttpUrl("{remote.url}"))')
        else:  # streamable-http
            base_class = "StreamableHTTPMCPServerConfig"
            extra_fields.append(f'    url: HttpUrl = Field(default=HttpUrl("{remote.url}"))')
    elif server.packages:
        # Fall back to package-based stdio
        base_class = "StdioMCPServerConfig"
        pkg = server.packages[0]
        if pkg.runtime_hint and pkg.identifier:
            extra_fields.append(f'    command: str = Field(default="{pkg.runtime_hint}")')
            args = [pkg.identifier]
            if pkg.version:
                args[0] = f"{pkg.identifier}@{pkg.version}"
            extra_fields.append(f"    args: list[str] = Field(default={args!r})")
    else:
        base_class = "BaseMCPServerConfig"

    # Escape description for docstring
    desc = server.description.replace('"""', '\\"\\"\\"').replace("\n", " ")
    if len(desc) > 100:
        desc = desc[:97] + "..."

    # Collect env vars and headers
    env_vars = get_env_vars_for_server(server)
    headers = get_headers_for_server(server)

    lines = [
        f"class {class_name}({base_class}):",
        f'    """{desc}',
        "",
        f"    Registry name: {server.name}",
        f"    Version: {server.version}",
        '    """',
        "",
    ]

    # Add name field with default
    safe_name = server.name.replace('"', '\\"')
    lines.append(f'    name: str | None = Field(default="{safe_name}")')
    lines.append("")

    # Add extra fields (type, url, command/args)
    for field_line in extra_fields:
        lines.append(field_line)
    if extra_fields:
        lines.append("")

    # Add env var fields
    if env_vars:
        lines.append("    # Environment variables")
        for ev in env_vars.values():
            lines.append(generate_env_field(ev))
        lines.append("")

    # Add header fields
    if headers:
        lines.append("    # Headers")
        for h in headers.values():
            lines.append(generate_header_field(h))
        lines.append("")

    # Add model_post_init to merge fields into env/headers dicts
    if env_vars or headers:
        lines.append("    def model_post_init(self, __context: Any) -> None:")
        lines.append('        """Merge typed fields into env/headers dicts."""')

        if env_vars:
            lines.append("        env_updates = {")
            for ev in env_vars.values():
                field_name = sanitize_env_var_name(ev.name)
                lines.append(f'            "{ev.name}": self.{field_name},')
            lines.append("        }")
            lines.append(
                "        env_updates = {k: v for k, v in env_updates.items() if v is not None}"
            )
            lines.append("        if env_updates:")
            lines.append("            self.env = {**(self.env or {}), **env_updates}")

        if headers:
            lines.append("        header_updates = {")
            for h in headers.values():
                field_name = h.name.lower().replace("-", "_")
                if keyword.iskeyword(field_name) or not field_name.isidentifier():
                    field_name = f"header_{field_name}"
                lines.append(f'            "{h.name}": self.{field_name},')
            lines.append("        }")
            lines.append(
                "        header_updates = {k: v for k, v in header_updates.items() if v is not None}"
            )
            lines.append("        if header_updates:")
            lines.append("            self.headers = {**(self.headers or {}), **header_updates}")

        lines.append("")

    return "\n".join(lines)


def generate_server_module(server: ServerDef) -> tuple[str, str]:
    """Generate complete module code for a server.

    Returns tuple of (module_name, code).
    """
    base_name = sanitize_name(server.name)
    module_name = re.sub(r"(?<!^)(?=[A-Z])", "_", base_name).lower()
    config_class_name = f"{base_name}Config"

    imports = [
        '"""Auto-generated MCP server configuration.',
        "",
        f"Server: {server.name}",
        f"Description: {server.description[:80]}..."
        if len(server.description) > 80
        else f"Description: {server.description}",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any, Literal",
        "",
        "from pydantic import Field, HttpUrl",
        "",
        "from llmling_agent_config.mcp_server import (",
        "    BaseMCPServerConfig,",
        "    SSEMCPServerConfig,",
        "    StdioMCPServerConfig,",
        "    StreamableHTTPMCPServerConfig,",
        ")",
        "",
        "",
    ]

    parts = ["\n".join(imports)]

    # Generate main config
    config_code = generate_server_config_code(server, config_class_name)
    parts.append(config_code)

    # Add __all__
    exports = [config_class_name]
    all_line = f"\n__all__ = {exports!r}\n"
    parts.insert(1, all_line)

    return module_name, "\n".join(parts)


def generate_init_file(servers: list[tuple[str, str, ServerDef]]) -> str:
    """Generate __init__.py that exports all servers."""
    lines = [
        '"""Auto-generated MCP registry server configurations.',
        "",
        "This module provides pre-configured MCP server classes for servers",
        "registered in the official MCP registry.",
        "",
        "Usage:",
        "    from llmling_agent_config.mcp_registry import ExaExaConfig",
        "",
        "    # Use with default settings",
        "    server = ExaExaConfig()",
        "",
        "    # Or customize",
        "    server = ExaExaConfig(timeout=120.0)",
        "",
        "    # Or look up by registry name",
        "    from llmling_agent_config.mcp_registry import get_registry_server",
        "    server = get_registry_server('ai.exa/exa')",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Union",
        "",
        "from llmling_agent_config.mcp_registry._lookup import get_registry_server",
        "",
    ]

    # Collect imports and exports
    imports: list[str] = []
    all_exports: list[str] = []
    config_classes: list[str] = []
    registry_map_entries: list[str] = []

    for module_name, _, server in sorted(servers, key=lambda x: x[0]):
        base_name = sanitize_name(server.name)
        config_class = f"{base_name}Config"
        imports.append(
            f"from llmling_agent_config.mcp_registry.{module_name} import {config_class}"
        )
        all_exports.append(config_class)
        config_classes.append(config_class)
        registry_map_entries.append(f'    "{server.name}": {config_class},')

    lines.extend(imports)
    lines.append("")
    lines.append("")

    # Add registry mapping
    lines.append("# Mapping from registry name to config class")
    lines.append("REGISTRY_SERVERS: dict[str, type] = {")
    lines.extend(registry_map_entries)
    lines.append("}")
    lines.append("")

    # Add Union type of all registry configs
    if config_classes:
        lines.append("")
        lines.append(
            "# Union of all registry server configs (add to MCPServerConfig with discriminator)"
        )
        lines.append("RegistryServerConfig = Union[")
        for cls in config_classes[:-1]:
            lines.append(f"    {cls},")
        lines.append(f"    {config_classes[-1]},")
        lines.append("]")
        lines.append("")
        all_exports.append("RegistryServerConfig")

    # Add __all__
    all_exports.append("REGISTRY_SERVERS")
    all_exports.append("get_registry_server")
    lines.append(f"__all__ = {all_exports!r}")
    lines.append("")

    return "\n".join(lines)


async def fetch_registry() -> list[dict[str, Any]]:
    """Fetch all servers from MCP registry, handling pagination."""
    all_servers: list[dict[str, Any]] = []
    cursor: str | None = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            url = REGISTRY_URL
            if cursor:
                url = f"{REGISTRY_URL}?cursor={cursor}"

            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            servers = data.get("servers", [])
            all_servers.extend(servers)

            # Check for next page
            metadata = data.get("metadata", {})
            next_cursor = metadata.get("nextCursor")

            if not next_cursor or not servers:
                break

            cursor = next_cursor
            print(f"  Fetched {len(all_servers)} servers so far...")

    return all_servers


async def main() -> int:
    """Main entry point."""
    print(f"Fetching servers from {REGISTRY_URL}...")
    entries = await fetch_registry()
    print(f"Found {len(entries)} server entries")

    # Parse servers
    servers: list[ServerDef] = []
    for entry in entries:
        server = parse_server(entry)
        if server:
            servers.append(server)

    print(f"Parsed {len(servers)} latest server versions")

    # Filter to servers with remotes or packages with env vars
    interesting_servers = [
        s for s in servers if s.remotes or any(pkg.env_vars for pkg in s.packages)
    ]
    print(f"Generating models for {len(interesting_servers)} servers with remotes or env vars")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate modules
    generated: list[tuple[str, str, ServerDef]] = []
    for server in interesting_servers:
        try:
            module_name, code = generate_server_module(server)
            output_path = OUTPUT_DIR / f"{module_name}.py"
            output_path.write_text(code)
            generated.append((module_name, code, server))
            print(f"  Generated {module_name}.py for {server.name}")
        except Exception as e:
            print(f"  Failed to generate for {server.name}: {e}")

    # Generate __init__.py
    init_code = generate_init_file(generated)
    init_path = OUTPUT_DIR / "__init__.py"
    init_path.write_text(init_code)
    print(f"\nGenerated __init__.py with {len(generated)} server configs")

    # Create py.typed marker
    (OUTPUT_DIR / "py.typed").touch()

    print(f"\nOutput written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
