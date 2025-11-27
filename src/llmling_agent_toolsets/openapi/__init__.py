"""OpenAPI toolset implementation with HTTP-aware reference resolution.

This package provides tools for loading OpenAPI specs and creating callable tools
from their operations. It includes a custom resolver that handles external HTTP
references in multi-file OpenAPI specifications.
"""

from __future__ import annotations

from llmling_agent_toolsets.openapi.loader import load_openapi_spec, parse_operations
from llmling_agent_toolsets.openapi.resolver import (
    OpenAPIResolver,
    resolve_openapi_refs,
)
from llmling_agent_toolsets.openapi.toolset import FORMAT_MAP, OpenAPITools


__all__ = [
    "FORMAT_MAP",
    "OpenAPIResolver",
    "OpenAPITools",
    "load_openapi_spec",
    "parse_operations",
    "resolve_openapi_refs",
]
