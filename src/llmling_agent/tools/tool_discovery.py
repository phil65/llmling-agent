"""Functions to discover available tools."""

from __future__ import annotations

import importlib
from inspect import getmembers, isclass
from typing import Literal


ToolSource = Literal["crewai", "langchain"]


def get_tools(
    source: ToolSource,
    include_descriptions: bool = False,
) -> dict[str, str | None]:
    """Get available tools from specified source.

    Args:
        source: Which tool source to query
        include_descriptions: Whether to include tool descriptions

    Returns:
        Dict mapping tool names to descriptions (or None if descriptions disabled)
    """
    tools = {}

    match source:
        case "crewai":
            import crewai_tools as module  # pyright: ignore

            # Look for all classes in single module
            for _, cls in getmembers(module, isclass):
                if hasattr(cls, "model_fields"):
                    fields = cls.model_fields
                    if "name" in fields and "description" in fields:
                        name = fields["name"].default
                        description = fields["description"].default
                        if name:
                            tools[name] = description

        case "langchain":
            # Need to import each tool module separately
            from langchain_community.tools import _module_lookup  # pyright: ignore

            for cls_name, module_path in _module_lookup.items():
                try:
                    module = importlib.import_module(module_path)
                    cls = getattr(module, cls_name)
                    if hasattr(cls, "model_fields"):
                        fields = cls.model_fields
                        if "name" in fields and "description" in fields:
                            name = fields["name"].default
                            description = fields["description"].default
                            if name:
                                tools[name] = description
                except (ImportError, AttributeError):
                    # Skip tools we can't import or that don't match our pattern
                    continue

    return tools


if __name__ == "__main__":
    tools = get_tools("crewai")
    print(tools)
    tools = get_tools("langchain")
    print(tools)
