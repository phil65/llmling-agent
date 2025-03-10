# Toolset Configuration

Toolsets are collections of tools that can be dynamically loaded from various sources. They provide a way to organize and manage groups of related tools.

## OpenAPI Toolset
Creates tools from an OpenAPI/Swagger specification.

```yaml
agents:
  api-agent:
    toolsets:
      - type: "openapi"  # toolset discriminator
        namespace: "github"  # optional prefix for tool names
        spec: "https://api.github.com/openapi.json"  # URL or path to spec
        base_url: "https://api.github.com"  # optional base URL override
```

## Entry Points Toolset
Loads tools registered through Python entry points.

```yaml
agents:
  plugin-agent:
    toolsets:
      - type: "entry_points"
        namespace: "my_plugins"  # optional namespace prefix
        module: "my_package"  # Python module containing entry points
```

## Custom Toolset
Creates tools from a custom Python class implementation.

```yaml
agents:
  custom-agent:
    toolsets:
      - type: "custom"
        namespace: "aws"  # optional namespace prefix
        import_path: "myapp.toolsets.AWSToolSet"  # path to toolset class
```

## Toolset Class Example
Example of implementing a custom toolset:

```python
from llmling.tools.toolsets import ToolSet
from llmling_agent.tools.base import Tool

class MyToolSet(ToolSet):
    """Custom toolset implementation."""

    def __init__(self, namespace: str | None = None):
        super().__init__(namespace=namespace)

    async def get_tools(self) -> list[Tool]:
        """Return list of tools in this set."""
        tools = []
        # Create and return tools
        return tools
```

## Configuration Notes

- The `type` field serves as discriminator for toolset types
- Namespaces help prevent tool name collisions
- Toolsets are loaded when the agent initializes
- Custom toolsets must inherit from `ToolSet`
- OpenAPI specs can be local files or URLs
- Entry points should use the format:
  ```
  [my_package.tools]
  tool_name = path.to:function
  ```
- Tools from toolsets can be filtered through agent capabilities
- Environment variables can be referenced in configuration
