"""Configuration models for AG-UI agents."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from llmling_agent_config.nodes import BaseAgentConfig
from llmling_agent_config.tools import ToolConfig  # noqa: TC001


class AGUIAgentConfig(BaseAgentConfig):
    """Configuration for AG-UI protocol agents.

    AG-UI agents connect to remote HTTP endpoints that implement the AG-UI protocol,
    enabling integration of any AG-UI compatible server into the llmling-agent pool.

    Example:
        ```yaml
        agui_agents:
          remote_assistant:
            endpoint: http://localhost:8000/agent/run
            timeout: 30.0
            headers:
              X-API-Key: ${API_KEY}
            tools:
              - import_path: mymodule.my_tool_function

          managed_agent:
            endpoint: http://localhost:8765/agent/run
            startup_command: "uv run ag-ui-server config.yml"
            startup_delay: 3.0
            tools:
              - import_path: package.tool_one
              - import_path: package.tool_two
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={"title": "AG-UI Agent Configuration", "x-icon": "mdi:api"}
    )

    endpoint: str = Field(
        ...,
        examples=["http://localhost:8000/agent/run", "https://api.example.com/v1/agent/run"],
    )
    """HTTP endpoint for the AG-UI agent server."""

    timeout: float = Field(default=60.0, ge=0.1, examples=[30.0, 60.0, 120.0])
    """Request timeout in seconds."""

    headers: dict[str, str] = Field(default_factory=dict)
    """Additional HTTP headers to send with requests.

    Useful for authentication or custom routing.
    Environment variables can be used: ${VAR_NAME}
    """

    startup_command: str | None = Field(
        default=None,
        examples=[
            "uv run ag-ui-server config.yml",
            "python -m my_agui_server --port 8000",
        ],
    )
    """Optional shell command to start the AG-UI server automatically.

    When provided, the agent will spawn this command on context entry
    and terminate it on exit. Useful for testing or self-contained deployments.
    """

    startup_delay: float = Field(default=2.0, ge=0.0, examples=[1.0, 2.0, 5.0])
    """Seconds to wait after starting server before connecting.

    Only relevant when startup_command is provided.
    """

    tools: list[ToolConfig] = Field(default_factory=list)
    """Tools to expose to the remote agent for client-side execution.

    When the remote AG-UI agent requests a tool call, these tools are executed
    locally and the result is sent back. This enables human-in-the-loop workflows
    and local capability exposure to remote agents.
    """
