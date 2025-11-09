"""Execution environment configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from anyenv.code_execution.models import Language
from pydantic import Field, SecretStr
from schemez import Schema


if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from anyenv.code_execution.beam_provider import BeamExecutionEnvironment
    from anyenv.code_execution.daytona_provider import DaytonaExecutionEnvironment
    from anyenv.code_execution.docker_provider import DockerExecutionEnvironment
    from anyenv.code_execution.e2b_provider import E2bExecutionEnvironment
    from anyenv.code_execution.local_provider import LocalExecutionEnvironment
    from anyenv.code_execution.mcp_python_provider import McpPythonExecutionEnvironment
    from anyenv.code_execution.models import ServerInfo
    from anyenv.code_execution.subprocess_provider import SubprocessExecutionEnvironment


class BaseExecutionEnvironmentConfig(Schema):
    """Base execution environment configuration."""

    type: str = Field(init=False)
    """Execution environment type."""

    dependencies: list[str] | None = None
    """List of packages to install (pip for Python, npm for JS/TS)."""

    timeout: float = 30.0
    """Execution timeout in seconds."""


class LocalExecutionEnvironmentConfig(BaseExecutionEnvironmentConfig):
    """Local execution environment configuration.

    Executes code in the same process. Fastest option but offers no isolation.
    """

    type: Literal["local"] = Field("local", init=False)

    def get_provider(
        self, lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None
    ) -> LocalExecutionEnvironment:
        """Create local execution environment instance."""
        from anyenv.code_execution.local_provider import LocalExecutionEnvironment

        return LocalExecutionEnvironment(
            lifespan_handler=lifespan_handler,
            dependencies=self.dependencies,
            timeout=self.timeout,
        )


class SubprocessExecutionEnvironmentConfig(BaseExecutionEnvironmentConfig):
    """Subprocess execution environment configuration.

    Executes code in a separate Python process for basic isolation.
    """

    type: Literal["subprocess"] = Field("subprocess", init=False)

    executable: str | None = None
    """Python executable to use (if None, auto-detect based on language)."""

    language: Language = "python"
    """Programming language to use."""

    def get_provider(
        self, lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None
    ) -> SubprocessExecutionEnvironment:
        """Create subprocess execution environment instance."""
        from anyenv.code_execution.subprocess_provider import (
            SubprocessExecutionEnvironment,
        )

        return SubprocessExecutionEnvironment(
            lifespan_handler=lifespan_handler,
            dependencies=self.dependencies,
            executable=self.executable,
            timeout=self.timeout,
            language=self.language,
        )


class DockerExecutionEnvironmentConfig(BaseExecutionEnvironmentConfig):
    """Docker execution environment configuration.

    Executes code in Docker containers for strong isolation and reproducible environments.
    """

    type: Literal["docker"] = Field("docker", init=False)

    image: str = "python:3.13-slim"
    """Docker image to use."""

    language: Language = "python"
    """Programming language to use."""

    timeout: float = 60.0
    """Execution timeout in seconds."""

    def get_provider(
        self, lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None
    ) -> DockerExecutionEnvironment:
        """Create Docker execution environment instance."""
        from anyenv.code_execution.docker_provider import DockerExecutionEnvironment

        return DockerExecutionEnvironment(
            lifespan_handler=lifespan_handler,
            dependencies=self.dependencies,
            image=self.image,
            timeout=self.timeout,
            language=self.language,
        )


class E2bExecutionEnvironmentConfig(BaseExecutionEnvironmentConfig):
    """E2B execution environment configuration.

    Executes code in E2B sandboxes for secure, ephemeral execution environments.
    """

    type: Literal["e2b"] = Field("e2b", init=False)

    template: str | None = None
    """E2B template to use."""

    keep_alive: bool = False
    """Keep sandbox running after execution."""

    language: Language = "python"
    """Programming language to use."""

    timeout: float = 300.0
    """Execution timeout in seconds."""

    def get_provider(
        self, lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None
    ) -> E2bExecutionEnvironment:
        """Create E2B execution environment instance."""
        from anyenv.code_execution.e2b_provider import E2bExecutionEnvironment

        return E2bExecutionEnvironment(
            lifespan_handler=lifespan_handler,
            dependencies=self.dependencies,
            template=self.template,
            timeout=self.timeout,
            keep_alive=self.keep_alive,
            language=self.language,
        )


class BeamExecutionEnvironmentConfig(BaseExecutionEnvironmentConfig):
    """Beam execution environment configuration.

    Executes code in Beam cloud sandboxes for scalable, serverless execution environments.
    """

    type: Literal["beam"] = Field("beam", init=False)

    cpu: float | str = 1.0
    """CPU cores allocated to the container."""

    memory: int | str = 128
    """Memory allocated to the container in MiB."""

    keep_warm_seconds: int = 600
    """Seconds to keep sandbox alive, -1 for no timeout."""

    language: Language = "python"
    """Programming language to use."""

    timeout: float = 300.0
    """Execution timeout in seconds."""

    def get_provider(
        self, lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None
    ) -> BeamExecutionEnvironment:
        """Create Beam execution environment instance."""
        from anyenv.code_execution.beam_provider import BeamExecutionEnvironment

        return BeamExecutionEnvironment(
            lifespan_handler=lifespan_handler,
            dependencies=self.dependencies,
            cpu=self.cpu,
            memory=self.memory,
            keep_warm_seconds=self.keep_warm_seconds,
            timeout=self.timeout,
            language=self.language,
        )


class DaytonaExecutionEnvironmentConfig(BaseExecutionEnvironmentConfig):
    """Daytona execution environment configuration.

    Executes code in remote Daytona sandboxes for cloud-based development environments.
    """

    type: Literal["daytona"] = Field("daytona", init=False)

    api_url: str | None = None
    """Daytona API URL (optional, uses env vars if not provided)."""

    api_key: SecretStr | None = None
    """API key for authentication."""

    target: str | None = None
    """Target configuration."""

    image: str = "python:3.13-slim"
    """Container image."""

    keep_alive: bool = False
    """Keep sandbox running after execution."""

    timeout: float = 300.0
    """Execution timeout in seconds."""

    def get_provider(
        self, lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None
    ) -> DaytonaExecutionEnvironment:
        """Create Daytona execution environment instance."""
        from anyenv.code_execution.daytona_provider import DaytonaExecutionEnvironment

        api_key_str = self.api_key.get_secret_value() if self.api_key else None
        return DaytonaExecutionEnvironment(
            lifespan_handler=lifespan_handler,
            dependencies=self.dependencies,
            api_url=self.api_url,
            api_key=api_key_str,
            target=self.target,
            image=self.image,
            timeout=self.timeout,
            keep_alive=self.keep_alive,
        )


class McpPythonExecutionEnvironmentConfig(BaseExecutionEnvironmentConfig):
    """MCP Python execution environment configuration.

    Executes Python code with Model Context Protocol support for AI integrations.
    """

    type: Literal["mcp_python"] = Field("mcp_python", init=False)

    allow_networking: bool = True
    """Allow network access."""

    def get_provider(
        self, lifespan_handler: AbstractAsyncContextManager[ServerInfo] | None = None
    ) -> McpPythonExecutionEnvironment:
        """Create MCP Python execution environment instance."""
        from anyenv.code_execution.mcp_python_provider import (
            McpPythonExecutionEnvironment,
        )

        return McpPythonExecutionEnvironment(
            lifespan_handler=lifespan_handler,
            dependencies=self.dependencies,
            allow_networking=self.allow_networking,
            timeout=self.timeout,
        )


# Union type for all execution environment configurations
ExecutionEnvironmentConfig = Annotated[
    LocalExecutionEnvironmentConfig
    | SubprocessExecutionEnvironmentConfig
    | DockerExecutionEnvironmentConfig
    | E2bExecutionEnvironmentConfig
    | BeamExecutionEnvironmentConfig
    | DaytonaExecutionEnvironmentConfig
    | McpPythonExecutionEnvironmentConfig,
    Field(discriminator="type"),
]
