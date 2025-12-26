"""Debug toolset for agent self-introspection and runtime debugging."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
from typing import Any, Literal

from pydantic_ai import RunContext  # noqa: TC002

from agentpool.agents.context import AgentContext  # noqa: TC001
from agentpool.resource_providers import StaticResourceProvider


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass
class LogRecord:
    """A captured log record."""

    level: str
    logger: str
    message: str
    timestamp: str
    extra: dict[str, Any] = field(default_factory=dict)


class MemoryLogHandler(logging.Handler):
    """A logging handler that stores records in memory for later retrieval."""

    def __init__(self, max_records: int = 1000) -> None:
        super().__init__()
        self.max_records = max_records
        self.records: deque[LogRecord] = deque(maxlen=max_records)
        self.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """Store the log record."""
        log_record = LogRecord(
            level=record.levelname,
            logger=record.name,
            message=self.format(record),
            timestamp=self.formatter.formatTime(record) if self.formatter else "",
            extra={k: v for k, v in record.__dict__.items() if k not in logging.LogRecord.__dict__},
        )
        self.records.append(log_record)

    def get_records(
        self,
        level: LogLevel | None = None,
        logger_filter: str | None = None,
        limit: int | None = None,
    ) -> list[LogRecord]:
        """Get filtered log records.

        Args:
            level: Minimum log level to include
            logger_filter: Only include loggers containing this string
            limit: Maximum number of records to return (newest first)

        Returns:
            List of matching log records
        """
        level_priority = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        min_level = level_priority.get(level, 0) if level else 0

        filtered = []
        for record in reversed(self.records):
            if level_priority.get(record.level, 0) < min_level:
                continue
            if logger_filter and logger_filter not in record.logger:
                continue
            filtered.append(record)
            if limit and len(filtered) >= limit:
                break

        return filtered

    def clear(self) -> None:
        """Clear all stored records."""
        self.records.clear()


# Global memory handler instance
_memory_handler: MemoryLogHandler | None = None


def get_memory_handler() -> MemoryLogHandler:
    """Get or create the global memory log handler."""
    global _memory_handler  # noqa: PLW0603
    if _memory_handler is None:
        _memory_handler = MemoryLogHandler()
        # Attach to root logger to capture everything
        logging.getLogger().addHandler(_memory_handler)
    return _memory_handler


def install_memory_handler(max_records: int = 1000) -> MemoryLogHandler:
    """Install the memory handler if not already installed.

    Args:
        max_records: Maximum number of records to keep in memory

    Returns:
        The memory handler instance
    """
    global _memory_handler  # noqa: PLW0603
    if _memory_handler is None:
        _memory_handler = MemoryLogHandler(max_records=max_records)
        logging.getLogger().addHandler(_memory_handler)
    return _memory_handler


# =============================================================================
# Introspection Tool
# =============================================================================

INTROSPECTION_USAGE = """
Execute Python code with full access to your runtime context.

Available in namespace:
- ctx: AgentContext (ctx.agent, ctx.pool, ctx.config, ctx.definition, etc.)
- run_ctx: pydantic-ai RunContext (current run state)
- me: Shortcut for ctx.agent (your Agent instance)

You can inspect yourself, the pool, other agents, your tools, and more.
Write an async main() function that returns the result.

Example - inspect your own tools:
```python
async def main():
    tools = me.tools.list_tools()
    return [t.name for t in tools]
```

Example - check pool state:
```python
async def main():
    if ctx.pool:
        agents = list(ctx.pool.agents.keys())
        return f"Agents in pool: {agents}"
    return "No pool available"
```

Example - explore with dir():
```python
async def main():
    return dir(ctx)
```
"""


async def execute_introspection(ctx: AgentContext, run_ctx: RunContext[Any], code: str) -> str:  # noqa: D417
    """Execute Python code with access to your own runtime context.

    This is a debugging/development tool that gives you full access to
    inspect and interact with your runtime environment.

    Args:
        code: Python code with async main() function to execute

    Returns:
        Result of execution or error message
    """
    # Build namespace with runtime context
    namespace: dict[str, Any] = {"ctx": ctx, "run_ctx": run_ctx, "me": ctx.agent}
    try:
        exec(code, namespace)
        if "main" not in namespace:
            return "Error: Code must define an async main() function"
        result = await namespace["main"]()
        return str(result) if result is not None else "Code executed successfully (no return value)"
    except Exception as e:  # noqa: BLE001
        return f"Error executing code: {type(e).__name__}: {e}"


# =============================================================================
# Log Tools
# =============================================================================


async def get_logs(
    level: LogLevel = "INFO",
    logger_filter: str | None = None,
    limit: int = 50,
) -> str:
    """Get recent log entries from memory.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_filter: Only show logs from loggers containing this string
        limit: Maximum number of log entries to return

    Returns:
        Formatted log entries
    """
    handler = get_memory_handler()
    records = handler.get_records(level=level, logger_filter=logger_filter, limit=limit)

    if not records:
        return "No log entries found matching criteria"

    lines = [f"=== {len(records)} log entries (newest first) ===\n"]
    for record in records:
        lines.append(record.message)  # noqa: PERF401

    return "\n".join(lines)


async def clear_logs() -> str:
    """Clear all captured log entries from memory.

    Returns:
        Confirmation message
    """
    handler = get_memory_handler()
    count = len(handler.records)
    handler.clear()
    return f"Cleared {count} log entries"


# =============================================================================
# Path Tools
# =============================================================================


async def get_platform_paths() -> str:
    """Get platform-specific paths for agentpool.

    Returns:
        Dictionary of platform paths
    """
    import platformdirs

    paths = {
        "config": platformdirs.user_config_dir("agentpool"),
        "data": platformdirs.user_data_dir("agentpool"),
        "cache": platformdirs.user_cache_dir("agentpool"),
        "logs": platformdirs.user_log_dir("agentpool"),
        "state": platformdirs.user_state_dir("agentpool"),
    }

    lines = ["Platform paths for agentpool:", ""]
    for name, path in paths.items():
        lines.append(f"  {name}: {path}")

    return "\n".join(lines)


# =============================================================================
# State Inspection Tools
# =============================================================================


async def dump_agent_state(ctx: AgentContext) -> str:
    """Dump current agent's state and configuration.

    Returns:
        Formatted agent state
    """
    agent = ctx.agent

    lines = [
        f"Agent: {agent.name}",
        f"Model: {agent.model_name or 'unknown'}",
    ]

    # sys_prompts is available on Agent and ClaudeCodeAgent, not ACPAgent
    from agentpool import Agent
    from agentpool.agents.claude_code_agent import ClaudeCodeAgent

    if isinstance(agent, (Agent, ClaudeCodeAgent)):
        formatted_prompt = await agent.sys_prompts.format_system_prompt(agent)
        lines.append(f"System prompt length: {len(formatted_prompt)} chars")

    lines.extend(["", "Tools:"])

    tools_status = await agent.tools.list_tools()
    for tool_name, enabled in tools_status.items():
        status = "enabled" if enabled else "disabled"
        lines.append(f"  - {tool_name} [{status}]")

    if ctx.pool:
        lines.extend([
            "",
            f"Pool agents: {len(ctx.pool.all_agents)}",
            f"Other agents: {[n for n in ctx.pool.all_agents if n != agent.name]}",
        ])

    return "\n".join(lines)


async def dump_pool_state(ctx: AgentContext) -> str:
    """Dump the agent pool state.

    Returns:
        Formatted pool state
    """
    if not ctx.pool:
        return "No agent pool available"

    pool = ctx.pool
    lines = [
        f"Agents: {len(pool.all_agents)}",
        f"Teams: {len(pool.teams)}",
        "",
    ]

    if pool.all_agents:
        lines.append("Agents:")
        for name, agent in pool.all_agents.items():
            agent_type = type(agent).__name__
            tools_status = await agent.tools.list_tools()
            lines.append(f"  {name} ({agent_type}):")
            lines.append(f"    model: {agent.model_name or 'unknown'}")
            lines.append(f"    tools: {len(tools_status)}")

    if pool.teams:
        lines.extend(["", "Teams:"])
        for name, team in pool.teams.items():
            team_type = type(team).__name__
            lines.append(f"  {name} ({team_type})")

    return "\n".join(lines)


# =============================================================================
# Toolset Class
# =============================================================================


class DebugTools(StaticResourceProvider):
    """Debug and introspection tools for agent development.

    Provides tools for:
    - Self-introspection via code execution with runtime context access
    - Log inspection and management
    - Platform path discovery
    - Agent and pool state inspection
    """

    def __init__(self, name: str = "debug", install_log_handler: bool = True) -> None:
        """Initialize debug tools.

        Args:
            name: Toolset name/namespace
            install_log_handler: Whether to install the memory log handler
        """
        super().__init__(name=name)

        if install_log_handler:
            install_memory_handler()

        introspection_desc = (execute_introspection.__doc__ or "") + "\n\n" + INTROSPECTION_USAGE
        self._tools = [
            # Introspection
            self.create_tool(
                execute_introspection,
                category="other",
                description_override=introspection_desc,
            ),
            # Logs
            self.create_tool(get_logs, category="other", read_only=True, idempotent=True),
            self.create_tool(clear_logs, category="other"),
            # Paths
            self.create_tool(get_platform_paths, category="other", read_only=True, idempotent=True),
            # State inspection
            self.create_tool(dump_agent_state, category="other", read_only=True, idempotent=True),
            self.create_tool(dump_pool_state, category="other", read_only=True, idempotent=True),
        ]
