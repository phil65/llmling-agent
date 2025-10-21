"""Capability tools."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import timedelta
import io
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from llmling import ToolError
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext

from llmling_agent.agent.context import AgentContext  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.utils.now import get_now


if TYPE_CHECKING:
    from llmling_agent.agent import AnyAgent
    from llmling_agent_config.mcp_server import MCPServerConfig

logger = get_logger(__name__)


async def delegate_to(  # noqa: D417
    ctx: AgentContext,
    agent_or_team_name: str,
    prompt: str,
) -> str:
    """Delegate a task to an agent or team.

    If an action requires you to delegate a task, this tool can be used to assign and
    execute a task. Instructions can be passed via the prompt parameter.

    Args:
        agent_or_team_name: The agent or team to delegate the task to
        prompt: Instructions for the agent or team to delegate to.

    Returns:
        The result of the delegated task
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "Agent needs to be in a pool to delegate tasks"
        raise ToolError(msg)

    if agent_or_team_name in ctx.pool.nodes:
        node = ctx.pool.nodes[agent_or_team_name]
        result = await node.run(prompt)
        return result.format(style="detailed", show_costs=True)

    msg = f"No agent or team found with name: {agent_or_team_name}"
    raise ToolError(msg)


async def list_available_agents(  # noqa: D417
    ctx: AgentContext,
    only_idle: bool = False,
    detailed: bool = False,
) -> str:
    """List all agents available in the current pool.

    Args:
        only_idle: If True, only returns agents that aren't currently busy.
                    Use this to find agents ready for immediate tasks.
        detailed: If True, additional info for each team is provided (e.g. description)

    Returns:
        List of agent names that you can use with delegate_to
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "Agent needs to be in a pool to list agents"
        raise ToolError(msg)

    agents = dict(ctx.pool.agents)
    if only_idle:
        agents = {name: agent for name, agent in agents.items() if not agent.is_busy()}
    if not detailed:
        return "\n".join(agents.keys())
    lines = []
    for name, agent in agents.items():
        lines.extend([
            f"name: {name}",
            f"description: {agent.description or 'No description'}",
            f"model: {agent.model_name}",
            "---",
        ])

    return "\n".join(lines) if lines else "No agents available"


async def list_available_teams(  # noqa: D417
    ctx: AgentContext,
    only_idle: bool = False,
    detailed: bool = False,
) -> str:
    """List all available teams in the pool.

    Args:
        only_idle: If True, only returns teams that aren't currently executing
        detailed: If True, additional info for each team is provided (e.g. description)

    Returns:
        Formatted list of teams with their descriptions and types
    """
    from llmling_agent import TeamRun

    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    teams = ctx.pool.teams
    if only_idle:
        teams = {name: team for name, team in teams.items() if not team.is_running}
    if not detailed:
        return "\n".join(teams.keys())
    lines = []
    for name, team in teams.items():
        lines.extend([
            f"name: {name}",
            f"description: {team.description or 'No description'}",
            f"type: {'sequential' if isinstance(team, TeamRun) else 'parallel'}",
            "members: " + ", ".join(a.name for a in team.agents),
            "---",
        ])

    return "\n".join(lines) if lines else "No teams available"


async def create_worker_agent[TDeps](
    ctx: AgentContext[TDeps],
    name: str,
    system_prompt: str,
    model: str | None = None,
) -> str:
    """Create a new agent and register it as a tool.

    The new agent will be available as a tool for delegating specific tasks.
    It inherits the current model unless overridden.
    """
    from llmling_agent import Agent

    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "Agent needs to be in a pool to list agents"
        raise ToolError(msg)

    model = model or ctx.agent.model_name
    agent = Agent[TDeps](name=name, model=model, system_prompt=system_prompt, context=ctx)
    assert ctx.agent
    tool_info = ctx.agent.register_worker(agent)
    return f"Created worker agent and registered as tool: {tool_info.name}"


async def spawn_delegate[TDeps](
    ctx: AgentContext[TDeps],
    task: str,
    system_prompt: str,
    model: str | None = None,
    connect_back: bool = False,
) -> str:
    """Spawn a temporary agent for a specific task.

    Creates an ephemeral agent that will execute the task and clean up automatically
    Optionally connects back to receive results.
    """
    from llmling_agent import Agent

    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    name = f"delegate_{uuid4().hex[:8]}"
    model = model or ctx.agent.model_name
    agent = Agent[TDeps](name=name, model=model, system_prompt=system_prompt, context=ctx)

    if connect_back:
        assert ctx.agent
        ctx.agent.connect_to(agent)
    try:
        await agent.run(task)
    except Exception as e:
        msg = f"Failed to spawn delegate {name}: {e}"
        raise ModelRetry(msg) from e
    return f"Spawned delegate {name} for task"


async def search_history(
    ctx: AgentContext,
    query: str | None = None,
    hours: int = 24,
    limit: int = 5,
) -> str:
    """Search conversation history."""
    from llmling_agent_storage.formatters import format_output

    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    provider = ctx.storage.get_history_provider()
    results = await provider.get_filtered_conversations(
        query=query,
        period=f"{hours}h",
        limit=limit,
    )
    return format_output(results)


async def show_statistics(
    ctx: AgentContext,
    group_by: Literal["agent", "model", "hour", "day"] = "model",
    hours: int = 24,
) -> str:
    """Show usage statistics for conversations."""
    from llmling_agent_storage.formatters import format_output
    from llmling_agent_storage.models import StatsFilters

    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    cutoff = get_now() - timedelta(hours=hours)
    filters = StatsFilters(cutoff=cutoff, group_by=group_by)

    provider = ctx.storage.get_history_provider()
    stats = await provider.get_conversation_stats(filters)

    return format_output(
        {
            "period": f"{hours}h",
            "group_by": group_by,
            "entries": [
                {
                    "name": key,
                    "messages": data["messages"],
                    "total_tokens": data["total_tokens"],
                    "models": sorted(data["models"]),
                }
                for key, data in stats.items()
            ],
        },
        output_format="text",
    )


async def execute_python(ctx: AgentContext, code: str) -> str:
    """Execute Python code directly."""
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    try:
        # Capture output
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__})
            return buf.getvalue() or "Code executed successfully"
    except Exception as e:  # noqa: BLE001
        return f"Error executing code: {e}"


async def execute_command(  # noqa: D417
    ctx: AgentContext,
    command: str,
    env: dict[str, str] | None = None,
    output_limit: int | None = None,
) -> str:
    """Execute a shell command.

    Args:
        command: Shell command to execute
        env: Environment variables to add to current environment
        output_limit: Maximum bytes of output to return
    """
    import os

    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    try:
        # Prepare environment
        proc_env = dict(os.environ)
        if env:
            proc_env.update(env)

        pipe = asyncio.subprocess.PIPE
        proc = await asyncio.create_subprocess_shell(
            command, stdout=pipe, stderr=pipe, env=proc_env
        )
        stdout, stderr = await proc.communicate()

        # Combine and decode output
        output = stdout.decode() or stderr.decode() or "Command completed"

        # Apply output limit if specified
        if output_limit and len(output.encode()) > output_limit:
            # Truncate from the end to keep most recent output
            truncated_output = output.encode()[-output_limit:].decode(errors="ignore")
            output = f"...[truncated]\n{truncated_output}"
    except Exception as e:  # noqa: BLE001
        return f"Error executing command: {e}"
    else:
        return output


async def add_agent(  # noqa: D417
    ctx: AgentContext,
    name: str,
    system_prompt: str,
    model: str | None = None,
    tools: list[str] | None = None,
    session: str | None = None,
    result_type: str | None = None,
) -> str:
    """Add a new agent to the pool.

    Args:
        name: Name for the new agent
        system_prompt: System prompt defining agent's role/behavior
        model: Optional model override (uses default if not specified)
        tools: Imort paths of the tools the agent should have, if any.
        session: Session ID to recover conversation state from
        result_type: Name of response type from manifest (for structured output)

    Returns:
        Confirmation message about the created agent
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    assert ctx.pool, "No agent pool available"
    try:
        agent: AnyAgent[Any, Any] = await ctx.pool.add_agent(
            name=name,
            system_prompt=system_prompt,
            model=model,
            tools=tools,
            result_type=result_type,
            session=session,
        )
    except ValueError as e:  # for wrong tool imports
        raise ModelRetry(message=f"Error creating agent: {e}") from None
    return f"Created agent {agent.name} using model {agent.model_name}"


async def add_team(  # noqa: D417
    ctx: AgentContext,
    agents: list[str],
    mode: Literal["sequential", "parallel"] = "sequential",
    name: str | None = None,
) -> str:
    """Create a team from existing agents.

    Args:
        agents: Names of agents to include in team
        mode: How the team should operate:
            - sequential: Agents process in sequence (pipeline)
            - parallel: Agents process simultaneously
        name: Optional name for the team
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    # Verify all agents exist
    for agent_name in agents:
        if agent_name not in ctx.pool.agents:
            msg = f"Agent not found: {agent_name}"
            raise ToolError(msg)
    if mode == "sequential":
        ctx.pool.create_team_run(agents, name=name)
    else:
        ctx.pool.create_team(agents, name=name)
    mode_str = "pipeline" if mode == "sequential" else "parallel"
    return f"Created {mode_str} team with agents: {', '.join(agents)}"


async def ask_agent(  # noqa: D417
    ctx: AgentContext,
    agent_name: str,
    message: str,
    *,
    model: str | None = None,
    store_history: bool = True,
) -> str:
    """Send a message to a specific agent and get their response.

    Args:
        agent_name: Name of the agent to interact with
        message: Message to send to the agent
        model: Optional temporary model override
        store_history: Whether to store this exchange in history

    Returns:
        The agent's response
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    assert ctx.pool, "No agent pool available"
    agent = ctx.pool.get_agent(agent_name)
    try:
        result = await agent.run(message, model=model, store_history=store_history)
    except Exception as e:
        msg = f"Failed to ask agent {agent_name}: {e}"
        raise ModelRetry(msg) from e
    return str(result.content)


async def connect_nodes(  # noqa: D417
    ctx: AgentContext,
    source: str,
    target: str,
    *,
    connection_type: Literal["run", "context", "forward"] = "run",
    priority: int = 0,
    delay_seconds: float | None = None,
    queued: bool = False,
    queue_strategy: Literal["concat", "latest", "buffer"] = "latest",
    wait_for_completion: bool = True,
    name: str | None = None,
) -> str:
    """Connect two nodes to enable message flow between them.

    Nodes can be agents, teams, or EventNodes.

    Args:
        source: Name of the source node
        target: Name of the target node
        connection_type: How messages should be handled:
            - run: Execute message as a new run in target
            - context: Add message as context to target
            - forward: Forward message to target's outbox
        priority: Task priority (lower = higher priority)
        delay_seconds: Optional delay before processing messages
        queued: Whether messages should be queued for manual processing
        queue_strategy: How to process queued messages:
            - concat: Combine all messages with newlines
            - latest: Use only the most recent message
            - buffer: Process all messages individually
        wait_for_completion: Whether to wait for target to complete
        name: Optional name for this connection

    Returns:
        Description of the created connection
    """
    from datetime import timedelta

    if isinstance(ctx, RunContext):
        ctx = ctx.deps
    if not ctx.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    # Get the nodes
    if source not in ctx.pool.nodes:
        msg = f"Source node not found: {source}"
        raise ToolError(msg)
    if target not in ctx.pool.nodes:
        msg = f"Target node not found: {target}"
        raise ToolError(msg)

    source_node = ctx.pool.nodes[source]
    target_node = ctx.pool.nodes[target]

    # Create the connection
    delay = timedelta(seconds=delay_seconds) if delay_seconds is not None else None
    _talk = source_node.connect_to(
        target_node,
        connection_type=connection_type,
        priority=priority,
        delay=delay,
        queued=queued,
        queue_strategy=queue_strategy,
        name=name,
    )
    source_node.connections.set_wait_state(target_node, wait=wait_for_completion)

    return (
        f"Created connection from {source} to {target} "
        f"(type={connection_type}, queued={queued}, "
        f"strategy={queue_strategy if queued else 'n/a'})"
    )


async def read_file(  # noqa: D417
    ctx: AgentContext,
    path: str,
    *,
    convert_to_markdown: bool = True,
    encoding: str = "utf-8",
    line: int | None = None,
    limit: int | None = None,
) -> str:
    """Read file content from local or remote path.

    Args:
        path: Path or URL to read
        convert_to_markdown: Whether to convert content to markdown
        encoding: Text encoding to use (default: utf-8)
        line: Optional line number to start reading from (1-based)
        limit: Optional maximum number of lines to read

    Returns:
        File content, optionally converted to markdown
    """
    from upathtools import read_path

    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        # First try to read raw content
        content = await read_path(path, encoding=encoding)

        # Convert to markdown if requested
        if convert_to_markdown and ctx.converter:
            try:
                content = await ctx.converter.convert_file(path)
            except Exception as e:  # noqa: BLE001
                msg = f"Failed to convert to markdown: {e}"
                logger.warning(msg)
                # Continue with raw content

        # Apply line filtering if requested
        if line is not None or limit is not None:
            lines = content.splitlines(keepends=True)
            start_idx = (line - 1) if line is not None else 0
            end_idx = start_idx + limit if limit is not None else len(lines)
            content = "".join(lines[start_idx:end_idx])

    except Exception as e:
        msg = f"Failed to read file {path}: {e}"
        raise ToolError(msg) from e
    else:
        return content


async def list_directory(
    path: str,
    *,
    pattern: str | None = None,
    recursive: bool = True,
    include_dirs: bool = False,
    exclude: list[str] | None = None,
    max_depth: int | None = None,
) -> str:
    """List files / subfolders in a folder.

    Args:
        path: Base directory to read from
        pattern: Glob pattern to match files against (e.g. "**/*.py" for Python files)
        recursive: Whether to search subdirectories
        include_dirs: Whether to include directories in results
        exclude: List of patterns to exclude (uses fnmatch against relative paths)
        max_depth: Maximum directory depth for recursive search

    Returns:
        A list of files / folders.
    """
    from upathtools import list_files

    pattern = pattern or "**/*"
    files = await list_files(
        path,
        pattern=pattern,
        include_dirs=include_dirs,
        recursive=recursive,
        exclude=exclude,
        max_depth=max_depth,
    )
    return "\n".join(str(f) for f in files)


async def start_process(  # noqa: D417
    ctx: AgentContext,
    command: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    output_limit: int | None = None,
) -> str:
    """Start a command in the background and return immediately with process ID.

    Args:
        command: Command to execute
        args: Command arguments
        cwd: Working directory
        env: Environment variables (added to current env)
        output_limit: Maximum bytes of output to retain

    Returns:
        Process ID for tracking the background process
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        process_id = await ctx.process_manager.start_process(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
            output_limit=output_limit,
        )
    except Exception as e:  # noqa: BLE001
        return f"Failed to start process: {e}"
    else:
        return f"Started process: {process_id}"


async def get_process_output(ctx: AgentContext, process_id: str) -> str:  # noqa: D417
    """Get current output from a background process.

    Args:
        process_id: Process identifier from start_background_process

    Returns:
        Current process output (stdout + stderr)
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        output = await ctx.process_manager.get_output(process_id)
        result = f"Process {process_id}:\n"
        if output.stdout:
            result += f"STDOUT:\n{output.stdout}\n"
        if output.stderr:
            result += f"STDERR:\n{output.stderr}\n"
        if output.exit_code is not None:
            result += f"Exit code: {output.exit_code}\n"
        if output.truncated:
            result += "Note: Output was truncated due to size limits\n"
        return result.strip()
    except ValueError as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"Error getting process output: {e}"


async def wait_for_process(ctx: AgentContext, process_id: str) -> str:  # noqa: D417
    """Wait for background process to complete and return final output.

    Args:
        process_id: Process identifier from start_process

    Returns:
        Final process output and exit code
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        exit_code = await ctx.process_manager.wait_for_exit(process_id)
        output = await ctx.process_manager.get_output(process_id)

        result = f"Process {process_id} completed with exit code {exit_code}\n"
        if output.stdout:
            result += f"STDOUT:\n{output.stdout}\n"
        if output.stderr:
            result += f"STDERR:\n{output.stderr}\n"
        if output.truncated:
            result += "Note: Output was truncated due to size limits\n"
        return result.strip()
    except ValueError as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"Error waiting for process: {e}"


async def kill_process(ctx: AgentContext, process_id: str) -> str:  # noqa: D417
    """Terminate a background process.

    Args:
        process_id: Process identifier from start_process

    Returns:
        Confirmation message
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        await ctx.process_manager.kill_process(process_id)
    except ValueError as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"Error killing process: {e}"
    else:
        return f"Process {process_id} has been terminated"


async def release_process(ctx: AgentContext, process_id: str) -> str:  # noqa: D417
    """Release resources for a background process.

    Args:
        process_id: Process identifier from start_process

    Returns:
        Confirmation message
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        await ctx.process_manager.release_process(process_id)
    except ValueError as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"Error releasing process: {e}"
    else:
        return f"Process {process_id} resources have been released"


async def list_processes(ctx: AgentContext) -> str:
    """List all active background processes.

    Returns:
        List of process IDs and basic information
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    try:
        process_ids = ctx.process_manager.list_processes()
        if not process_ids:
            return "No active processes"

        result = "Active processes:\n"
        for process_id in process_ids:
            try:
                info = await ctx.process_manager.get_process_info(process_id)
                status = (
                    "running" if info["is_running"] else f"exited ({info['exit_code']})"
                )
                args = " ".join(info["args"])
                result += f"- {process_id}: {info['command']} {args} [{status}]\n"
            except Exception as e:  # noqa: BLE001
                result += f"- {process_id}: Error getting info - {e}\n"
        return result.strip()
    except Exception as e:  # noqa: BLE001
        return f"Error listing processes: {e}"


async def ask_user(  # noqa: D417
    ctx: AgentContext,
    prompt: str,
    response_schema: dict[str, Any] | None = None,
) -> str:
    """Allow LLM to ask user a clarifying question during processing.

    This tool enables agents to ask users for additional information or clarification
    when needed to complete a task effectively.

    Args:
        prompt: Question to ask the user
        response_schema: Optional JSON schema for structured response (defaults to string)

    Returns:
        The user's response as a string
    """
    from mcp.types import ElicitRequestParams, ElicitResult, ErrorData

    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    schema = response_schema or {"type": "string"}  # string schema if no none provided
    params = ElicitRequestParams(message=prompt, requestedSchema=schema)
    result = await ctx.handle_elicitation(params)

    match result:
        case ElicitResult(action="accept", content=content):
            return str(content)
        case ElicitResult(action="cancel"):
            return "User cancelled the request"
        case ElicitResult():
            return "User declined to answer"
        case ErrorData(message=message):
            return f"Error: {message}"
        case _:
            return "Unknown error occurred"


async def add_local_mcp_server(  # noqa: D417
    ctx: AgentContext,
    name: str,
    command: str,
    args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> str:
    """Add a local MCP server via stdio transport.

    Args:
        name: Unique name for the MCP server
        command: Command to execute for the server
        args: Command arguments
        env_vars: Environment variables to pass to the server

    Returns:
        Confirmation message about the added server
    """
    from llmling_agent_config.mcp_server import StdioMCPServerConfig

    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    env = env_vars or {}
    config = StdioMCPServerConfig(name=name, command=command, args=args or [], env=env)
    ctx.agent.mcp.add_server_config(config)
    await ctx.agent.mcp.setup_server(config)

    return f"Added local MCP server {name!r} with command: {command}"


async def add_remote_mcp_server(  # noqa: D417
    ctx: AgentContext,
    name: str,
    url: str,
    transport: Literal["sse", "streamable-http"] = "streamable-http",
) -> str:
    """Add a remote MCP server via HTTP-based transport.

    Args:
        name: Unique name for the MCP server
        url: Server URL endpoint
        transport: HTTP transport type to use (http is preferred)

    Returns:
        Confirmation message about the added server
    """
    from llmling_agent_config.mcp_server import (
        SSEMCPServerConfig,
        StreamableHTTPMCPServerConfig,
    )

    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    match transport:
        case "sse":
            config: MCPServerConfig = SSEMCPServerConfig(name=name, url=url)
        case "streamable-http":
            config = StreamableHTTPMCPServerConfig(name=name, url=url)

    ctx.agent.mcp.add_server_config(config)
    await ctx.agent.mcp.setup_server(config)

    return f"Added remote MCP server '{name}' at {url} using {transport} transport"


if __name__ == "__main__":
    # import logging
    from llmling_agent import AgentPool, Capabilities

    user_prompt = """Add a stdio MCP server:
// 	"command": "npx",
// 	"args": ["mcp-graphql"],
// 	"env": { "ENDPOINT": "https://diego.one/graphql" }

."""

    async def main():
        async with AgentPool() as pool:
            caps = Capabilities(can_add_mcp_servers=True)
            agent = await pool.add_agent(
                "X", capabilities=caps, model="openai:gpt-5-nano"
            )
            agent.tool_used.connect(print)
            result = await agent.run(user_prompt)
            print(result)
            result = await agent.run("Which tools does it have?")
            print(result)

    asyncio.run(main())


async def load_skill(ctx: AgentContext, skill_name: str) -> str:  # noqa: D417
    """Load a Claude Code Skill and return its instructions.

    Args:
        skill_name: Name of the skill to load

    Returns:
        The full skill instructions for execution
    """
    if isinstance(ctx, RunContext):
        ctx = ctx.deps

    registry = ctx.agent.skills_registry
    await registry.discover_skills()

    try:
        skill = registry.get(skill_name)
        instructions = skill.load_instructions()

        # Format the skill content for Claude to follow
    except Exception as e:  # noqa: BLE001
        return f"Failed to load skill '{skill_name}': {e}"
    else:
        return f"""
<command-message>The "{skill_name}" skill is loading</command-message>

# {skill.name}

{instructions}

---
Skill loaded from: {skill.source}
Skill directory: {skill.skill_path}
"""
