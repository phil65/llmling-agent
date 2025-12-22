---
title: ACP Integration
description: Agent Client Protocol integration for desktop applications
icon: material/desktop-tower
---

## What is ACP?

The Agent Client Protocol (ACP) is a standardized JSON-RPC 2.0 protocol that enables communication between code editors and AI agents. It allows AgentPool to integrate seamlessly with desktop applications and IDEs that support the protocol.

ACP provides:

- Bidirectional communication between editor and agent
- Session management and conversation history
- File system operations with permission handling
- Terminal integration for command execution
- Tool confirmation mode switching for different workflows

## ACP Agents as First-Class Citizens

In AgentPool, external ACP agents are **first-class citizens** in the agent ecosystem. They:

- ✅ **Participate fully in the agent pool** - Can be discovered, delegated to, and coordinated with
- ✅ **Support most features of "native" Pydantic-AI based agents** - Tools, contexts, delegation, state management
- ✅ **Run in configurable environments** - Local, Docker, E2B, remote sandboxes
- ✅ **Access internal toolsets** - Via automatic MCP bridge for supported agents
- ✅ **Auto-managed lifecycle** - Automatic spawn, cleanup, and error handling

The key difference is that ACP agents run as **separate processes** communicating via the ACP protocol, while still appearing as integrated members of the pool to other agents.

## Installation & Setup

Using uvx for one-off usage:

```bash
uvx --python 3.13 agentpool[default]@latest serve-acp --help
```

## CLI Usage

### Basic Commands

Start an ACP server from a configuration file:

```bash
agentpool serve-acp agents.yml
```

### Available Options

- `--file-access/--no-file-access`: Enable file system operations (default: enabled)
- `--terminal-access/--no-terminal-access`: Enable terminal integration (default: enabled)
- `--session-support/--no-session-support`: Enable session loading (default: enabled)
- `--model-provider`: Specify model providers to search (can be repeated)
- `--show-messages`: Show message activity in logs
- `--log-level`: Set logging level (debug, info, warning, error)

## IDE Configuration

### Zed Editor

Add this configuration to your Zed `settings.json`:

```json
{
  "agent_servers": {
    "AgentPool": {
      "command": "uvx",
      "args": [
        "--python",
        "3.13",
        "agentpool[default,coding]@latest",
        "serve-acp",
        "https://raw.githubusercontent.com/phil65/agentpool/refs/heads/main/docs/examples/pick_experts/config.yml", # <- insert your agent config here
        "--model-provider",
        "openai"
      ],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

This configuration:

- Uses uvx to run the latest version without local installation
- Points to a remote configuration file with multiple expert agents
- Specifies OpenAI as the model provider
- Sets the required API key via environment variables

### Other IDEs

For IDEs that support ACP, the general pattern is:

1. Set the command to `agentpool` (or `uvx agentpool[default]@latest`)
2. Add `serve-acp` as the first argument
3. Specify your configuration file path
4. Add any desired CLI options
5. Set required environment variables (API keys, etc.)

## The ACP Bridge Concept

### AgentPool as Both Server AND Client

Understanding AgentPool's ACP integration requires grasping a key architectural concept: **AgentPool acts as BOTH an ACP server AND an ACP client simultaneously**. This dual role is what makes the integration so powerful.

#### As an ACP Server (IDE Integration)

When you run `agentpool serve-acp`, it becomes an **ACP server** that IDEs can connect to:

```mermaid
graph LR
    A["IDE (Zed)<br/>ACP Client"] <-->|ACP Protocol| B["agentpool<br/>ACP Server<br/>(serve-acp command)"]
```

In this mode, AgentPool:

- Receives prompts from the IDE
- Sends back agent responses
- Handles file operations on behalf of the IDE
- Manages terminal sessions
- Provides tool confirmation modes for the IDE to switch between

#### As an ACP Client (External Agent Integration)

At the same time, AgentPool can act as an **ACP client** to integrate external ACP agents:

```mermaid
graph LR
    A["agentpool<br/>ACP Client<br/>(external agent)"] <-->|ACP Protocol| B["Claude Code<br/>ACP Server<br/>(subprocess)"]
```

In this mode, AgentPool:

- Spawns external ACP agent processes (Claude Code, Gemini CLI, etc.)
- Sends prompts to those agents
- Receives their responses
- Manages their lifecycle (startup, cleanup)
- Provides them access to internal toolsets via MCP

#### The Bridge: Both Roles Together

The real power comes when **both roles work together**:

```mermaid
graph TB
    IDE["IDE (Zed)<br/>ACP Client"]
    LLM["agentpool<br/>Server ↔ Pool ↔ Client"]
    Claude["Claude Code<br/>ACP Server"]
    Pool["Agent Pool<br/>• Internal Agents<br/>• Teams<br/>• Resources<br/>• Tools"]
    
    IDE <-->|ACP| LLM
    LLM <-->|ACP| Claude
    LLM <--> Pool
```

#### Real-World Example

When you configure Zed to use AgentPool with external ACP agents:

1. **Zed** connects to AgentPool via ACP (IDE → Server)
2. **AgentPool** maintains an agent pool with both:
   - Internal agents (regular Python-based agents)
   - External agents (spawned ACP processes like Claude Code)
3. When you send a prompt through Zed:
   - It goes to AgentPool (Server role)
   - AgentPool routes it to the appropriate agent
   - If the agent is internal → direct execution
   - If the agent is external → ACP client call to that subprocess
4. **Orchestration** becomes possible:
   - An internal agent can delegate to Claude Code
   - Claude Code can call back to internal agents via MCP toolsets
   - All coordinated through the agent pool

#### Why This Matters

This bridge architecture enables:

- ✅ **Unified interface**: IDE sees one server, regardless of how many agents
- ✅ **Heterogeneous agents**: Mix internal Python agents with external protocol-bridged agents
- ✅ **Transparent delegation**: Agents can work together across process boundaries
- ✅ **Toolset sharing**: External agents access internal capabilities via MCP bridge
- ✅ **Environment flexibility**: External agents can run in Docker, E2B, remote servers
- ✅ **Protocol translation**: IDE ↔ ACP ↔ Internal Pool ↔ ACP ↔ External Agents

The key insight: **AgentPool doesn't just implement ACP—it bridges multiple ACP worlds together into a unified orchestration layer**.

## Tool Confirmation Modes

The IDE's mode selector controls **tool confirmation behavior** - how the agent handles tool executions that may affect your files or system.

Available modes:

- **Auto-approve**: Tools execute without confirmation (fastest workflow)
- **Confirm destructive**: Only confirm file writes, deletions, and terminal commands  
- **Confirm all**: Confirm every tool execution (safest)

The mode affects tool confirmation, not agent selection. To work with different agents, use the `/spawn` command or configure delegation through toolsets.

## External ACP Agents

AgentPool can integrate external ACP-enabled agents (like Claude Code, Codex, Goose, etc.) into your agent pool through YAML configuration. This allows you to delegate tasks to specialized external agents while maintaining a unified interface.

### Configuration

Define external ACP agents in your manifest:

```yaml
# agents.yml
agents:
  claude:
    type: acp
    provider: claude
    display_name: "Claude Code"
    description: "Claude Code through ACP"
  
  codex:
    type: acp
    provider: codex
    display_name: "Codex"
    description: "Codex Code through ACP"
  
  goose:
    type: acp
    provider: goose
    display_name: "Goose"
    description: "Block's Goose agent through ACP"
  
  fast-agent:
    type: acp
    provider: fast-agent
    display_name: "Fast Agent"
    description: "fast-agent through ACP"
    model: openai.gpt-4o

agents:
  coordinator:
    model: openai:gpt-5-mini
    toolsets:
      - type: agent_management  # Enables delegation to ACP agents
```

### Supported External Agents

#### MCP-Capable Agents (with Toolset Bridging)

These agents support MCP servers and can use internal AgentPool toolsets:

- **claude**: Claude Code (Anthropic's coding assistant)
  - Full MCP support via `--mcp-config`
  - Permission modes, structured outputs
  - Tool filtering and access control

- **gemini**: Google's Gemini CLI
  - Native MCP integration
  - Configurable temperature, context window
  - Multi-modal support

- **auggie**: Augment Code
  - MCP support via `--mcp-config`
  - GitHub integration
  - Project context awareness

- **kimi**: Moonshot AI's Kimi CLI
  - MCP support via `--mcp-config`
  - Chinese language optimization
  - Work directory management

#### Standard ACP Agents

These agents use ACP protocol but don't support MCP toolset bridging:

- **codex**: OpenAI Codex (Zed integration)
- **opencode**: OpenCode agent
- **goose**: Block's Goose agent
- **fast-agent**: Fast Agent with configurable models
- **openhands**: OpenHands (formerly OpenDevin)
- **amp**: AmpCode agent
- **cagent**: Docker-based cagent
- **stakpak**: Stakpak Agent
- **vtcode**: VT Code agent

### Usage with Agent Pool

Once configured, external ACP agents are automatically available in the agent pool:

```python
from agentpool.delegation import AgentPool

async def main():
    async with AgentPool("agents.yml") as pool:
        # Access external ACP agents just like regular agents
        claude = pool.acp_agents["claude"]
        result = await claude.run("Refactor this code to use async/await")
        
        # Or delegate from a coordinator agent
        coordinator = pool.agents["coordinator"]
        result = await coordinator.run(
            "Use the Claude agent to review and improve the codebase"
        )
        
        # Check what agents are available
        print(f"Regular agents: {list(pool.agents.keys())}")
        print(f"ACP agents: {list(pool.acp_agents.keys())}")
```

The external agents are spawned as subprocess instances and communicate via the ACP protocol, with automatic lifecycle management and cleanup.

## ACP Agent Architecture

### How ACP Agents Operate

ACP agents in AgentPool follow this execution model:

1. **Process Spawning**
   - Agent configuration defines command and arguments
   - Process spawned in configured execution environment
   - stdin/stdout used for JSON-RPC communication

2. **Protocol Initialization**
   - Client sends `initialize` request with capabilities
   - Agent responds with its capabilities (MCP support, etc.)
   - Session negotiation completes

3. **Session Management**
   - `new_session` request creates working session
   - MCP servers passed during session creation
   - Working directory and permissions established

4. **Message Exchange**
   - Prompts sent via `session/prompt` requests
   - Streaming responses via session updates
   - Tool calls handled through MCP or native protocols

5. **Lifecycle Management**
   - Automatic cleanup on pool exit
   - Process termination and resource cleanup
   - Error handling and recovery

### Execution Environments

ACP agents can run in different environments based on configuration:

```yaml
agents:
  claude_local:
    type: acp
    provider: claude
    execution_environment: local  # Default
    
  claude_docker:
    type: acp
    provider: claude
    execution_environment:
      type: docker
      image: python:3.13-slim
      
  claude_e2b:
    type: acp
    provider: claude
    execution_environment:
      type: e2b
      template: python-sandbox
      
  claude_remote:
    type: acp
    provider: claude
    execution_environment:
      type: srt  # Secure Remote Terminal
      host: remote-server.com
```

**Environment types:**

- **local**: Run on local machine (default)
- **docker**: Run in Docker container with specified image
- **e2b**: Run in E2B sandbox (cloud sandboxes)
- **beam**: Run on Beam Cloud infrastructure
- **daytona**: Run in Daytona development environment
- **srt**: Run on remote server via Secure Remote Terminal

The execution environment handles:

- File system access (sandboxed or real)
- Terminal operations (bash, command execution)
- Network access controls
- Resource limits and isolation

### Remote Agent Operations with Client Execution Environment

ACP agents support a **dual environment** model, allowing the agent process to run locally while operating on remote filesystems. This is useful when you want an agent to work on files in a Docker container, remote server, or cloud sandbox without installing the agent there.

```mermaid
flowchart LR
    subgraph Local["Your Machine"]
        UI[IDE / UI]
        Agent[ACP Agent Process]
    end
    
    subgraph Remote["Remote Environment"]
        FS[Remote Filesystem]
        Term[Remote Terminals]
    end
    
    UI <-->|ACP Protocol| Agent
    Agent -->|client_env| FS
    Agent -->|client_env| Term
```

**Configuration:**

```yaml
agents:
  remote_coder:
    type: acp
    provider: claude
    # Agent's toolsets use this environment
    execution_environment: local
    # Subprocess file/terminal requests go here instead
    client_execution_environment:
      type: docker
      image: python:3.12
      mount: /home/user/project:/workspace
```

**How it works:**

| Property | Purpose | Example |
|----------|---------|---------|
| `execution_environment` | Where the agent's internal toolsets operate | Local filesystem (or forwarded to UI via ACP) |
| `client_execution_environment` | Where subprocess file/terminal requests go | Docker, SSH, E2B sandbox |

!!! note "Environment Override When Using ACP Server"
    When agents run through the ACP server (e.g., via IDE integration), the `execution_environment` 
    setting is **overridden** by the ACP session. The session injects an `ACPExecutionEnvironment` 
    that routes all toolset operations back to the IDE/client filesystem.
    
    This means:
    
    - **Standalone usage**: `execution_environment` controls where toolsets operate
    - **Via ACP server**: Toolsets always operate on the client's filesystem (the IDE's workspace)
    
    The `client_execution_environment` is **not** overridden - it always controls where the 
    subprocess's file/terminal requests go, regardless of how the agent is used.

**Use cases:**

- **Development in containers**: Agent runs locally but edits files inside a Docker container
- **Remote server development**: Agent operates on a remote server via SSH without being installed there
- **Sandboxed execution**: Agent runs commands in an isolated E2B sandbox for security

If `client_execution_environment` is not set, it falls back to `execution_environment` (the default behavior).

### Permission Model

ACP agents have fine-grained permission controls:

```yaml
agents:
  restricted_claude:
    type: acp
    provider: claude
    allow_file_operations: false  # Disable file read/write
    allow_terminal: false         # Disable terminal access
    auto_grant_permissions: false # Require user confirmation
    permission_mode: dontAsk      # Claude-specific: deny by default
```

Permissions are enforced at multiple levels:

- **Client side**: AgentPool's ACPClientHandler validates requests
- **Agent side**: External agent's own permission system
- **Environment side**: Execution environment provides isolation

## Toolset Bridging via MCP

### Overview

For MCP-capable ACP agents (Claude, Gemini, Auggie, Kimi), agentpool can automatically expose internal toolsets via an in-process MCP server bridge. This allows external agents to use powerful internal capabilities like subagent delegation, agent management, and custom tools.

### How It Works

1. **Configuration**: Declare toolsets in agent config
2. **Bridge Creation**: ToolManagerBridge spawns HTTP/SSE MCP server
3. **Registration**: Tools registered with FastMCP server
4. **Session Integration**: MCP server URL passed to agent session
5. **Tool Invocation**: Agent calls tools via standard MCP protocol
6. **Context Bridging**: Calls proxied to internal tools with full context

The bridge runs **in-process** on the same event loop as the agent pool, providing zero-latency access to pool state while exposing a standard network interface to the external agent.

### Configuration Example

```yaml
agents:
  claude_orchestrator:
    type: acp
    provider: claude
    description: "Claude with delegation capabilities"
    permission_mode: acceptEdits
    toolsets:
      # Subagent delegation tools
      - type: subagent
      
      # Agent lifecycle management
      - type: agent_management
      
      # Custom tools
      - type: custom
        tools:
          - my_custom_tool
  specialist_a:
    type: native
    model: openai:gpt-4
    system_prompt: "Expert in data processing"
    
  specialist_b:
    type: native
    model: anthropic:claude-sonnet-4
    system_prompt: "Expert in API design"
```

With this configuration, the Claude agent can:

- List available agents via `list_available_nodes`
- Delegate work via `delegate_to` or `ask_agent`
- Create new agents dynamically via `add_agent`
- Use any custom tools you've registered

### Available Toolsets

All standard toolsets can be exposed to MCP-capable agents:

- **subagent**: Delegation tools (`delegate_to`, `ask_agent`, `list_available_nodes`)
- **agent_management**: Lifecycle tools (`add_agent`, `remove_agent`, `get_agent_info`)
- **search**: Web/news search (requires provider configuration)
- **notifications**: Send notifications via Apprise
- **semantic_memory**: Vector-based memory and retrieval
- **custom**: Your own tools via ResourceProvider

### Usage Example

Once configured, the external agent automatically has access to the tools:

```python
async with AgentPool("config.yml") as pool:
    claude = pool.acp_agents["claude_orchestrator"]
    
    # Claude can now delegate to internal agents
    result = await claude.run(
        "Use the list_available_nodes tool to see what agents are available, "
        "then delegate the data processing task to specialist_a"
    )
    
    # The tool bridge handles the MCP protocol translation
    # and injects proper AgentContext for internal tool execution
```

### Tool Bridge Architecture

```mermaid
graph TB
    subgraph External["External ACP Agent (Claude Code Process)"]
        MCP["MCP Client<br/>• Discovers tools via MCP<br/>• Calls tools over HTTP/SSE"]
    end
    
    subgraph Bridge["ToolManagerBridge (In-Process MCP Server)"]
        FastMCP["FastMCP Server<br/>• Exposes tools as MCP endpoints<br/>• Handles MCP protocol negotiation"]
        Context["Context Bridge<br/>• Creates synthetic AgentContext<br/>• Injects pool access for delegation<br/>• Provides progress reporting"]
    end
    
    subgraph Internal["Internal Toolsets (Same Process)"]
        Subagent["SubagentTools<br/>• delegate_to() → Other agents in pool<br/>• ask_agent() → Query specific agent<br/>• list_available_nodes() → Pool inspection"]
        Management["AgentManagementTools<br/>• add_agent() → Dynamic agent creation<br/>• remove_agent() → Lifecycle management"]
    end
    
    MCP -->|HTTP/SSE<br/>MCP Protocol| FastMCP
    Context -->|Direct Function Call| Subagent
    Context -->|Direct Function Call| Management
```

**Key Features:**

- **Zero IPC overhead**: Bridge runs in same process as pool
- **Type-safe**: Full type checking from tool definition to MCP
- **Context injection**: Tools receive proper AgentContext with pool access
- **Auto-discovery**: Tools automatically exposed, no manual registration
- **Per-agent isolation**: Each ACP agent gets its own bridge instance
- **Clean lifecycle**: Bridges automatically cleaned up with pool

### Advanced Configuration

For more control over the bridge:

```yaml
agents:
  claude_advanced:
    type: acp
    provider: claude
    toolsets:
      - type: subagent
        # Toolset-specific config here
        
      - type: custom
        tools:
          - only_specific_tool
          
    # Additional MCP servers (alongside toolset bridge)
    mcp_servers:
      - name: filesystem
        type: stdio
        command: npx
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/project"]
        
      - name: external-api
        type: sse
        url: https://api.example.com/mcp
```

The agent will have access to:

1. **Internal toolsets** via automatic bridge
2. **External MCP servers** you explicitly configure
3. **Agent's built-in tools** (if any)

### Security Considerations

Toolset bridging is secure by design:

- ✅ **Explicit opt-in**: Only agents with `toolsets` field get bridges
- ✅ **Scoped access**: Tools see only what AgentContext provides
- ✅ **Permission gates**: Input providers can confirm dangerous operations
- ✅ **Network isolation**: Bridge binds to localhost by default
- ✅ **Process isolation**: External agent runs in separate process
- ✅ **Environment sandboxing**: Execution environment provides additional isolation

Best practices:

- Use `allow_file_operations: false` for read-only agents
- Use `auto_grant_permissions: false` for sensitive operations
- Use sandboxed execution environments (Docker, E2B) for untrusted agents
- Limit toolsets to only what the agent needs

## Configuration

### Remote Configurations

You can reference remote configuration files directly:

```bash
agentpool serve-acp https://example.com/config.yml
```

### Provider Selection

Limit which providers are searched for models:

```bash
agentpool serve-acp config.yml --model-provider openai --model-provider anthropic
```

If not provider is passed, OpenRouter is used.

Available model providers:

- `anthropic`
- `groq`
- `mistral`
- `openai`
- `openrouter`
- `github`
- `copilot`
- `cerebras`
- `gemini`
- `cohere`
- `deepseek`
- `requesty`
- `xai`
- `comet`
- `novita`
- `vercel-gateway`
- `chutes`
- `cortecs`
- `azure`
- `fireworks-ai`
- `ollama`
- `moonshotai`
- `zai`

The providers depend on environment variables being set, like `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, etc.
