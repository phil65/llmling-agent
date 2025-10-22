# Agent Toolsets

## What are Toolsets?

Toolsets in LLMling define groups of related tools that agents can access. Rather than configuring individual tool permissions, you enable entire categories of functionality through toolset configurations.

Think of toolsets as "skill packages" that give agents specific capabilities - from file operations to process management to agent coordination.

## Available Toolsets

### Agent Management (`agent_management`)
Enables agents to discover, coordinate with, and manage other agents:

```yaml
agents:
  coordinator:
    toolsets:
      - type: agent_management
```

**Provides tools:**
- `list_available_agents` - Discover other agents in the pool
- `list_available_teams` - Discover available teams
- `delegate_to` - Assign tasks to other agents
- `ask_agent` - Ask other agents directly
- `add_agent` - Add new agents to the pool
- `add_team` - Create new teams
- `connect_nodes` - Connect agents/teams in workflows
- `create_worker_agent` - Create worker agents as tools
- `spawn_delegate` - Create temporary delegate agents

### File Access (`file_access`)
Basic file system operations:

```yaml
agents:
  reader:
    toolsets:
      - type: file_access
```

**Provides tools:**
- `read_file` - Read local and remote files
- `list_directory` - List directory contents

### Resource Access (`resource_access`)
Access to LLMling resources and configurations:

```yaml
agents:
  assistant:
    toolsets:
      - type: resource_access
```

**Provides tools:**
- `load_resource` - Load resource content
- `get_resources` - Discover available resources

### Code Execution (`code_execution`)
Execute Python code and system commands:

```yaml
agents:
  developer:
    toolsets:
      - type: code_execution
```

**Provides tools:**
- `execute_python` - Execute Python code (WARNING: No sandbox)
- `execute_command` - Execute CLI commands

### Process Management (`process_management`)
Start and manage background processes:

```yaml
agents:
  build_manager:
    toolsets:
      - type: process_management
```

**Provides tools:**
- `start_process` - Start background processes
- `get_process_output` - Check process output
- `wait_for_process` - Wait for process completion
- `kill_process` - Terminate processes
- `release_process` - Clean up process resources
- `list_processes` - Show active processes

### Tool Management (`tool_management`)
Register and manage tools dynamically:

```yaml
agents:
  admin:
    toolsets:
      - type: tool_management
```

**Provides tools:**
- `register_tool` - Register importable functions as tools
- `register_code_tool` - Create tools from code
- `install_package` - Install Python packages

### User Interaction (`user_interaction`)
Direct interaction with users:

```yaml
agents:
  assistant:
    toolsets:
      - type: user_interaction
```

**Provides tools:**
- `ask_user` - Ask users clarifying questions

### History (`history`)
Access conversation history and statistics:

```yaml
agents:
  analyst:
    toolsets:
      - type: history
```

**Provides tools:**
- `search_history` - Search conversation history
- `show_statistics` - Display usage statistics

### Integrations (`integrations`)
External service integrations:

```yaml
agents:
  integrator:
    toolsets:
      - type: integrations
```

**Provides tools:**
- `add_local_mcp_server` - Add local MCP servers
- `add_remote_mcp_server` - Add remote MCP servers
- `load_skill` - Load Claude Code Skills

## Common Patterns

### Basic Assistant
```yaml
agents:
  assistant:
    model: openai:gpt-4
    toolsets:
      - type: resource_access
      - type: file_access
      - type: user_interaction
```

### Team Coordinator
```yaml
agents:
  coordinator:
    model: openai:gpt-4
    toolsets:
      - type: agent_management
      - type: history
    system_prompts:
      - You coordinate tasks across multiple agents
```

### Developer Agent
```yaml
agents:
  developer:
    model: anthropic:claude-3-5-sonnet-20241022
    toolsets:
      - type: file_access
      - type: code_execution
      - type: process_management
      - type: tool_management
    system_prompts:
      - You are a software developer with full system access
```

### Restricted Agent
```yaml
agents:
  restricted:
    model: openai:gpt-4-mini
    toolsets: []  # No toolsets = only predefined tools
    tools:
      - calculator
      - web_search
```

## Security Considerations

Toolsets provide different levels of system access:

**Low Risk:**
- `file_access` - Read-only file operations
- `resource_access` - Access to configured resources
- `user_interaction` - User prompts only

**Medium Risk:**
- `agent_management` - Can create and coordinate agents
- `history` - Access to conversation data
- `integrations` - External service access

**High Risk:**
- `code_execution` - Can execute arbitrary code
- `process_management` - System process control
- `tool_management` - Can modify available tools

Always use the principle of least privilege - only enable toolsets that agents actually need.

## Custom Toolsets

You can also create custom toolsets by implementing your own provider:

```yaml
agents:
  specialized:
    toolsets:
      - type: custom
        import_path: "mypackage.toolsets.SpecializedTools"
```

See the [Custom Tools](../tools/custom.md) guide for implementation details.