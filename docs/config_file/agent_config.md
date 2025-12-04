---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/nodes.py
---

# Agent Configuration

Individual agent configurations define the behavior, capabilities, and settings for each agent in your manifest. Each agent entry in the `agents` dictionary represents a complete agent setup.

## Basic Structure

```yaml
agents:
  agent_name:  # Agent identifier (key in agents dict)
    # Basic configuration
    name: "agent_name"  # Optional override for agent name
    inherits: "base_agent"  # Optional parent config to inherit from
    description: "Agent description"
    model: "openai:gpt-5"  # Model specification
    debug: false

    # Additional configuration...
```

## Core Configuration

### Basic Settings

```yaml
agents:
  my_agent:
    name: "my_agent"  # Optional: Override the key name
    description: "What this agent does"
    model: "openai:gpt-5"  # or structured model definition
    debug: false  # Enable debug logging
    inherits: "base_config"  # Inherit from another config
```

### Provider Behavior

```yaml
agents:
  my_agent:
    retries: 1  # Number of retry attempts
    end_strategy: "early"  # "early" | "complete" | "confirm"
    model_settings: {}  # Additional model parameters
```

## Output Configuration

### Structured Output

Define how the agent should format its responses:

```yaml
agents:
  analyzer:
    output_type:
      type: "inline"  # Define schema inline
      fields:
        success:
          type: "bool"
          description: "Whether operation succeeded"
        result:
          type: "str" 
          description: "Analysis result"
    
    # Or reference imported type
    output_type:
      type: "import"
      import_path: "myapp.types.AnalysisResult"
    
    result_tool_name: "final_result"  # Tool name for result validation
    result_tool_description: "Create final response"  # Optional description
    output_retries: 3  # Validation retry attempts
```

## Prompts and Behavior

### System and User Prompts

```yaml
agents:
  my_agent:
    system_prompts: 
      - "You are a helpful assistant specializing in..."
      - type: "file"
        path: "prompts/system.txt"
    
    user_prompts: 
      - "Example default query"  # Default user inputs
      - "Another example"
```

## State Management

### Session Configuration

```yaml
agents:
  my_agent:
    session:
      name: "my_session"  # Session identifier
      since: "1h"  # Load messages from last hour
      # Other session options...
    
    avatar: "path/to/avatar.png"  # UI avatar image
```

## Capabilities

### Toolsets

Enable specific toolset capabilities:

```yaml
agents:
  my_agent:
    toolsets:
      - type: "agent_management"  # Enables delegation to other agents
      - type: "resource_access"   # Enables loading resources
      - type: "web_search"        # Enables web search capabilities
      # Additional toolsets...
```

### Knowledge Sources

Configure knowledge and context:

```yaml
agents:
  my_agent:
    knowledge:
      paths: 
        - "docs/**/*.md"
        - "src/**/*.py"
      resources:
        - type: "repository"
          url: "https://github.com/user/repo"
        - type: "web"
          url: "https://example.com/docs"
      prompts:
        - type: "file"
          path: "prompts/context.txt"
        - type: "inline"
          content: "Additional context information"
```

### MCP Server Integration

Configure Model Context Protocol servers:

```yaml
agents:
  my_agent:
    mcp_servers:
      # Detailed configuration
      - type: "stdio"
        command: "python"
        args: ["-m", "mcp_server"]
        env:
          API_KEY: "${MCP_API_KEY}"
      
      # Shorthand syntax
      - "python -m other_server"
      - "uvx some-tool-server"
```

## Agent Relationships

### Workers (Sub-agents)

Configure child agents that work under this agent:

```yaml
agents:
  senior_dev:
    workers:
      # Detailed configuration
      - type: "agent"
        name: "code_reviewer"
        reset_history_on_run: true    # Fresh conversation each time
        pass_message_history: false   # Don't share parent's history
      
      # Shorthand syntax
      - "bug_analyzer"  # Simple reference
      - "formatter"
```

### Message Routing

Configure how messages flow to other agents:

```yaml
agents:
  my_agent:
    connections:
      - type: "node"
        name: "reporter"
        connection_type: "run"  # "run" | "context" | "forward"
        wait_for_completion: true
      
      - type: "file"
        path: "outputs/results.txt"
        format: "json"
```

## Event Handling

### Triggers

Configure event-based activation:

```yaml
agents:
  file_watcher:
    triggers:
      - type: "file"
        name: "code_change"
        paths: ["src/**/*.py"]
        extensions: [".py"]
        recursive: true
        
      - type: "schedule"
        name: "daily_report"
        cron: "0 9 * * *"  # Every day at 9 AM
```

## Complete Example

Here's a comprehensive agent configuration:

```yaml
agents:
  code_analyzer:
    name: "Code Analysis Specialist"
    description: "Analyzes code quality, security, and performance"
    model: "openai:gpt-5"
    debug: false
    
    # Output configuration
    output_type:
      type: "inline"
      fields:
        severity:
          type: "str"
          enum: ["low", "medium", "high", "critical"]
        issues:
          type: "list"
          items:
            type: "object"
            properties:
              file:
                type: "str"
              line:
                type: "int"
              message:
                type: "str"
    
    # Behavior
    retries: 2
    end_strategy: "complete"
    output_retries: 3
    
    # Prompts
    system_prompts:
      - "You are an expert code analyzer with deep knowledge of security and performance best practices."
      - type: "file"
        path: "prompts/code_analysis_system.md"
    
    # Capabilities
    toolsets:
      - type: "resource_access"
      - type: "agent_management"
    
    # Knowledge
    knowledge:
      paths:
        - "src/**/*.py"
        - "docs/coding_standards.md"
      resources:
        - type: "repository"
          url: "https://github.com/company/style-guide"
    
    # Sub-agents
    workers:
      - type: "agent"
        name: "security_scanner"
        reset_history_on_run: true
      - "performance_profiler"
    
    # Event triggers
    triggers:
      - type: "file"
        name: "code_changes"
        paths: ["src/**/*.py", "tests/**/*.py"]
        extensions: [".py"]
```

## Configuration Inheritance

Agents can inherit configuration from other agents or base configurations:

```yaml
agents:
  base_agent:
    model: "openai:gpt-5"
    retries: 2
    toolsets:
      - type: "resource_access"
  
  specialized_agent:
    inherits: "base_agent"  # Inherits all settings from base_agent
    description: "Specialized version"
    # Override or add specific settings
    system_prompts:
      - "You are a specialized agent..."
```

## Next Steps

- [Worker Configuration](worker_config.md) for configuring sub-agents
- [Tool Configuration](tool_config.md) for custom tools
- [Knowledge Configuration](knowledge_config.md) for advanced knowledge setup
- [MCP Configuration](mcp_config.md) for Model Context Protocol servers