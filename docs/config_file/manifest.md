# Agent Manifest

The agent manifest is a YAML file that defines your complete agent setup.
The config part is powered by [Pydantic](https://docs.pydantic.dev/latest/) and provides excellent validation
and IDE support for YAML linters by providing an extensive, detailed schema.

Let's look at a complete, correctly structured example:

```yaml
# Root level configuration
agents:
  analyzer:  # Agent name (key in agents dict)
    # Basic configuration
    name: "analyzer"  # Optional override for agent name
    inherits: "base_agent"  # Optional parent config to inherit from
    description: "Code analysis specialist"
    model: "openai:gpt-5"  # or structured model definition
    debug: false

    # Provider behavior
    retries: 1
    end_strategy: "early"  # "early" | "complete" | "confirm"

    # Structured output
    output_type:
      type: "inline"  # or "import" for Python types
      fields:
        success:
          type: "bool"
          description: "Whether analysis succeeded"
    result_tool_name: "final_result"  # Name for result validation tool
    result_tool_description: "Create final response"  # Optional description
    output_retries: 3  # Validation retry count

    # Agent behavior
    system_prompts: ["You are a code analyzer..."]
    user_prompts: ["Example query..."]  # Default queries
    model_settings: {}  # Additional model parameters

    # State management
    session:                 # Initial session loading
      name: my_session       # Optional session identifier
      since: 1h             # Only messages from last hour
    avatar: "path/to/avatar.png"  # Optional UI avatar

    # Toolsets
    toolsets:
      - type: agent_management  # Enables delegation
      - type: resource_access   # Enables resource loading


    # Knowledge configuration
    knowledge:
      paths: ["docs/**/*.md"]
      resources:
        - type: "repository"
          url: "https://github.com/user/repo"
      prompts:
        - type: "file"
          path: "prompts/analysis.txt"

    # MCP integration
    mcp_servers:
      - type: "stdio"
        command: "python"
        args: ["-m", "mcp_server"]
      - "python -m other_server"  # shorthand syntax

    # Agent relationships
    workers:
      - type: agent
        name: "formatter"
        reset_history_on_run: true
        pass_message_history: false
      - "linter"  # shorthand syntax

    # Message routing
    connections:
      - type: node
        name: "reporter"
        connection_type: "run"  # "run" | "context" | "forward"
        wait_for_completion: true

    # Event handling
    triggers:
      - type: "file"
        name: "code_change"
        paths: ["src/**/*.py"]
        extensions: [".py"]
        recursive: true

  # Additional agents...
  planner:
    model: "openai:gpt-5-nano"
    # ... configuration for planner agent

teams:
  # Complex workflows via YAML
  full_pipeline:
    mode: sequential
    members:
      - analyzer
      - planner
    connections:
      - type: node
        name: final_reviewer
        wait_for_completion: true
      - type: file
        path: "reports/{date}_workflow.txt"

# Shared response definitions
responses:
  AnalysisResult:
    response_schema:
      type: "inline"
      fields:
        severity:
          type: "str"
          description: "Issue severity"
    CodeMetrics:
      type: "import"
      import_path: "myapp.types.CodeMetrics"

# Storage configuration
storage:
  providers:
    - type: "sql"
      url: "sqlite:///history.db"
      pool_size: 5
    - type: "text_file"
      path: "logs/chat.log"
      format: "chronological"
  log_messages: true
  log_conversations: true
  log_commands: true

# Pre-defined tasks
jobs:
  analyze_code:
    prompt: "Analyze this code: {code}"
    output_type: "AnalysisResult"
    knowledge:
      paths: ["src/**/*.py"]
    tools:
      - "analyze_complexity"
      - import_path: "myapp.tools.analyze_security"
```

## Key Concepts

### Agent Configuration
Each agent entry defines:
- Provider type and model
- Response formatting
- Capabilities and permissions
- Environment and knowledge sources
- Connections to other agents

### Response Types
Define structured output formats either:
- Inline in the YAML (`type: "inline"`)
- By importing Python types (`type: "import"`)

### Storage
Configure how agent interactions are stored:
- SQL databases
- Text logs
- File storage
- Memory storage (for testing)

### Tasks
Predefine common operations with:
- Prompt templates
- Required knowledge
- Expected response types
- Tool configurations

## Usage

Load a manifest i your code:
```python
from llmling_agent import AgentPool

async with AgentPool("agents.yml") as pool:
    agent = pool.get_agent("analyzer")
    result = await agent.run("Analyze this code...")
```

!!! note
    You can get linter support by adding this line at the top of your YAML:
    `# yaml-language-server: $schema=https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/schema/config-schema.json`
    Versioned config files will arrive soon!

LLMling-Agent supports the YAML inheritance functionality for the manifest also known from MkDocs, using the
`INHERIT` key on the top level. It even supports UPaths (universal-pathlib)


 ## Next Steps
 - [Environment Configuration](environment.md) for detailed tool/resource setup
 - [Response Types](responses.md) for structured output configuration
 - [Storage Configuration](storage.md) for history and logging options
