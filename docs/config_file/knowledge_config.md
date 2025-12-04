---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/knowledge.py
---

# Knowledge Configuration

Knowledge defines information sources that are loaded during agent initialization to provide context for the agent's operations. Unlike environment resources (which are loaded on-demand), knowledge sources are loaded at startup and remain available in the agent's context.

```yaml
agents:
  research_assistant:
    knowledge:
      # Simple File Paths
      paths:
        - "docs/**/*.md"          # Glob patterns
        - "https://api.docs/v1"   # URLs
        - "data/context.txt"      # Single files

      # Rich Resource Definitions
      resources:
        - type: "repository"      # Git repository
          url: "https://github.com/org/repo"
          branch: "main"
          paths: ["docs/", "README.md"]

        - type: "database"        # Database query results
          query: "SELECT * FROM docs"
          connection: "postgresql:///"

        - type: "cli"            # Command output
          command: "git log --oneline"
          shell: true

        - type: "source"         # Python source code
          module: "myapp.core"
          include_docstrings: true

        - type: "callable"       # Function results
          import_path: "myapp.data:get_context"
          refresh_interval: 3600  # seconds

      # Dynamic Content via Prompts
      prompts:
        - type: "file"          # From file
          path: "prompts/context.txt"

        - type: "dynamic"       # From Python function
          import_path: "myapp.prompts:generate_context"

      # Global Settings
      convert_to_markdown: true  # Convert content when possible
```

## Components

### Simple Paths
Quick access to files and URLs:
```yaml
knowledge:
  paths:
    - "docs/*.md"              # Local files
    - "https://docs.api/v1"    # Web content
    - "s3://bucket/data.json"  # Cloud storage
```

### Rich Resources
Detailed resource configurations:
```yaml
knowledge:
  resources:
    - type: "repository"      # Git repos
      url: "https://github.com/org/repo"

    - type: "database"        # Database content
      query: "SELECT * FROM docs"

    - type: "cli"            # Command output
      command: "git status"
```

### Dynamic Prompts
Content generated at load time:
```yaml
knowledge:
  prompts:
    - type: "dynamic"
      import_path: "myapp.prompts:generate"

```

## Key Differences to Environment

- **Knowledge**: Loaded at startup
  - Provides base context for the agent
  - Always available in memory
  - Used for foundational information
  - Loaded in defined order
  - Can affect agent behavior from start

- **Environment**: Loaded on-demand
  - Resources accessed by tools when needed
  - Loaded and potentially unloaded
  - Used for task-specific information
  - Accessed through tool calls
  - More efficient for large/rarely used data

## Examples

### Documentation Assistant
```yaml
agents:
  docs_helper:
    knowledge:
      paths: ["docs/**/*.md"]
      resources:
        - type: "repository"
          url: "https://github.com/org/docs"
      prompts:
        - type: "file"
          path: "prompts/docs_context.txt"
```

### Code Reviewer
```yaml
agents:
  reviewer:
    knowledge:
      paths: ["src/**/*.py"]
      resources:
        - type: "source"
          module: "myapp"
        - type: "cli"
          command: "git diff main"
```
