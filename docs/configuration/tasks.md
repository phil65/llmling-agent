---
title: Tasks
description: Task definition and configuration
icon: material/format-list-checks
---

# Task Configuration

Tasks (called "jobs" in YAML) define reusable operations that agents can execute. They can be defined in YAML and include:

- Prompt templates
- Required tools
- Knowledge sources
- Result type validation

## Basic Task

Simple task with prompt and result type:

```yaml
jobs:
  analyze_code:
    prompt: "Analyze the code in src directory for potential improvements"
    required_return_type: "myapp.types:AnalysisResult"
    description: "Analyze code quality and suggest improvements"
```

## Knowledge Sources

Tasks can load context from various sources:

```yaml
jobs:
  summarize_docs:
    prompt: "Summarize the documentation changes"
    output_type: "myapp.types.DocSummary"
    knowledge:
      # Simple file paths
      paths:
        - "docs/**/*.md"
        - "README.md"

      # Rich resource definitions
      resources:
        - type: "cli"
          command: "git diff --name-only"
        - type: "repository"
          url: "https://github.com/user/repo"
          paths: ["docs/"]

      # Context prompts
      prompts:
        - "Consider the project's documentation standards..."
```

## Required Tools

Specify tools needed for the task:

```yaml
jobs:
  security_audit:
    prompt: "Perform security audit on the codebase"
    output_type: "myapp.types.AuditReport"
    tools:
      - "analyze_dependencies"
      - "check_vulnerabilities"
      - name: "custom_scanner"
        import_path: "myapp.tools.security.scan_code"
        description: "Custom security scanner"
```

## Dependencies and Context

Tasks can require specific data:

```yaml
jobs:
  review_pr:
    prompt: "Review the pull request changes"
    required_return_type: "myapp.types:ReviewResult"
    required_dependency: "myapp.types:PRContext"  # Type hint for required context
    min_context_tokens: 1000       # Minimum context window size
    requires_vision: false         # Whether vision capability is needed
```

## Using Tasks

Execute tasks through the API:

```python
# Python API
result = await agent.run_job("analyze_code")
```

You can also use the Job directly:

```python
from llmling_agent import AgentPool

# Get job definition
job = pool.get_job("analyze_code")

# Execute job with specific agent
result = await pool.run_job("analyze_code", agent="my-agent")
```

## Complete Example

Complex task with all features:

```yaml
jobs:
  deep_code_review:
    description: "Perform comprehensive code review"
    prompt: "Review the code changes focusing on:"
    required_return_type: "myapp.types:ReviewResult"
    required_dependency: "myapp.types:CodeContext"
    requires_vision: false

    # Required knowledge
    knowledge:
      paths: ["src/**/*.py"]
      resources:
        - type: "cli"
          command: "git diff main"
        - type: "repository"
          url: "https://github.com/user/repo"
      prompts:
        - "Consider these coding standards..."
        - "Focus on performance aspects..."

    # Required tools
    tools:
      - "analyze_complexity"
      - "check_style"
      - import_path: "myapp.tools:metrics"

    # Context requirements
    min_context_tokens: 2000
```

## Task Registry

All tasks are available through the agent pool:

```python
# Get job definition
job = pool.get_job("analyze_code")

# Register new task 
from llmling_agent_config.task import Job
job = Job(
    name="new_task",
    description="A new task",
    prompt="Do something interesting",
    required_return_type="myapp.types:Result"
)
pool.register_job("new_task", job)

# List available tasks
jobs = pool.list_jobs()
```
