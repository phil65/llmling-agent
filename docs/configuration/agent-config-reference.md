---
title: Agent Config Reference
description: Complete agent configuration reference
icon: material/file-document
---

# Agent Configuration

Agent configurations are defined in YAML files and consist of three main sections: agents, responses, and roles.

## Agents Section

Complete example of an agent configuration:

```yaml
agents:
  web_assistant:                   # Name of the agent
    description: "Helps with web tasks"  # Optional description
    model: openai:gpt-5           # Model to use

    tools:
      open_browser:
        import_path: webbrowser.open
        description: "Opens URLs in browser"

    # Response type for structured output (optional)
    output_type: WebResult       # Must be defined in 'responses' section

    # Base behavior definition
    system_prompts:
      - "You are a web assistant."
      - "Use open_browser to open URLs."

    # Default prompts for testing
    user_prompts:
      - "Open Python website"

    # Advanced settings
    retries: 2                   # Number of retries for failed operations
    model_settings:              # Model-specific settings
      temperature: 0.7
      max_tokens: 1000
```

### Key Fields Explained

**model**
The language model to use. Can be:
- Simple string: `openai:gpt-5`
- Model name: `gpt-5`
- Structured model configuration (for testing/development)

**environment**
Defines available tools and resources. Two formats:
1. File reference: Path to LLMling environment file
2. Inline configuration: Complete environment defined in agent file

**system_prompts**
List of prompts that define the agent's behavior. These are sent to the model before user input and can include:
- Role definitions
- Tool usage instructions
- Response formatting requirements

**output_type**
Optional reference to a response type (defined in responses section) for structured output. Ensures the model returns data in a specific format.

## Responses Section

Complete example of response definitions:

```yaml
responses:
  WebResult:                     # Name of the response type
    response_schema:
      type: inline                # Can be 'inline' or 'import'
      description: "Web operation result"
      fields:                     # Field definitions
        success:
          type: bool
          description: "Whether operation succeeded"
        url:
          type: str
          description: "URL that was processed"

  AnalysisResult:
    type: import                # Use existing Pydantic model
    import_path: myapp.models.Analysis
    description: "Complex analysis result"
```

### Key Fields Explained

**type**
Determines how the response type is defined:
- `inline`: Define structure directly in YAML
- `import`: Use existing Pydantic model

**fields** (for inline types)
Define the structure of the response including:
- Field names and types
- Descriptions
- Optional constraints (min/max values, regex patterns, etc.)

## Roles Section

```yaml
roles:
  analyst:                      # Name of the role
    can_list_agents: false     # Agent discovery permission
    history_access: "own"      # History access level
```

> **Note**: The roles system is currently in development. The built-in roles
> (`overseer`, `specialist`, `assistant`) are available with predefined capabilities.


## Advanced Configuration Examples

### Complex Response Types

Define sophisticated structured outputs:

```yaml
responses:
  WebResult:
    response_schema:
      type: inline
      fields:
        success:
          type: bool
          description: "Whether operation succeeded"
        url:
          type: str
          description: "URL that was processed"
        error:
          type: str | None
          description: "Error message if failed"
        attempts:
          type: int
          description: "Number of attempts made"
          constraints:
            ge: 1
            le: 5
```

### Model Settings

Basic model configuration options:

```yaml
agents:
  assistant:
    model: openai:gpt-5
    model_settings:
      temperature: 0.7
      max_tokens: 2000
    retries: 3              # Number of retries for failed operations
```
