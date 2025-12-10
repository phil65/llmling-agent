---
title: System Prompts
description: Advanced system prompt features
icon: material/comment
---

# System Prompts

System prompts are a crucial part of agent configuration, defining the agent's role, behavior, and capabilities. LLMling provides flexible ways to manage and format system prompts.

## Basic Usage

The simplest way is to provide string prompts:

```python
agent = Agent(
    name="helper",
    system_prompt="You are a helpful assistant.",
)
```

Multiple prompts are concatenated with proper spacing:

```python
agent = Agent(
    system_prompt=[
        "You are a helpful assistant.",
        "You always provide concise answers.",
    ]
)
```

## Configuration Types

System prompts support four different configuration types:

### String Prompts (Shortcut)

Simple strings are automatically converted to static prompt configs:

```yaml
agents:
  helper:
    system_prompts:
      - "You are a helpful assistant."
      - "You provide concise answers."
```

### Static Prompt Configs

Explicit static prompt definitions:

```yaml
agents:
  helper:
    system_prompts:
      - type: static
        content: "You are a professional assistant."
      - type: static
        content: "Always maintain a helpful attitude."
```

### File-based Prompts

Load prompts from Jinja2 template files:

```yaml
agents:
  expert:
    system_prompts:
      - type: file
        path: "prompts/role.j2"
        variables:
          role: "code reviewer"
          language: "Python"
          experience: "senior"
```


### Library Reference Prompts

Reference prompts from the configured prompt library:

```yaml
prompts:
  system_prompts:
    expert_analyst:
      content: |
        You are an expert data analyst.
        Focus on finding patterns and insights in data.
        Always provide evidence for your conclusions.
      category: role

agents:
  analyst:
    system_prompts:
      - type: library
        reference: "expert_analyst"
```

### Function-generated Prompts

Generate prompts dynamically using callable functions:

```yaml
agents:
  dynamic_agent:
    system_prompts:
      - type: function
        function: "my_module:generate_context_prompt"
        arguments:
          user_id: "12345"
          session_type: "technical_support"
```

Function example:
```python
def generate_context_prompt(user_id: str, session_type: str) -> str:
    # Could fetch user preferences, history, etc.
    user_data = get_user_context(user_id)

    if session_type == "technical_support":
        return f"""
        You are providing technical support for {user_data.name}.
        User's technical level: {user_data.tech_level}
        Previous issues: {user_data.recent_issues}

        Adapt your communication style accordingly.
        """
    return f"You are assisting {user_data.name} with {session_type}."
```

## Dynamic System Prompts

System prompts can be dynamic, either evaluating once on first run or for each interaction:

```python
from llmling_agent import Agent

async def get_weather_context():
    weather = await fetch_weather()
    return f"Current weather: {weather}"

agent = Agent(
    name="weather_advisor",
    system_prompt=[
        "You are a weather advisor.",
        get_weather_context  # Updates each run
    ]
)
agent.sys_prompts.dynamic = True  # Already default
```

## Mixed Prompt Types

You can combine different prompt types in a single agent:

```yaml
agents:
  advanced_assistant:
    system_prompts:
      # String shortcut
      - "You are an advanced AI assistant."

      # Static config
      - type: static
        content: "You have access to real-time information."

      # File template
      - type: file
        path: "prompts/capabilities.j2"
        variables:
          version: "2.0"
          features: ["analysis", "generation", "reasoning"]

      # Library reference
      - type: library
        reference: "professional_tone"

      # Function-generated
      - type: function
        function: "context:get_current_capabilities"
        arguments:
          include_experimental: false
```

## Structured Prompts

You can use Pydantic models to create structured prompts:

```python
from pydantic import BaseModel

class PiratePersonality(BaseModel):
    behavior: str = "You are a fearsome pirate captain."
    goals: str = "Find treasure and command your crew."
    style: str = "Speak in pirate dialect, use nautical terms."

agent = Agent(
    system_prompt=PiratePersonality()
)
```

!!! info
    You can basically use any structured context or "dataclass-ish" objects
    from stdlib as well as dataclass-equivalents of many libraries as prompts.
    This also applies to the Agent.run() methods.

## Tool Integration

System prompts can include information about available tools:

```python
agent = Agent(
    name="log_analyzer",
    system_prompt=["Analyze system logs for issues"],
    tools=[read_logs, analyze_logs, report_issue]
)

# Make tools part of agent's core identity
agent.sys_prompts.inject_tools = "all"  # Include all enabled tools
agent.sys_prompts.tool_usage_style = "suggestive"  # or "strict"
```

This will automatically include tool descriptions in the system prompt:
```
You are log_analyzer. Analyze system logs for issues.

You have access to these tools:

- read_logs: Read system log files
- analyze_logs: Analyze logs for patterns
- report_issue: Create issue report

Use them when appropriate to complete your tasks.
```

For strict enforcement:
```python
agent.sys_prompts.inject_tools = "required"
agent.sys_prompts.tool_usage_style = "strict"
```

This changes the tone:
```
You MUST use these tools to complete your tasks:

- read_logs: Read system log files
- analyze_logs: Analyze logs for patterns
- report_issue: Create issue report

Do not attempt to perform tasks without using appropriate tools.
```

### 3. Temporary System prompts

For temporary system prompt changes, The system prompt manager provides an async context manager:

```python
# Temporarily use a different system prompt
with agent.sys_prompts.temporary_prompt(prompt):
    # temporary prompt additional to agent's prompts
    ...

# Temporarily add sys_prompts and disable all others
with agent.sys_prompts.temporary_prompt(prompt, exclusive=True):
    # Only prompt is used here
    ...

# Original prompts are restored after context exit
```

## Caching

Dynamic prompts can be cached to avoid unnecessary re-evaluation:

```python
# Cache after first evaluation
agent.sys_prompts.dynamic = False

# Always re-evaluate (default)
agent.sys_prompts.dynamic = True
```

## Agent Info Injection

System prompts can automatically include agent identity:

```python
agent = Agent(
    name="analyst",
    description="Expert in data analysis",
    system_prompt=["Analyze data thoroughly"]
)

# Control agent info injection
agent.sys_prompts.inject_agent_info = True  # Default
agent.sys_prompts.inject_agent_info = False  # Disable
```

## Custom Templates

While rarely needed, you can customize the complete template used to generate the system prompt.

```python
agent.sys_prompts.template = custom_template
```

In the global spacename, you will have

- the `agent`
- the `prompts` in form of `AnyPromptType`.
- `to_prompt` helper (AnyPromptType -> str)
- the `dynamic` setting (bool)
- the `inject_tools` setting (bool)
- the `tool_usage_style` setting (Literal["suggestive", "strict"])

For further information, check out agent/sys_prompts.py in the codebase.

!!! info "About Prompt Engineering"
    By default, LLMling-Agent does not engage in prompt engineering or manipulation. The features described
    above (tool injection, strict mode, etc.) are strictly opt-in. We believe in transparency and
    explicit control - any modifications to system prompts are clearly visible and configurable.
    We do include agent name and description though because we consider this essential
    for proper coordination and context. You can disable this behavior by setting `inject_agent_info=False`.

## Prompt Library

LLMling includes a library of pre-defined system prompts that can be used across agents. These prompts are organized by type:

### Prompt Types

- `role`: Defines WHO the agent is (identity, expertise, personality)
- `methodology`: Defines HOW the agent approaches tasks (process, methods)
- `tone`: Defines the STYLE of communication (language, attitude)
- `format`: Defines OUTPUT STRUCTURE (not content)

### Using Library Prompts

You can reference prompts from the library:

```python
# Add a single library prompt by reference
agent.sys_prompts.add("step_by_step")

# In YAML configuration:
agents:
  my_agent:
    model: gpt-5
    system_prompts:
      - "You are a helpful assistant"
      - type: library
        reference: step_by_step
      - type: library
        reference: professional
```

### Defining Library Prompts

```yaml
prompts:
  system_prompts:
    expert_analyst:
      content: |
        You are an expert data analyst.
        Focus on finding patterns and insights in data.
        Always provide evidence for your conclusions.
      category: role

    step_by_step:
      content: |
        Break tasks into clear, sequential steps.
        For each step:
        1. Explain what to do
        2. Note important considerations
        3. Define completion criteria
      category: methodology
```

!!! tip "Organizing Prompts"
    It's recommended to keep prompt libraries in separate files and use YAML inheritance
    to include them. This keeps your agent configurations clean and promotes reuse:
    ```yaml
    # prompts.yml
    prompts:
      system_prompts:
        my_prompt:
          content: ...
          category: role

    # agents.yml
    INHERIT: prompts.yml
    agents:
      my_agent:
        system_prompts:
          - type: library
            reference: my_prompt
    ```

### Available Prompts

By default, INHERIT is set to builtin prompt library with a few silly prompts to get started.
These can all be referenced by name without any further configuration.

Roles:

- `technical_writer`: Expert in clear, precise documentation
- `code_reviewer`: Expert code analysis and feedback
- `rubber_duck`: Debugging assistant with personality

Methodologies:

- `step_by_step`: Break tasks into clear sequences
- `minimalist`: Concise, essential responses
- `detailed`: Comprehensive coverage of topics

Tones:

- `professional`: Formal business communication
- `pirate`: Maritime flavor (fun!)
- `shakespearean`: Classical, poetic style

Formats:

- `markdown`: Structured Markdown formatting

## File Organization

Keep your configuration organized:

```
project/
├── config.yml
├── prompts/
│   ├── roles/
│   │   ├── expert.j2
│   │   └── specialist.j2
│   ├── styles/
│   │   ├── formal.j2
│   │   └── casual.j2
│   └── methodologies/
│       └── step_by_step.j2
└── functions/
    └── prompt_generators.py
```

## Best Practices

1. **Use String Shortcuts**: For simple, static prompts
2. **File Templates**: For complex, reusable prompts with variables
3. **Library References**: For team consistency and prompt management
4. **Function Generation**: For dynamic, context-aware prompts
5. **Mixed Approaches**: Combine types as needed for flexibility
6. **Clear Organization**: Separate prompts logically
7. **Version Control**: Include templates in version control
8. **Documentation**: Document variables and function signatures

## Function Limitations

!!! info "Async Function Support"
    Currently, function-generated prompts only support synchronous functions.
    Async function support is planned for future releases when the agent
    initialization process becomes asynchronous.
