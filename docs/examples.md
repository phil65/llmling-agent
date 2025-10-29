# Examples

These examples demonstrate how to create and use agents through YAML configuration files.

## Simple Text Agent

Create a simple agent that opens websites in your browser:

```yaml
# agents.yml
agents:
  url_opener:
    model: openai:gpt-5-mini
    environment: env_web.yml
    system_prompts:
      - |
        You help users open websites. Use the open_url tool to open URLs.
        When given a website name, find its URL in the bookmarks resource.
        Always confirm what you're about to open.
    user_prompts:
      - "Open the Python website for me"
      - "Open my favorite coding sites"
```

```yaml
# env_web.yml
tools:
  open_url:
    import_path: webbrowser.open
    description: "Open URL in default browser"

resources:
  bookmarks:
    type: text
    content: |
      Python Website: https://python.org
      Documentation: https://docs.python.org
    description: "Common Python URLs"
```

Use the agent:
```bash
# Add configuration
llmling-agent add web-helper agents.yml

# Start chatting
llmling-agent chat url_opener

You: Open the Python website
Assistant: I'll open the Python website (https://python.org) for you.
[Opening browser...]

You: Open the documentation
Assistant: I'll open the Python documentation (https://docs.python.org) in your browser.
[Opening browser...]
```

Or programmatically:
```python
from llmling_agent import AgentPool

async with AgentPool("agents.yml") as pool:
    agent = pool.get_agent("url_opener")
    result = await agent.run("Open the Python website")
    print(result.data)
```

## Structured Responses

Define structured outputs for consistent response formats:

```yaml
# agents.yml
responses:
  CodeReview:
    response_schema:
      type: inline
      description: "Code review result"
      fields:
        issues:
          type: list[str]
          description: "Found issues"
        score:
          type: int
          description: "Quality score (0-100)"

agents:
  code_reviewer:
    model: openai:gpt-5
    output_type: CodeReview  # Use structured response
    environment: env_code.yml
    system_prompts:
      - "You review Python code and provide structured feedback."
```

## Tool Usage

Create an agent that interacts with the file system:

```yaml
# agents.yml
agents:
  file_manager:
    model: openai:gpt-5
    environment: env_files.yml
    system_prompts:
      - "You help users manage their files and directories."
      - |
        Available tools:
        - list_files: Show directory contents
        - read_file: Read file contents
        - file_info: Get file metadata

        Always confirm before modifying files.

```

```yaml
# env_files.yml
tools:
  list_files:
    import_path: os.listdir
    description: "List directory contents"

  read_file:
    import_path: builtins.open
    description: "Read file contents"

  file_info:
    import_path: os.stat
    description: "Get file information"

  delete_file:
    import_path: os.remove
    description: "Delete a file"

  modify_file:
    import_path: custom_tools.modify_file
    description: "Modify file contents"

resources:
  help_text:
    type: text
    content: |
      File Operations Help:
      - Use list_files to see contents
      - Use read_file to view contents
      - Use file_info for metadata
```

Use the file manager:
```python
from llmling_agent import Agent

async with AgentPool("agents.yml" as pool:
    agent = pool.get_agent("file_manager")
    # List files
    result = await agent.run("What files are in the current directory?")
    print(result.data)

    # Read a file
    result = await agent.run("Show me the contents of config.py")
    print(result.data)
```
